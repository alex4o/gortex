package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestLSP_PathToURI(t *testing.T) {
	uri := pathToURI("/repo/main.go")
	assert.Equal(t, "file:///repo/main.go", uri)
}

func TestLSP_URIToPath(t *testing.T) {
	tests := []struct {
		name string
		uri  string
		root string
		want string
	}{
		{"inside root", "file:///repo/pkg/a/a.go", "/repo", "pkg/a/a.go"},
		{"outside root returns empty", "file:///elsewhere/x.go", "/repo", ""},
		{"malformed uri returns empty", "://broken", "/repo", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, uriToPath(tt.uri, tt.root))
		})
	}
}

func TestLSP_ExtractTypeFromHover(t *testing.T) {
	tests := []struct {
		name  string
		hover string
		want  string
	}{
		{"go func with code fence", "```go\nfunc Hello() string\n```", "func Hello() string"},
		{"plain func", "func Hello() string", "func Hello() string"},
		{"type", "type Greeter interface { Greet() string }", "type Greeter interface { Greet() string }"},
		{"var", "var Counter int", "var Counter int"},
		{"const", "const Pi = 3.14", "const Pi = 3.14"},
		{"short type without space", "*Foo", "*Foo"},
		{"long prose returns empty", "This is a long English sentence describing something.", ""},
		{"empty hover", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractTypeFromHover(tt.hover))
		})
	}
}

func TestLSP_JSONRPCError_Error(t *testing.T) {
	err := &jsonRPCError{Code: -32601, Message: "Method not found"}
	assert.Equal(t, "LSP error -32601: Method not found", err.Error())
}

func TestLSP_Provider_Available_FalseWhenMissing(t *testing.T) {
	p := NewProvider("definitely-not-a-real-lsp-binary-xyz", nil, []string{"go"}, false, 0, 0, zap.NewNop())
	assert.False(t, p.Available())
	assert.Equal(t, "lsp-definitely-not-a-real-lsp-binary-xyz", p.Name())
}

func TestLSP_NewClient_FailsForBadCommand(t *testing.T) {
	_, err := NewClient("/nonexistent/path/to/lsp", nil, t.TempDir(), zap.NewNop(), time.Second)
	require.Error(t, err)
}

func TestLSP_Provider_UsesConfiguredTimeout(t *testing.T) {
	p := NewProvider("fake-lsp", nil, []string{"typescript"}, true, 0, 7, zap.NewNop())
	assert.Equal(t, 7*time.Second, p.requestTimeout)

	p = NewProvider("fake-lsp", nil, []string{"typescript"}, true, 0, 0, zap.NewNop())
	assert.Equal(t, 120*time.Second, p.requestTimeout)
}

// ---------------------------------------------------------------------------
// Client framing / request-response round-trip via io.Pipe pairs.
// These tests exercise the JSON-RPC plumbing without requiring a real LSP
// server subprocess.
// ---------------------------------------------------------------------------

// readFramed reads one Content-Length-framed JSON-RPC message off r. Returns
// (nil, false) on EOF so server-side goroutines can exit cleanly when the
// client closes the pipe at test teardown.
func readFramed(r *bufio.Reader) ([]byte, bool) {
	contentLength := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, false
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			contentLength, _ = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "Content-Length:")))
		}
	}
	if contentLength < 0 {
		return nil, false
	}
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, false
	}
	return body, true
}

// writeFramed writes a JSON-RPC message with Content-Length framing to w.
func writeFramed(t *testing.T, w io.Writer, payload any) {
	t.Helper()
	data, err := json.Marshal(payload)
	require.NoError(t, err)
	_, err = fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(data))
	require.NoError(t, err)
	_, err = w.Write(data)
	require.NoError(t, err)
}

// newPipedClient builds a Client whose stdin/stdout are connected to in-memory
// pipes. The returned write-end-of-server-input and read-end-of-server-output
// let the test impersonate an LSP server.
func newPipedClient(t *testing.T) (client *Client, serverIn *bufio.Reader, serverOut io.Writer, cleanup func()) {
	t.Helper()
	clientStdinR, clientStdinW := io.Pipe() // client writes → server reads
	clientStdoutR, clientStdoutW := io.Pipe()

	c := &Client{
		stdin:  clientStdinW,
		stdout: bufio.NewReader(clientStdoutR),
		logger: zap.NewNop(),
		done:   make(chan struct{}),
	}
	go c.readResponses()

	cleanup = func() {
		_ = clientStdinW.Close()
		_ = clientStdoutW.Close()
		_ = clientStdinR.Close()
		_ = clientStdoutR.Close()
		// Unblock any pending Call invocations.
		select {
		case <-c.done:
		default:
			close(c.done)
		}
	}
	return c, bufio.NewReader(clientStdinR), clientStdoutW, cleanup
}

func TestLSP_Client_CallRoundTrip(t *testing.T) {
	c, serverIn, serverOut, cleanup := newPipedClient(t)
	defer cleanup()

	// Fake server: read one request, echo "pong" back as the result.
	serverDone := make(chan error, 1)
	go func() {
		body, ok := readFramed(serverIn)
		if !ok {
			serverDone <- fmt.Errorf("server: unexpected EOF")
			return
		}
		var req jsonRPCRequest
		if err := json.Unmarshal(body, &req); err != nil {
			serverDone <- err
			return
		}
		writeFramed(t, serverOut, jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`"pong"`),
		})
		serverDone <- nil
	}()

	var result string
	err := c.Call("test/echo", map[string]string{"msg": "ping"}, &result)
	require.NoError(t, err)
	assert.Equal(t, "pong", result)

	select {
	case err := <-serverDone:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("fake server did not finish in time")
	}
}

func TestLSP_Client_CallReturnsError(t *testing.T) {
	c, serverIn, serverOut, cleanup := newPipedClient(t)
	defer cleanup()

	go func() {
		body, ok := readFramed(serverIn)
		if !ok {
			return
		}
		var req jsonRPCRequest
		_ = json.Unmarshal(body, &req)
		writeFramed(t, serverOut, jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &jsonRPCError{Code: -32601, Message: "method not found"},
		})
	}()

	err := c.Call("unknown/method", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "method not found")
	assert.Contains(t, err.Error(), "-32601")
}

func TestLSP_Client_CallTimesOut(t *testing.T) {
	c, serverIn, _, cleanup := newPipedClient(t)
	defer cleanup()
	c.requestTimeout = 20 * time.Millisecond

	// Drain the request but deliberately never answer it.
	go func() {
		_, _ = readFramed(serverIn)
	}()

	start := time.Now()
	err := c.Call("test/hangs", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timed out")
	assert.Contains(t, err.Error(), "test/hangs")
	assert.Less(t, time.Since(start), time.Second)
}

func TestLSP_Client_CallUnblocksOnServerExit(t *testing.T) {
	c, serverIn, _, cleanup := newPipedClient(t)
	defer cleanup()

	// Drain whatever the client writes so the io.Pipe send doesn't block.
	go func() {
		for {
			if _, ok := readFramed(serverIn); !ok {
				return
			}
		}
	}()

	// Close c.done to simulate the server having exited.
	close(c.done)

	err := c.Call("test/method", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exited")
}

func TestLSP_Client_AnswersServerRequests(t *testing.T) {
	_, serverIn, serverOut, cleanup := newPipedClient(t)
	defer cleanup()

	serverDone := make(chan error, 1)
	go func() {
		writeFramed(t, serverOut, map[string]any{
			"jsonrpc": "2.0",
			"id":      int64(99),
			"method":  "workspace/configuration",
			"params":  map[string]any{"items": []any{map[string]any{"section": "typescript"}}},
		})
		body, ok := readFramed(serverIn)
		if !ok {
			serverDone <- fmt.Errorf("server: unexpected EOF")
			return
		}
		var resp jsonRPCResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			serverDone <- err
			return
		}
		if resp.ID != 99 || string(resp.Result) != "[null]" {
			serverDone <- fmt.Errorf("unexpected response: id=%d result=%s", resp.ID, string(resp.Result))
			return
		}
		serverDone <- nil
	}()

	select {
	case err := <-serverDone:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("client did not answer server request")
	}
}

func TestLSP_Client_NotifyDoesNotWaitForResponse(t *testing.T) {
	c, serverIn, _, cleanup := newPipedClient(t)
	defer cleanup()

	gotMethod := make(chan string, 1)
	go func() {
		body, ok := readFramed(serverIn)
		if !ok {
			return
		}
		var notif jsonRPCNotification
		_ = json.Unmarshal(body, &notif)
		gotMethod <- notif.Method
	}()

	err := c.Notify("textDocument/didOpen", map[string]string{"uri": "file:///x.go"})
	require.NoError(t, err)

	select {
	case method := <-gotMethod:
		assert.Equal(t, "textDocument/didOpen", method)
	case <-time.After(2 * time.Second):
		t.Fatal("fake server did not receive the notification")
	}
}

func TestLSP_Client_CallAssignsUniqueIDs(t *testing.T) {
	c, serverIn, serverOut, cleanup := newPipedClient(t)
	defer cleanup()

	// Server: echo each request's ID back as the result.
	go func() {
		for {
			body, ok := readFramed(serverIn)
			if !ok {
				return
			}
			var req jsonRPCRequest
			if err := json.Unmarshal(body, &req); err != nil {
				return
			}
			writeFramed(t, serverOut, jsonRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result:  json.RawMessage(strconv.FormatInt(req.ID, 10)),
			})
		}
	}()

	var first, second int64
	require.NoError(t, c.Call("a", nil, &first))
	require.NoError(t, c.Call("b", nil, &second))

	assert.NotZero(t, first)
	assert.Equal(t, first+1, second, "request IDs should monotonically increase")
}

// pathToURI on a relative path delegates to filepath.Abs and prefixes "file://".
// Verify that joining a known absolute path keeps the URI parseable round-trip.
func TestLSP_PathToURI_RoundTrip(t *testing.T) {
	abs, err := filepath.Abs(".")
	require.NoError(t, err)
	uri := pathToURI(abs)
	assert.True(t, strings.HasPrefix(uri, "file://"))
	got := uriToPath(uri, abs)
	assert.Equal(t, ".", got)
}
