package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/zzet/gortex/internal/graph"
)

// ---------------------------------------------------------------------------
// fakeLSPServer — a handler-driven JSON-RPC router that impersonates a
// language server. Tests register canned responses per method; notifications
// are logged silently.
// ---------------------------------------------------------------------------

type fakeLSPServer struct {
	handlers             map[string]func(params json.RawMessage) (any, *jsonRPCError)
	notificationHandlers map[string]func(params json.RawMessage)

	mu       sync.Mutex
	notifLog []string
}

func newFakeLSPServer() *fakeLSPServer {
	return &fakeLSPServer{
		handlers:             make(map[string]func(json.RawMessage) (any, *jsonRPCError)),
		notificationHandlers: make(map[string]func(json.RawMessage)),
	}
}

func (f *fakeLSPServer) handle(method string, fn func(params json.RawMessage) (any, *jsonRPCError)) {
	f.handlers[method] = fn
}

func (f *fakeLSPServer) handleNotification(method string, fn func(params json.RawMessage)) {
	f.notificationHandlers[method] = fn
}

func (f *fakeLSPServer) notifications() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.notifLog))
	copy(out, f.notifLog)
	return out
}

// run consumes framed JSON-RPC messages from in, dispatches to handlers, and
// writes framed responses to out. Returns when in EOFs.
func (f *fakeLSPServer) run(in *bufio.Reader, out io.Writer) {
	for {
		body, ok := readFramed(in)
		if !ok {
			return
		}

		var probe struct {
			Method string `json:"method"`
			ID     *int64 `json:"id,omitempty"`
		}
		if err := json.Unmarshal(body, &probe); err != nil {
			continue
		}

		if probe.ID == nil {
			// Notification — log and continue.
			f.mu.Lock()
			f.notifLog = append(f.notifLog, probe.Method)
			f.mu.Unlock()
			if h, ok := f.notificationHandlers[probe.Method]; ok {
				var notif struct {
					Params json.RawMessage `json:"params"`
				}
				if err := json.Unmarshal(body, &notif); err == nil {
					h(notif.Params)
				}
			}
			continue
		}

		var req struct {
			ID     int64           `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			continue
		}

		resp := jsonRPCResponse{JSONRPC: "2.0", ID: req.ID}
		if h, ok := f.handlers[req.Method]; ok {
			result, errResp := h(req.Params)
			if errResp != nil {
				resp.Error = errResp
			} else if result != nil {
				raw, err := json.Marshal(result)
				if err != nil {
					resp.Error = &jsonRPCError{Code: -32603, Message: "marshal: " + err.Error()}
				} else {
					resp.Result = raw
				}
			}
		} else {
			resp.Result = json.RawMessage("null")
		}

		data, err := json.Marshal(resp)
		if err != nil {
			continue
		}
		_, _ = fmt.Fprintf(out, "Content-Length: %d\r\n\r\n", len(data))
		_, _ = out.Write(data)
	}
}

// providerWithFakeServer returns a Provider whose internal client is wired to a
// running fakeLSPServer. The cleanup func tears down pipes and goroutines.
func providerWithFakeServer(t *testing.T, server *fakeLSPServer, languages []string) (*Provider, func()) {
	t.Helper()
	c, serverIn, serverOut, cleanup := newPipedClient(t)

	go server.run(serverIn, serverOut)

	p := NewProvider("fake-lsp", nil, languages, false, 0, 0, zap.NewNop())
	p.client = c // skip ensureClient — the client is already wired.
	return p, cleanup
}

// ---------------------------------------------------------------------------
// Tests for Provider.Enrich orchestration.
// ---------------------------------------------------------------------------

func TestLSP_LanguageID_UsesReactVariants(t *testing.T) {
	assert.Equal(t, "typescriptreact", lspLanguageID("src/App.tsx", []string{"typescript"}))
	assert.Equal(t, "javascriptreact", lspLanguageID("src/App.jsx", []string{"javascript"}))
	assert.Equal(t, "typescript", lspLanguageID("src/main.ts", []string{"javascript"}))
}

func TestLSP_DiskRelPath_StripsRepoPrefixWhenNeeded(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(repoRoot, "apps", "web"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(repoRoot, "apps", "web", "main.ts"), []byte("export const x = 1\n"), 0o644))

	assert.Equal(t, "apps/web/main.ts", diskRelPath(repoRoot, "kmono/apps/web/main.ts"))
	assert.Equal(t, "apps/web/main.ts", diskRelPath(repoRoot, "apps/web/main.ts"))
}

func TestLSP_Provider_EnrichesNodeMetaFromHover(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.go"),
		[]byte("package main\n\nfunc F() string { return \"hi\" }\n"),
		0o644,
	))

	server := newFakeLSPServer()
	gotHoverPosition := make(chan Position, 1)
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		var hoverParams HoverParams
		_ = json.Unmarshal(params, &hoverParams)
		gotHoverPosition <- hoverParams.Position
		return HoverResult{
			Contents: MarkupContent{Kind: "plaintext", Value: "func F() string"},
		}, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{
		ID: "main.go::F", Kind: graph.KindFunction, Name: "F",
		FilePath: "main.go", StartLine: 3, EndLine: 3, Language: "go",
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	node := g.GetNode("main.go::F")
	require.NotNil(t, node.Meta)
	assert.Equal(t, "func F() string", node.Meta["semantic_type"])
	assert.Equal(t, "lsp-fake-lsp", node.Meta["semantic_source"])

	select {
	case pos := <-gotHoverPosition:
		assert.Equal(t, Position{Line: 2, Character: 5}, pos)
	default:
		t.Fatal("fake server did not receive hover request")
	}

	// didOpen should have been sent for main.go.
	assert.Contains(t, server.notifications(), "textDocument/didOpen")
}

func TestLSP_Provider_AddsImplementsFromLSP(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.go"),
		[]byte("package main\n\ntype Greeter interface { Greet() string }\n\ntype EnglishGreeter struct{}\n"),
		0o644,
	))

	server := newFakeLSPServer()
	server.handle("textDocument/implementation", func(params json.RawMessage) (any, *jsonRPCError) {
		return []Location{
			{
				URI: pathToURI(filepath.Join(repoRoot, "main.go")),
				// Implementation site at line 5 (1-indexed). LSP uses 0-indexed,
				// so we send Line: 4. Provider adds 1 back when matching.
				Range: Range{Start: Position{Line: 4, Character: 5}, End: Position{Line: 4, Character: 19}},
			},
		}, nil
	})
	// Hover is queried for every node; return nothing useful.
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		return nil, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{
		ID: "main.go::Greeter", Kind: graph.KindInterface, Name: "Greeter",
		FilePath: "main.go", StartLine: 3, EndLine: 3, Language: "go",
	})
	g.AddNode(&graph.Node{
		ID: "main.go::EnglishGreeter", Kind: graph.KindType, Name: "EnglishGreeter",
		FilePath: "main.go", StartLine: 5, EndLine: 5, Language: "go",
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	edges := g.GetOutEdges("main.go::EnglishGreeter")
	var impl *graph.Edge
	for _, e := range edges {
		if e.Kind == graph.EdgeImplements && e.To == "main.go::Greeter" {
			impl = e
			break
		}
	}
	require.NotNil(t, impl, "expected EdgeImplements from EnglishGreeter to Greeter")
	assert.Equal(t, 1.0, impl.Confidence)
	assert.Equal(t, graph.OriginLSPDispatch, impl.Origin)
	assert.Equal(t, "lsp-fake-lsp", impl.Meta["semantic_source"])
}

func TestLSP_Provider_ConfirmsEdgeFromMatchingReferences(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.go"),
		[]byte("package main\n\nfunc Hello() string { return \"\" }\n\nfunc Caller() { _ = Hello() }\n"),
		0o644,
	))

	server := newFakeLSPServer()
	// References for Hello at line 3 returns one ref inside Caller (lines 5-5).
	server.handle("textDocument/references", func(params json.RawMessage) (any, *jsonRPCError) {
		return []Location{
			{
				URI:   pathToURI(filepath.Join(repoRoot, "main.go")),
				Range: Range{Start: Position{Line: 4, Character: 22}, End: Position{Line: 4, Character: 27}},
			},
		}, nil
	})
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		return nil, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{
		ID: "main.go::Hello", Kind: graph.KindFunction, Name: "Hello",
		FilePath: "main.go", StartLine: 3, EndLine: 3, Language: "go",
	})
	g.AddNode(&graph.Node{
		ID: "main.go::Caller", Kind: graph.KindFunction, Name: "Caller",
		FilePath: "main.go", StartLine: 5, EndLine: 5, Language: "go",
	})
	// Pre-seed an INFERRED call edge that LSP should confirm.
	g.AddEdge(&graph.Edge{
		From: "main.go::Caller", To: "main.go::Hello", Kind: graph.EdgeCalls,
		Confidence: 0.7, ConfidenceLabel: "INFERRED", Origin: graph.OriginASTInferred,
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	edges := g.GetOutEdges("main.go::Caller")
	var confirmed *graph.Edge
	for _, e := range edges {
		if e.To == "main.go::Hello" && e.Kind == graph.EdgeCalls {
			confirmed = e
			break
		}
	}
	require.NotNil(t, confirmed)
	assert.Equal(t, 1.0, confirmed.Confidence)
	assert.Equal(t, "EXTRACTED", confirmed.ConfidenceLabel)
	assert.Equal(t, graph.OriginLSPResolved, confirmed.Origin)
	assert.Equal(t, "lsp-fake-lsp", confirmed.Meta["semantic_source"])
}

func TestLSP_Provider_DoesNotConfirmWhenReferencesDontMatchCaller(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.go"),
		[]byte("package main\n\nfunc Hello() string { return \"\" }\n\nfunc Caller() { _ = Hello() }\n\nfunc Other() {}\n"),
		0o644,
	))

	server := newFakeLSPServer()
	// Reference falls inside neither Caller's range — so no confirmation.
	server.handle("textDocument/references", func(params json.RawMessage) (any, *jsonRPCError) {
		return []Location{
			{
				URI:   pathToURI(filepath.Join(repoRoot, "main.go")),
				Range: Range{Start: Position{Line: 99, Character: 0}, End: Position{Line: 99, Character: 5}},
			},
		}, nil
	})
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		return nil, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{
		ID: "main.go::Hello", Kind: graph.KindFunction, Name: "Hello",
		FilePath: "main.go", StartLine: 3, EndLine: 3, Language: "go",
	})
	g.AddNode(&graph.Node{
		ID: "main.go::Caller", Kind: graph.KindFunction, Name: "Caller",
		FilePath: "main.go", StartLine: 5, EndLine: 5, Language: "go",
	})
	g.AddEdge(&graph.Edge{
		From: "main.go::Caller", To: "main.go::Hello", Kind: graph.EdgeCalls,
		Confidence: 0.7, ConfidenceLabel: "INFERRED", Origin: graph.OriginASTInferred,
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	edges := g.GetOutEdges("main.go::Caller")
	var seen *graph.Edge
	for _, e := range edges {
		if e.To == "main.go::Hello" && e.Kind == graph.EdgeCalls {
			seen = e
			break
		}
	}
	require.NotNil(t, seen)
	assert.InDelta(t, 0.7, seen.Confidence, 0.001, "confidence should not be upgraded when refs don't match")
	assert.Equal(t, "INFERRED", seen.ConfidenceLabel)
}

func TestLSP_Provider_FiltersByLanguage(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.py"),
		[]byte("def f(): return 1\n"),
		0o644,
	))

	hoverCalls := 0
	server := newFakeLSPServer()
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		hoverCalls++
		return nil, nil
	})

	// Provider supports only Go.
	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	// Python node — should be ignored entirely.
	g.AddNode(&graph.Node{
		ID: "main.py::f", Kind: graph.KindFunction, Name: "f",
		FilePath: "main.py", StartLine: 1, EndLine: 1, Language: "python",
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	assert.Zero(t, hoverCalls, "hover should not be queried for non-matching languages")
	assert.NotContains(t, server.notifications(), "textDocument/didOpen",
		"non-matching languages should not trigger didOpen")
}

func TestLSP_Provider_OpensEachFileOnce(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "shared.go"),
		[]byte("package shared\n\nfunc A() {}\nfunc B() {}\nfunc C() {}\n"),
		0o644,
	))

	server := newFakeLSPServer()
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		return nil, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	for i, name := range []string{"A", "B", "C"} {
		g.AddNode(&graph.Node{
			ID:        "shared.go::" + name,
			Kind:      graph.KindFunction,
			Name:      name,
			FilePath:  "shared.go",
			StartLine: 3 + i,
			EndLine:   3 + i,
			Language:  "go",
		})
	}

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	openCount := 0
	for _, n := range server.notifications() {
		if n == "textDocument/didOpen" {
			openCount++
		}
	}
	assert.Equal(t, 1, openCount,
		"didOpen should be sent exactly once for shared.go even with 3 nodes inside it")
}

func TestLSP_Provider_EnrichFileIsNoOp(t *testing.T) {
	server := newFakeLSPServer()
	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	result, err := p.EnrichFile(g, t.TempDir(), "main.go")
	require.NoError(t, err)
	// Current contract: EnrichFile is a no-op (see provider.go:256). Returning
	// nil is the documented behavior; this test pins that down so any future
	// change is a deliberate decision.
	assert.Nil(t, result)
}

func TestLSP_Provider_EnrichSurvivesHoverFailures(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(repoRoot, "main.go"),
		[]byte("package main\n\nfunc F() {}\nfunc G() {}\n"),
		0o644,
	))

	hoverCalls := 0
	server := newFakeLSPServer()
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		hoverCalls++
		// First hover fails; subsequent ones succeed with usable type info.
		if hoverCalls == 1 {
			return nil, &jsonRPCError{Code: -32603, Message: "internal error"}
		}
		return HoverResult{Contents: MarkupContent{Kind: "plaintext", Value: "func G()"}}, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"go"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{
		ID: "main.go::F", Kind: graph.KindFunction, Name: "F",
		FilePath: "main.go", StartLine: 3, EndLine: 3, Language: "go",
	})
	g.AddNode(&graph.Node{
		ID: "main.go::G", Kind: graph.KindFunction, Name: "G",
		FilePath: "main.go", StartLine: 4, EndLine: 4, Language: "go",
	})

	done := make(chan error, 1)
	go func() {
		_, err := p.Enrich(g, repoRoot)
		done <- err
	}()
	select {
	case err := <-done:
		require.NoError(t, err, "Enrich must not return error when individual hover calls fail")
	case <-time.After(3 * time.Second):
		t.Fatal("Enrich timed out")
	}

	// At least one of the two nodes should have been enriched (the one whose
	// hover succeeded). Map iteration order isn't deterministic, so check that
	// at least one carries semantic_type.
	enriched := 0
	for _, id := range []string{"main.go::F", "main.go::G"} {
		n := g.GetNode(id)
		if n != nil && n.Meta != nil {
			if _, ok := n.Meta["semantic_type"]; ok {
				enriched++
			}
		}
	}
	assert.GreaterOrEqual(t, enriched, 1, "at least one node should have been enriched despite a failed hover")
}

func TestLSP_Provider_SkipsUnsupportedExtensions(t *testing.T) {
	repoRoot := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(repoRoot, "theme.css"), []byte(".x { color: red; }\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(repoRoot, "main.ts"), []byte("export function ok() { return 1 }\n"), 0o644))

	opened := make(chan string, 2)
	server := newFakeLSPServer()
	server.handleNotification("textDocument/didOpen", func(params json.RawMessage) {
		var openParams DidOpenTextDocumentParams
		_ = json.Unmarshal(params, &openParams)
		opened <- openParams.TextDocument.URI
	})
	server.handle("textDocument/hover", func(params json.RawMessage) (any, *jsonRPCError) {
		return HoverResult{Contents: MarkupContent{Kind: "plaintext", Value: "function ok(): number"}}, nil
	})

	p, cleanup := providerWithFakeServer(t, server, []string{"typescript", "javascript"})
	defer cleanup()

	g := graph.New()
	g.AddNode(&graph.Node{ID: "theme.css::bad", Kind: graph.KindFunction, Name: "bad", FilePath: "theme.css", StartLine: 1, EndLine: 1, Language: "typescript"})
	g.AddNode(&graph.Node{ID: "main.ts::ok", Kind: graph.KindFunction, Name: "ok", FilePath: "main.ts", StartLine: 1, EndLine: 1, Language: "typescript"})

	res, err := p.Enrich(g, repoRoot)
	require.NoError(t, err)
	assert.Equal(t, 1, res.SymbolsTotal)
	assert.Nil(t, g.GetNode("theme.css::bad").Meta)
	assert.NotNil(t, g.GetNode("main.ts::ok").Meta)

	select {
	case uri := <-opened:
		assert.Contains(t, uri, "main.ts")
		assert.NotContains(t, uri, "theme.css")
	case <-time.After(time.Second):
		t.Fatal("expected didOpen for supported file")
	}
	select {
	case uri := <-opened:
		assert.NotContains(t, uri, "theme.css")
	default:
	}
}
