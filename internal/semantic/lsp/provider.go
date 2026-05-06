package lsp

import (
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/zzet/gortex/internal/graph"
	"github.com/zzet/gortex/internal/semantic"
)

// Provider uses an LSP server for on-demand semantic queries.
type Provider struct {
	command        string
	args           []string
	languages      []string
	daemon         bool
	maxParallel    int
	requestTimeout time.Duration
	logger         *zap.Logger

	runMu      sync.Mutex
	client     *Client
	clientRoot string
}

// NewProvider creates an LSP provider.
func NewProvider(command string, args []string, languages []string, daemon bool, maxParallel, timeoutSec int, logger *zap.Logger) *Provider {
	if maxParallel <= 0 {
		maxParallel = 10
	}
	if timeoutSec <= 0 {
		timeoutSec = 120
	}
	return &Provider{
		command:        command,
		args:           args,
		languages:      languages,
		daemon:         daemon,
		maxParallel:    maxParallel,
		requestTimeout: time.Duration(timeoutSec) * time.Second,
		logger:         logger,
	}
}

func (p *Provider) Name() string        { return "lsp-" + p.command }
func (p *Provider) Languages() []string { return p.languages }

func (p *Provider) Available() bool {
	_, err := exec.LookPath(p.command)
	return err == nil
}

func (p *Provider) Close() error {
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.client != nil {
		err := p.client.Shutdown()
		p.client = nil
		p.clientRoot = ""
		return err
	}
	return nil
}

func (p *Provider) Enrich(g *graph.Graph, repoRoot string) (*semantic.EnrichResult, error) {
	p.runMu.Lock()
	defer p.runMu.Unlock()

	start := time.Now()

	absRoot, err := filepath.Abs(repoRoot)
	if err != nil {
		return nil, err
	}

	// Start or reuse client.
	if err := p.ensureClient(absRoot); err != nil {
		return nil, fmt.Errorf("start LSP server: %w", err)
	}

	result := &semantic.EnrichResult{
		Provider: p.Name(),
		Language: p.languages[0],
	}

	// Collect nodes that need enrichment (AMBIGUOUS or INFERRED edges).
	type enrichTarget struct {
		node *graph.Node
		edge *graph.Edge
	}

	var targets []enrichTarget
	for _, e := range g.AllEdges() {
		if e.Confidence >= 1.0 {
			continue
		}
		fromNode := g.GetNode(e.From)
		if fromNode == nil {
			continue
		}
		if p.nodeMatches(fromNode, absRoot) {
			targets = append(targets, enrichTarget{node: fromNode, edge: e})
		}
	}

	// Count total symbols.
	for _, n := range g.AllNodes() {
		if n.Kind == graph.KindFile || n.Kind == graph.KindImport {
			continue
		}
		if p.nodeMatches(n, absRoot) {
			result.SymbolsTotal++
		}
	}

	// Open documents for files that have targets.
	openedFiles := make(map[string]bool)
	for _, t := range targets {
		diskPath := diskRelPath(absRoot, t.node.FilePath)
		if !openedFiles[diskPath] {
			if err := p.openDocument(absRoot, diskPath); err != nil {
				p.logger.Debug("LSP: failed to open document",
					zap.String("file", t.node.FilePath),
					zap.String("disk_path", diskPath),
					zap.Error(err),
				)
				continue
			}
			openedFiles[diskPath] = true
		}
	}

	// Query hover info for nodes to enrich metadata.
	var hoverNodes []*graph.Node
	for _, n := range g.AllNodes() {
		if n.Kind == graph.KindFile || n.Kind == graph.KindImport {
			continue
		}
		if !p.nodeMatches(n, absRoot) {
			continue
		}

		diskPath := diskRelPath(absRoot, n.FilePath)
		if !openedFiles[diskPath] {
			if err := p.openDocument(absRoot, diskPath); err != nil {
				continue
			}
			openedFiles[diskPath] = true
		}
		hoverNodes = append(hoverNodes, n)
	}

	enrichedNodes := make(map[string]bool)
	var enrichMu sync.Mutex
	workCh := make(chan *graph.Node)
	var wg sync.WaitGroup
	workers := p.maxParallel
	if workers <= 0 {
		workers = 1
	}
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := range workCh {
				diskPath := diskRelPath(absRoot, n.FilePath)
				hoverResult, err := p.hover(absRoot, diskPath, n.StartLine-1, symbolColumn(absRoot, diskPath, n.StartLine, n.Name))
				if err != nil || hoverResult == nil {
					continue
				}

				typeInfo := extractTypeFromHover(hoverResult.Contents.Value)
				if typeInfo == "" {
					continue
				}
				semantic.EnrichNodeMeta(n, "semantic_type", typeInfo, p.Name())
				enrichMu.Lock()
				if !enrichedNodes[n.ID] {
					result.NodesEnriched++
					result.SymbolsCovered++
					enrichedNodes[n.ID] = true
				}
				enrichMu.Unlock()
			}
		}()
	}
	for _, n := range hoverNodes {
		workCh <- n
	}
	close(workCh)
	wg.Wait()

	// Query implementations for interface nodes.
	for _, n := range g.AllNodes() {
		if n.Kind != graph.KindInterface {
			continue
		}
		if !p.nodeMatches(n, absRoot) {
			continue
		}

		diskPath := diskRelPath(absRoot, n.FilePath)
		impls, err := p.findImplementations(absRoot, diskPath, n.StartLine-1, symbolColumn(absRoot, diskPath, n.StartLine, n.Name))
		if err != nil || len(impls) == 0 {
			continue
		}

		for _, loc := range impls {
			implPath := uriToPath(loc.URI, absRoot)
			if implPath == "" {
				continue
			}
			implNode := matchNodeByFileLine(g, implPath, loc.Range.Start.Line+1)
			if implNode == nil {
				continue
			}

			existing := semantic.FindMatchingEdge(g, implNode.ID, n.ID, graph.EdgeImplements)
			if existing != nil {
				if existing.Confidence < 1.0 {
					semantic.ConfirmEdge(existing, p.Name())
					result.EdgesConfirmed++
				}
			} else {
				semantic.AddSemanticEdge(g, implNode.ID, n.ID, graph.EdgeImplements,
					implNode.FilePath, implNode.StartLine, p.Name())
				result.EdgesAdded++
			}
		}
	}

	// Query references for AMBIGUOUS edges to confirm/refute.
	for _, t := range targets {
		toNode := g.GetNode(t.edge.To)
		if toNode == nil || !p.nodeMatches(toNode, absRoot) {
			continue
		}

		toDiskPath := diskRelPath(absRoot, toNode.FilePath)
		refs, err := p.findReferences(absRoot, toDiskPath, toNode.StartLine-1, symbolColumn(absRoot, toDiskPath, toNode.StartLine, toNode.Name))
		if err != nil || len(refs) == 0 {
			continue
		}

		// Check if any reference matches the caller's location.
		confirmed := false
		for _, ref := range refs {
			refPath := uriToPath(ref.URI, absRoot)
			if sameGraphOrDiskPath(refPath, t.node.FilePath) &&
				ref.Range.Start.Line+1 >= t.node.StartLine &&
				ref.Range.Start.Line+1 <= t.node.EndLine {
				confirmed = true
				break
			}
		}

		if confirmed {
			semantic.ConfirmEdge(t.edge, p.Name())
			result.EdgesConfirmed++
		}
	}

	if result.SymbolsTotal > 0 {
		result.CoveragePercent = float64(result.SymbolsCovered) / float64(result.SymbolsTotal) * 100
	}

	result.DurationMs = time.Since(start).Milliseconds()
	return result, nil
}

func (p *Provider) EnrichFile(g *graph.Graph, repoRoot, filePath string) (*semantic.EnrichResult, error) {
	// LSP supports incremental updates, but for simplicity we skip it.
	// The full Enrich pass handles this.
	return nil, nil
}

func (p *Provider) nodeMatches(n *graph.Node, repoRoot string) bool {
	if n == nil || !p.languageMatches(n.Language) {
		return false
	}
	return p.supportsFile(diskRelPath(repoRoot, n.FilePath))
}

func (p *Provider) languageMatches(language string) bool {
	for _, lang := range p.languages {
		if language == lang {
			return true
		}
	}
	return false
}

func (p *Provider) supportsFile(relPath string) bool {
	ext := strings.ToLower(filepath.Ext(relPath))
	for _, lang := range p.languages {
		switch lang {
		case "typescript":
			if ext == ".ts" || ext == ".tsx" || ext == ".mts" || ext == ".cts" {
				return true
			}
		case "javascript":
			if ext == ".js" || ext == ".jsx" || ext == ".mjs" || ext == ".cjs" {
				return true
			}
		case "go":
			if ext == ".go" {
				return true
			}
		default:
			return true
		}
	}
	return false
}

// ensureClient starts the LSP server if not already running.
func (p *Provider) ensureClient(workspaceRoot string) error {
	if p.client != nil {
		if p.clientRoot == "" || p.clientRoot == workspaceRoot {
			return nil
		}
		_ = p.client.Shutdown()
		p.client = nil
		p.clientRoot = ""
	}

	client, err := NewClient(p.command, p.args, workspaceRoot, p.logger, p.requestTimeout)
	if err != nil {
		return err
	}

	// Send initialize request.
	initParams := InitializeParams{
		ProcessID: os.Getpid(),
		RootURI:   pathToURI(workspaceRoot),
		Capabilities: ClientCapabilities{
			TextDocument: TextDocumentClientCapabilities{
				Implementation: &ImplementationCapability{DynamicRegistration: true},
				References:     &ReferencesCapability{DynamicRegistration: true},
				Definition:     &DefinitionCapability{DynamicRegistration: true},
				Hover:          &HoverCapability{ContentFormat: []string{"plaintext"}},
			},
		},
	}

	var initResult InitializeResult
	if err := client.Call("initialize", initParams, &initResult); err != nil {
		_ = client.Shutdown()
		return fmt.Errorf("initialize: %w", err)
	}

	// Send initialized notification.
	if err := client.Notify("initialized", struct{}{}); err != nil {
		_ = client.Shutdown()
		return fmt.Errorf("initialized: %w", err)
	}

	p.client = client
	p.clientRoot = workspaceRoot
	return nil
}

// openDocument sends textDocument/didOpen for a file.
func (p *Provider) openDocument(repoRoot, relPath string) error {
	if !p.supportsFile(relPath) {
		return fmt.Errorf("unsupported LSP document extension: %s", relPath)
	}
	absPath := filepath.Join(repoRoot, relPath)
	content, err := os.ReadFile(absPath)
	if err != nil {
		return err
	}

	langID := lspLanguageID(relPath, p.languages)

	return p.client.Notify("textDocument/didOpen", DidOpenTextDocumentParams{
		TextDocument: TextDocumentItem{
			URI:        pathToURI(absPath),
			LanguageID: langID,
			Version:    1,
			Text:       string(content),
		},
	})
}

// hover queries hover info for a position.
func (p *Provider) hover(repoRoot, relPath string, line, col int) (*HoverResult, error) {
	absPath := filepath.Join(repoRoot, relPath)
	params := HoverParams{
		TextDocumentPositionParams: TextDocumentPositionParams{
			TextDocument: TextDocumentIdentifier{URI: pathToURI(absPath)},
			Position:     Position{Line: line, Character: col},
		},
	}

	var result HoverResult
	if err := p.client.Call("textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	if result.Contents.Value == "" {
		return nil, nil
	}
	return &result, nil
}

// findImplementations queries textDocument/implementation.
func (p *Provider) findImplementations(repoRoot, relPath string, line, col int) ([]Location, error) {
	absPath := filepath.Join(repoRoot, relPath)
	params := ImplementationParams{
		TextDocumentPositionParams: TextDocumentPositionParams{
			TextDocument: TextDocumentIdentifier{URI: pathToURI(absPath)},
			Position:     Position{Line: line, Character: col},
		},
	}

	var locations []Location
	if err := p.client.Call("textDocument/implementation", params, &locations); err != nil {
		return nil, err
	}
	return locations, nil
}

// findReferences queries textDocument/references.
func (p *Provider) findReferences(repoRoot, relPath string, line, col int) ([]Location, error) {
	absPath := filepath.Join(repoRoot, relPath)
	params := ReferenceParams{
		TextDocumentPositionParams: TextDocumentPositionParams{
			TextDocument: TextDocumentIdentifier{URI: pathToURI(absPath)},
			Position:     Position{Line: line, Character: col},
		},
		Context: ReferenceContext{IncludeDeclaration: false},
	}

	var locations []Location
	if err := p.client.Call("textDocument/references", params, &locations); err != nil {
		return nil, err
	}
	return locations, nil
}

func lspLanguageID(relPath string, languages []string) string {
	switch strings.ToLower(filepath.Ext(relPath)) {
	case ".tsx":
		return "typescriptreact"
	case ".ts", ".mts", ".cts":
		return "typescript"
	case ".jsx":
		return "javascriptreact"
	case ".js", ".mjs", ".cjs":
		return "javascript"
	}
	if len(languages) > 0 {
		return languages[0]
	}
	return "go"
}

func diskRelPath(repoRoot, graphPath string) string {
	if graphPath == "" || filepath.IsAbs(graphPath) {
		return graphPath
	}
	if _, err := os.Stat(filepath.Join(repoRoot, graphPath)); err == nil {
		return graphPath
	}
	parts := strings.SplitN(filepath.ToSlash(graphPath), "/", 2)
	if len(parts) == 2 {
		if _, err := os.Stat(filepath.Join(repoRoot, parts[1])); err == nil {
			return parts[1]
		}
	}
	return graphPath
}

func sameGraphOrDiskPath(a, b string) bool {
	a = filepath.ToSlash(a)
	b = filepath.ToSlash(b)
	return a == b || strings.HasSuffix(a, "/"+b) || strings.HasSuffix(b, "/"+a)
}

func matchNodeByFileLine(g *graph.Graph, filePath string, line int) *graph.Node {
	if n := semantic.MatchNodeByFileLine(g, filePath, line); n != nil {
		return n
	}
	for _, candidate := range g.AllNodes() {
		if candidate.Kind == graph.KindFile || candidate.Kind == graph.KindImport {
			continue
		}
		if sameGraphOrDiskPath(filePath, candidate.FilePath) && candidate.StartLine <= line && line <= candidate.EndLine {
			return candidate
		}
	}
	return nil
}

// pathToURI converts a file path to a file:// URI.
func pathToURI(path string) string {
	absPath, _ := filepath.Abs(path)
	return "file://" + absPath
}

// uriToPath converts a file:// URI to a repo-relative path.
func uriToPath(uri, repoRoot string) string {
	parsed, err := url.Parse(uri)
	if err != nil {
		return ""
	}
	absPath := parsed.Path
	if !strings.HasPrefix(absPath, repoRoot) {
		return ""
	}
	rel, err := filepath.Rel(repoRoot, absPath)
	if err != nil {
		return ""
	}
	return filepath.ToSlash(rel)
}

// symbolColumn returns a best-effort zero-based column for name on line. LSP
// queries at column 0 often hit whitespace/export keywords and return nothing.
func symbolColumn(repoRoot, relPath string, line int, name string) int {
	if line <= 0 || name == "" {
		return 0
	}
	content, err := os.ReadFile(filepath.Join(repoRoot, relPath))
	if err != nil {
		return 0
	}
	lines := strings.Split(string(content), "\n")
	if line > len(lines) {
		return 0
	}
	needle := name
	if i := strings.LastIndex(needle, "."); i >= 0 && i+1 < len(needle) {
		needle = needle[i+1:]
	}
	idx := strings.Index(lines[line-1], needle)
	if idx < 0 {
		return 0
	}
	return idx
}

// extractTypeFromHover extracts type information from hover text.
func extractTypeFromHover(hover string) string {
	// Remove markdown code fences.
	for _, lang := range []string{"go", "typescript", "javascript", "ts", "js"} {
		hover = strings.TrimPrefix(hover, "```"+lang+"\n")
	}
	hover = strings.TrimPrefix(hover, "```\n")
	hover = strings.TrimSuffix(hover, "\n```")
	hover = strings.TrimSpace(hover)

	lines := strings.SplitN(hover, "\n", 2)
	if len(lines) > 0 {
		line := strings.TrimSpace(lines[0])
		if strings.HasPrefix(line, "func ") ||
			strings.HasPrefix(line, "function ") ||
			strings.HasPrefix(line, "type ") ||
			strings.HasPrefix(line, "interface ") ||
			strings.HasPrefix(line, "class ") ||
			strings.HasPrefix(line, "namespace ") ||
			strings.HasPrefix(line, "var ") ||
			strings.HasPrefix(line, "let ") ||
			strings.HasPrefix(line, "const ") ||
			strings.HasPrefix(line, "field ") ||
			strings.HasPrefix(line, "property ") ||
			strings.HasPrefix(line, "package ") {
			return line
		}
		// Short type like "string", "*Foo", "[]byte".
		if !strings.Contains(line, " ") && len(line) > 0 && len(line) < 100 {
			return line
		}
	}
	return ""
}
