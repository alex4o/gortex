package main

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/zzet/gortex/internal/config"
	"github.com/zzet/gortex/internal/contracts"
	"github.com/zzet/gortex/internal/daemon"
	"github.com/zzet/gortex/internal/embedding"
	"github.com/zzet/gortex/internal/graph"
	"github.com/zzet/gortex/internal/indexer"
	gortexmcp "github.com/zzet/gortex/internal/mcp"
	"github.com/zzet/gortex/internal/parser"
	"github.com/zzet/gortex/internal/parser/languages"
	"github.com/zzet/gortex/internal/persistence"
	"github.com/zzet/gortex/internal/query"
	"github.com/zzet/gortex/internal/savings"
	"github.com/zzet/gortex/internal/semantic"
	"github.com/zzet/gortex/internal/semantic/goanalysis"
	"github.com/zzet/gortex/internal/semantic/lsp"
	"github.com/zzet/gortex/internal/semantic/scip"
	"github.com/zzet/gortex/internal/server"
	"github.com/zzet/gortex/internal/server/hub"
	"github.com/zzet/gortex/internal/workspace"
)

var (
	mcpIndex           string
	mcpTransport       string
	mcpPort            int
	mcpBind            string
	mcpAuthToken       string
	mcpWatch           bool
	mcpServerAPI       bool
	mcpCORSOrigin      string
	mcpDebounce        int
	mcpTrack           []string
	mcpProject         string
	mcpCacheDir        string
	mcpNoCache         bool
	mcpEmbeddings      bool
	mcpEmbeddingsURL   string
	mcpEmbeddingsModel string
	mcpSemantic        bool
	mcpNoSemantic      bool
	mcpSemanticMode    string
	mcpNoDaemon        bool
	mcpForceProxy      bool
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start the MCP server (stdio transport)",
	RunE:  runMCP,
}

func init() {
	mcpCmd.Flags().StringVar(&mcpIndex, "index", "", "repository path to index on startup")
	mcpCmd.Flags().StringVar(&mcpTransport, "transport", "stdio", "transport: stdio")
	mcpCmd.Flags().IntVar(&mcpPort, "port", 8765, "port for HTTP server API when --server is set")
	mcpCmd.Flags().BoolVar(&mcpWatch, "watch", false, "keep graph in sync with filesystem changes")
	mcpCmd.Flags().IntVar(&mcpDebounce, "debounce", 150, "debounce delay in ms")
	mcpCmd.Flags().BoolVar(&mcpServerAPI, "server", false, "start HTTP server API alongside MCP stdio")
	mcpCmd.Flags().StringVar(&mcpBind, "bind", "127.0.0.1", "bind address for --server; requires --auth-token when not localhost")
	mcpCmd.Flags().StringVar(&mcpAuthToken, "auth-token", "", "bearer token for --server's /v1/* requests (fallback: $GORTEX_SERVER_TOKEN)")
	mcpCmd.Flags().StringVar(&mcpCORSOrigin, "cors-origin", "*", "allowed CORS origin for server API")
	mcpCmd.Flags().StringSliceVar(&mcpTrack, "track", nil, "additional repository paths to track")
	mcpCmd.Flags().StringVar(&mcpProject, "project", "", "active project name")
	mcpCmd.Flags().StringVar(&mcpCacheDir, "cache-dir", "", "graph cache directory (default ~/.cache/gortex/)")
	mcpCmd.Flags().BoolVar(&mcpNoCache, "no-cache", false, "disable graph caching")
	mcpCmd.Flags().BoolVar(&mcpEmbeddings, "embeddings", false, "enable semantic search (built-in word vectors or transformer if compiled in)")
	mcpCmd.Flags().StringVar(&mcpEmbeddingsURL, "embeddings-url", "", "embedding API URL (e.g. http://localhost:11434 for Ollama)")
	mcpCmd.Flags().StringVar(&mcpEmbeddingsModel, "embeddings-model", "", "embedding model name (default: auto-detect)")
	mcpCmd.Flags().BoolVar(&mcpSemantic, "semantic", false, "enable semantic enrichment (SCIP, go/types, LSP)")
	mcpCmd.Flags().BoolVar(&mcpNoSemantic, "no-semantic", false, "disable semantic enrichment")
	mcpCmd.Flags().StringVar(&mcpSemanticMode, "semantic-mode", "typecheck", "Go analysis mode: typecheck or callgraph")
	mcpCmd.Flags().BoolVar(&mcpNoDaemon, "no-daemon", false, "force embedded server, do not connect to a running daemon")
	mcpCmd.Flags().BoolVar(&mcpForceProxy, "proxy", false, "require a running daemon and proxy through it (error if unavailable)")
	rootCmd.AddCommand(mcpCmd)
}

func runMCP(cmd *cobra.Command, args []string) error {
	// Daemon-first: if stdio indicates an MCP client spawned us AND a
	// daemon is listening, proxy through it instead of spinning up an
	// embedded server. Terminal invocations fall through to embedded by
	// default.
	if shouldTryProxy(mcpNoDaemon, mcpForceProxy) {
		ran, proxyErr := runProxy(cmd.Context())
		if proxyErr != nil {
			return proxyErr
		}
		if ran {
			return nil
		}
		// Daemon unavailable — fall through to embedded.
		if mcpForceProxy {
			return fmt.Errorf("--proxy was passed but no daemon is running (socket: %s)",
				daemon.SocketPath())
		}
	}

	logger := newLogger()
	defer func() { _ = logger.Sync() }()

	// Two-entry-point handshake. The MCP server binds at
	// either the workspace root (`.gortex/workspace.toml` present in
	// cwd) or a single-project root (`.gortex/` directory in cwd).
	// Anywhere else fails the handshake with a clear message naming
	// both supported entry points — there is no walk-up.
	cwd, cwdErr := os.Getwd()
	if cwdErr != nil {
		return fmt.Errorf("resolving cwd for MCP handshake: %w", cwdErr)
	}
	bind, bindErr := workspace.Resolve(cwd)
	if bindErr != nil {
		fmt.Fprintf(os.Stderr, "[gortex] MCP handshake failed: %v\n", bindErr)
		return bindErr
	}
	for _, w := range workspace.FormatMarkerWarnings(bind.Marker) {
		fmt.Fprintf(os.Stderr, "[gortex] %s\n", w)
	}
	switch bind.Mode {
	case workspace.ModeWorkspace:
		fmt.Fprintf(os.Stderr, "[gortex] workspace mode: %s (%d members)\n",
			bind.Root, len(bind.Members))
	case workspace.ModeSingleProject:
		fmt.Fprintf(os.Stderr, "[gortex] single-project mode: %s\n", bind.Root)
	}

	cfg, err := config.Load(cfgFile)
	if err != nil {
		return err
	}

	g := graph.New()
	reg := parser.NewRegistry()
	languages.RegisterAll(reg)

	idx := indexer.New(g, reg, cfg.Index, logger)

	// Set up embedding provider for semantic search. Held in a local
	// variable so it can be plumbed into MultiIndexer below — without
	// that, per-repo indexers created by MultiIndexer have embedder=nil
	// and skip the vector pass.
	var embedder embedding.Provider
	if mcpEmbeddingsURL != "" {
		embedder = embedding.NewAPIProvider(mcpEmbeddingsURL, mcpEmbeddingsModel)
		fmt.Fprintf(os.Stderr, "[gortex] semantic search enabled (API: %s)\n", mcpEmbeddingsURL)
	} else if mcpEmbeddings {
		e, err := embedding.NewLocalProvider()
		if err != nil {
			fmt.Fprintf(os.Stderr, "[gortex] warning: embeddings disabled: %v\n", err)
		} else {
			embedder = e
			fmt.Fprintf(os.Stderr, "[gortex] semantic search enabled (local)\n")
		}
	}
	if embedder != nil {
		idx.SetEmbedder(embedder)
	}

	// Set up semantic enrichment.
	if !mcpNoSemantic && (mcpSemantic || cfg.Semantic.Enabled) {
		semCfg := cfg.Semantic
		semCfg.Enabled = true

		// Convert config provider entries to semantic.Config format.
		semInternalCfg := semantic.Config{
			Enabled:           semCfg.Enabled,
			TimeoutSeconds:    semCfg.TimeoutSeconds,
			EnrichOnWatch:     semCfg.EnrichOnWatch,
			WatchDebounceMs:   semCfg.WatchDebounceMs,
			RefuteUnconfirmed: semCfg.RefuteUnconfirmed,
		}
		for _, pc := range semCfg.Providers {
			semInternalCfg.Providers = append(semInternalCfg.Providers, semantic.ProviderConfig{
				Name:        pc.Name,
				Command:     pc.Command,
				Args:        pc.Args,
				Languages:   pc.Languages,
				Priority:    pc.Priority,
				Enabled:     pc.Enabled,
				Mode:        pc.Mode,
				Daemon:      pc.Daemon,
				MaxParallel: pc.MaxParallel,
			})
		}

		semMgr := semantic.NewManager(semInternalCfg, logger)

		// Register go/types provider (always available for Go).
		mode := goanalysis.ModeTypeCheck
		if mcpSemanticMode == "callgraph" {
			mode = goanalysis.ModeCallGraph
		}
		goProvider := goanalysis.NewProvider(mode, false, logger)
		semMgr.RegisterProvider(goProvider)
		// Wire the goanalysis provider as the contract pipeline's
		// BindingResolver — once Enrich runs, contract enrichment can
		// upgrade Origin from ast_inferred to lsp_resolved using
		// compiler-grade type info. See spec-contract-extraction.md §4.5.
		contracts.SetBindingResolver(goProvider)

		// Register SCIP providers from config.
		for _, pc := range semCfg.Providers {
			if !pc.Enabled {
				continue
			}
			switch {
			case strings.HasPrefix(pc.Name, "scip-") && pc.Command != "":
				semMgr.RegisterProvider(scip.NewProvider(pc.Command, pc.Args, pc.Languages, semCfg.TimeoutSeconds, logger))
			case strings.HasPrefix(pc.Name, "gopls") || pc.Daemon:
				semMgr.RegisterProvider(lsp.NewProvider(pc.Command, pc.Args, pc.Languages, pc.Daemon, pc.MaxParallel, semCfg.TimeoutSeconds, logger))
			}
		}

		idx.SetSemanticManager(semMgr)
		fmt.Fprintf(os.Stderr, "[gortex] semantic enrichment enabled (mode: %s)\n", mcpSemanticMode)
	}

	// Initialize ConfigManager for multi-repo support.
	cm, err := config.NewConfigManager("")
	if err != nil {
		// Non-fatal: fall back to single-repo mode.
		fmt.Fprintf(os.Stderr, "[gortex] warning: could not load global config: %v\n", err)
	}

	// Add --track repos to GlobalConfig.
	if cm != nil && len(mcpTrack) > 0 {
		for _, trackPath := range mcpTrack {
			absPath, err := filepath.Abs(trackPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[gortex] warning: could not resolve --track path %s: %v\n", trackPath, err)
				continue
			}
			// Skip duplicates.
			if err := cm.Global().AddRepo(config.RepoEntry{Path: absPath}); err != nil {
				fmt.Fprintf(os.Stderr, "[gortex] warning: could not add --track repo %s: %v\n", absPath, err)
			}
		}
	}

	// Determine active project.
	activeProject := mcpProject
	if activeProject == "" && cm != nil {
		activeProject = cm.Global().ActiveProject
	}
	if cm != nil {
		cm.Global().ActiveProject = activeProject
	}

	// Initialize MultiIndexer when we have a ConfigManager.
	var mi *indexer.MultiIndexer
	if cm != nil {
		mi = indexer.NewMultiIndexer(g, reg, idx.Search(), cm, logger)
		if embedder != nil {
			mi.SetEmbedder(embedder)
		}
	}

	// Build multi-repo options for the MCP server.
	var multiOpts []gortexmcp.MultiRepoOptions
	if mi != nil || cm != nil {
		multiOpts = append(multiOpts, gortexmcp.MultiRepoOptions{
			MultiIndexer:  mi,
			ConfigManager: cm,
			ActiveProject: activeProject,
		})
	}

	// Create MCP server immediately so the stdio handshake can complete
	// before indexing (which may take time on large repos).
	eng := query.NewEngine(g)
	eng.SetSearchProvider(idx.Search)
	gortexmcp.Version = version
	srv := gortexmcp.NewServer(eng, g, idx, nil, logger, cfg.Guards.Rules, multiOpts...)
	srv.SetBind(bind)

	// Wire semantic manager to MCP server for stats reporting.
	if semMgr := idx.SemanticManager(); semMgr != nil {
		srv.SetSemanticManager(semMgr)
	}

	// Initialize feedback persistence for cross-session context learning.
	srv.InitFeedback(mcpCacheDir, mcpIndex)
	// Combo tracker persists (query → chosen symbol) associations per repo
	// so the next time the agent asks the same thing, the previously-picked
	// symbol floats to the top of search results.
	srv.InitCombo(mcpCacheDir, mcpIndex, gortexmcp.ModeAI)
	// Frecency: per-symbol access timestamps with AI-tuned (3-day half-life)
	// decay. Hot symbols in the current session float up in search results.
	srv.InitFrecency(mcpCacheDir, mcpIndex, gortexmcp.ModeAI)

	// Initialize cumulative token-savings persistence. Path defaults to
	// ~/.cache/gortex/savings.json; the store operates in-memory when the
	// cache dir is unavailable.
	savingsPath := savings.DefaultPath()
	if mcpCacheDir != "" {
		savingsPath = filepath.Join(mcpCacheDir, "savings.json")
	}
	if savingsStore, err := savings.Open(savingsPath); err == nil {
		srv.InitSavings(savingsStore, mcpIndex)
		stopSavingsFlush := srv.StartPeriodicSavingsFlush(5 * time.Minute)
		defer stopSavingsFlush()
		defer func() { _ = srv.FlushSavings() }()
	} else {
		fmt.Fprintf(os.Stderr, "[gortex] savings persistence disabled: %v\n", err)
	}

	fmt.Fprintf(os.Stderr, "[gortex] MCP server ready (transport: %s)\n", mcpTransport)

	// Start server HTTP API if requested.
	if mcpServerAPI {
		authToken := mcpAuthToken
		if authToken == "" {
			authToken = os.Getenv("GORTEX_SERVER_TOKEN")
		}
		if authToken == "" {
			if !isLocalhostBind(mcpBind) {
				return fmt.Errorf("--bind %q requires --auth-token (or $GORTEX_SERVER_TOKEN); refusing to expose unauthenticated server on external interface", mcpBind)
			}
			fmt.Fprintln(os.Stderr, "[gortex] server: unauthenticated mode; localhost only")
		}

		serverHandler := server.NewHandler(srv.MCPServer(), g, version, logger)
		if cm != nil {
			serverHandler.SetConfigManager(cm)
		}
		if id, err := resolveServerID(mcpCacheDir); err == nil {
			serverHandler.SetServerID(id)
		} else {
			fmt.Fprintf(os.Stderr, "[gortex] server: id persistence disabled: %v\n", err)
		}
		handler := server.WithAuth(serverHandler, authToken)
		corsOpts := server.CORSOptions{AllowOrigins: []string{mcpCORSOrigin}}
		handler = server.WithCORS(handler, corsOpts)
		go func() {
			serverAddr := fmt.Sprintf("%s:%d", mcpBind, mcpPort)
			fmt.Fprintf(os.Stderr, "[gortex] server API at http://%s\n", serverAddr)
			if err := http.ListenAndServe(serverAddr, handler); err != nil && err != http.ErrServerClosed {
				fmt.Fprintf(os.Stderr, "[gortex] server API error: %v\n", err)
			}
		}()
	}

	// Start MCP stdio in a goroutine so we can do background init.
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.ServeStdio()
	}()

	// Create persistence store.
	var store persistence.Store
	if mcpNoCache {
		store = persistence.NopStore{}
	} else {
		var err error
		store, err = persistence.NewFileStore(mcpCacheDir, version)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[gortex] warning: cache disabled: %v\n", err)
			store = persistence.NopStore{}
		}
	}

	// Background: index, watch, analyze — graph populates while MCP is live.
	go func() {
		if mcpIndex != "" {
			commitHash := gitCommitHash(mcpIndex)
			cached := false

			if commitHash != "" && store.Check(mcpIndex, commitHash) && store.Validate(mcpIndex, commitHash) {
				snap, err := store.Load(mcpIndex, commitHash)
				if err == nil {
					for _, n := range snap.Nodes {
						g.AddNode(n)
					}
					for _, e := range snap.Edges {
						g.AddEdge(e)
					}
					idx.SetFileMtimes(snap.FileMtimes)
					idx.SetRootPath(mcpIndex)

					// Restore vector index if available.
					if len(snap.VectorIndex) > 0 && snap.VectorDims > 0 {
						if err := idx.ImportVectorIndex(snap.VectorIndex, snap.VectorDims, snap.VectorCount); err != nil {
							fmt.Fprintf(os.Stderr, "[gortex] vector index restore failed: %v\n", err)
						}
					}

					result, err := idx.IncrementalReindex(mcpIndex)
					if err != nil {
						fmt.Fprintf(os.Stderr, "[gortex] incremental reindex failed: %v\n", err)
					} else {
						fmt.Fprintf(os.Stderr, "[gortex] restored graph (%d nodes, %d edges), re-indexed %d stale files in %dms\n",
							result.NodeCount, result.EdgeCount, result.FileCount, result.DurationMs)
					}
					cached = true
				} else {
					fmt.Fprintf(os.Stderr, "[gortex] cache load failed, will re-index: %v\n", err)
				}
			}

			if !cached {
				fmt.Fprintf(os.Stderr, "[gortex] indexing %s...\n", mcpIndex)
				result, err := idx.Index(mcpIndex)
				if err != nil {
					fmt.Fprintf(os.Stderr, "[gortex] indexing failed: %v\n", err)
					return
				}
				fmt.Fprintf(os.Stderr, "[gortex] indexed %d files (%d nodes, %d edges) in %dms\n",
					result.FileCount, result.NodeCount, result.EdgeCount, result.DurationMs)
			}
		}

		// Search backend is auto-updated via SearchProvider (idx.Search)

		// Pass contract registry to MCP server.
		// In multi-repo mode, merge all per-repo registries.
		if mi != nil {
			srv.SetContractRegistry(mi.MergedContractRegistry())
		} else if cr := idx.ContractRegistry(); cr != nil {
			srv.SetContractRegistry(cr)
		}

		// Start watcher if requested.
		if mcpWatch {
			wcfg := cfg.Watch
			wcfg.Enabled = true
			if mcpDebounce > 0 {
				wcfg.DebounceMs = mcpDebounce
			}

			watcher, err := indexer.NewWatcher(idx, wcfg, logger)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[gortex] watcher setup failed: %v\n", err)
				return
			}

			watchPaths := wcfg.Paths
			if len(watchPaths) == 0 && mcpIndex != "" {
				watchPaths = []string{mcpIndex}
			}
			if len(watchPaths) == 0 {
				watchPaths = []string{"."}
			}

			if err := watcher.Start(watchPaths); err != nil {
				fmt.Fprintf(os.Stderr, "[gortex] watcher start failed: %v\n", err)
				return
			}
			srv.SetWatcher(watcher)

			// Create hub for fan-out of watcher events.
			eventHub := hub.New()
			go eventHub.Run(watcher.Events())

			srv.WatchForReanalysis(eventHub, 500)
			fmt.Fprintf(os.Stderr, "[gortex] watch mode active\n")
		}

		// Run initial analysis.
		srv.RunAnalysis()
	}()

	// Handle graceful shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		return err
	case sig := <-sigCh:
		fmt.Fprintf(os.Stderr, "\n[gortex] received %s, shutting down\n", sig)

		// Persist graph snapshot on shutdown.
		if mcpIndex != "" {
			commitHash := gitCommitHash(mcpIndex)
			if commitHash != "" {
				snap := &persistence.Snapshot{
					Version:    version,
					RepoPath:   mcpIndex,
					CommitHash: commitHash,
					IndexedAt:  time.Now(),
					Nodes:      g.AllNodes(),
					Edges:      g.AllEdges(),
					FileMtimes: idx.FileMtimes(),
				}
				// Include vector index if available.
				snap.VectorIndex, snap.VectorDims, snap.VectorCount = idx.ExportVectorIndex()
				if err := store.Save(snap); err != nil {
					fmt.Fprintf(os.Stderr, "[gortex] cache save failed: %v\n", err)
				} else {
					fmt.Fprintf(os.Stderr, "[gortex] saved graph snapshot (%d nodes, %d edges)\n",
						len(snap.Nodes), len(snap.Edges))
				}
			}
		}

		return nil
	}
}
