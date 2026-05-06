package indexer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/zzet/gortex/internal/config"
	"github.com/zzet/gortex/internal/contracts"
	"github.com/zzet/gortex/internal/embedding"
	"github.com/zzet/gortex/internal/graph"
	"github.com/zzet/gortex/internal/parser"
	"github.com/zzet/gortex/internal/resolver"
	"github.com/zzet/gortex/internal/search"
	"github.com/zzet/gortex/internal/semantic"
)

// RepoMetadata holds per-repo indexing state.
type RepoMetadata struct {
	RepoPrefix    string
	RootPath      string
	Identity      *RepoIdentity
	LastIndexTime time.Time
	FileCount     int
	NodeCount     int
	EdgeCount     int
	ParseErrors   []IndexError
	FileMtimes    map[string]int64
}

// MultiIndexer orchestrates indexing across multiple repositories.
type MultiIndexer struct {
	graph       *graph.Graph
	registry    *parser.Registry
	search      search.Backend
	embedder    embedding.Provider
	semanticMgr *semantic.Manager
	repos       map[string]*RepoMetadata // repoPrefix → metadata
	indexers    map[string]*Indexer      // repoPrefix → per-repo indexer
	configMgr   *config.ConfigManager
	logger      *zap.Logger
	mu          sync.RWMutex
}

// SetEmbedder installs the embedding provider every per-repo indexer
// should use. Must be called before IndexAll / TrackRepo for vectors
// to land in the graph — without this the fresh Indexer created per
// repo has embedder=nil and buildSearchIndex skips the vector pass.
// Safe to call zero or one times; subsequent calls silently replace.
func (mi *MultiIndexer) SetEmbedder(e embedding.Provider) {
	mi.mu.Lock()
	defer mi.mu.Unlock()
	mi.embedder = e
}

// SetSemanticManager installs the semantic enrichment manager every per-repo
// indexer should use. Must be called before IndexAll / TrackRepo; otherwise
// multi-repo indexing creates fresh Indexers with no semantic pass wired.
func (mi *MultiIndexer) SetSemanticManager(m *semantic.Manager) {
	mi.mu.Lock()
	defer mi.mu.Unlock()
	mi.semanticMgr = m
}

// NewMultiIndexer creates a MultiIndexer.
func NewMultiIndexer(
	g *graph.Graph,
	reg *parser.Registry,
	s search.Backend,
	cm *config.ConfigManager,
	logger *zap.Logger,
) *MultiIndexer {
	return &MultiIndexer{
		graph:     g,
		registry:  reg,
		search:    s,
		repos:     make(map[string]*RepoMetadata),
		indexers:  make(map[string]*Indexer),
		configMgr: cm,
		logger:    logger,
	}
}

// IndexAll indexes all active repos concurrently. Each repo gets its own
// Indexer instance with repo-specific config. Returns per-repo IndexResults.
func (mi *MultiIndexer) IndexAll() (map[string]*IndexResult, error) {
	return mi.IndexScoped("", "")
}

// IndexScoped is IndexAll restricted to repos whose workspace and
// project slugs match. Empty filters disable that axis (so empty/empty
// is equivalent to IndexAll). Resolution honours the same precedence
// as resolveWorkspaceID/resolveProjectID — RepoEntry override →
// `.gortex.yaml::workspace` → repo prefix — so a `gortex server
// --workspace foo` invocation matches both repos that declare
// `workspace: foo` in their own `.gortex.yaml` and repos pinned to
// `foo` from the user's global config.
//
// Returns an error when the filters exclude every active repo, so a
// `--workspace typo` surfaces as a startup failure rather than a
// silently empty graph.
func (mi *MultiIndexer) IndexScoped(workspaceSlug, projectSlug string) (map[string]*IndexResult, error) {
	repos := mi.configMgr.ActiveRepos()
	if len(repos) == 0 {
		return nil, nil
	}
	if workspaceSlug != "" || projectSlug != "" {
		filtered := mi.filterReposByScope(repos, workspaceSlug, projectSlug)
		if len(filtered) == 0 {
			return nil, fmt.Errorf("scope filter matched zero of %d active repos (workspace=%q project=%q)", len(repos), workspaceSlug, projectSlug)
		}
		repos = filtered
	}

	// Single-repo mode: delegate without prefixing.
	if len(repos) == 1 {
		r, err := mi.indexSingleRepo(repos[0])
		if err == nil {
			mi.ReconcileContractEdges()
		}
		return r, err
	}

	r, err := mi.indexMultiRepo(repos)
	if err == nil {
		mi.ReconcileContractEdges()
	}
	return r, err
}

// filterReposByScope returns the subset of repos whose resolved
// workspace and project slugs match the supplied filters. Empty
// filters disable that axis. Loads each repo's `.gortex.yaml` first so
// resolution sees the workspace/project declared there — matching only
// against `RepoEntry.Workspace` would miss repos that declare their
// slug in their own config file (the typical case for first-party
// repos).
func (mi *MultiIndexer) filterReposByScope(repos []config.RepoEntry, workspaceSlug, projectSlug string) []config.RepoEntry {
	if workspaceSlug == "" && projectSlug == "" {
		return repos
	}
	out := make([]config.RepoEntry, 0, len(repos))
	for _, e := range repos {
		absPath, err := filepath.Abs(e.Path)
		if err != nil {
			continue
		}
		prefix := config.ResolvePrefix(e)
		mi.configMgr.LoadWorkspaceConfig(prefix, absPath)
		cfg := mi.configMgr.GetRepoConfig(prefix)
		entryCopy := e
		if workspaceSlug != "" && resolveWorkspaceID(&entryCopy, cfg, prefix) != workspaceSlug {
			continue
		}
		if projectSlug != "" && resolveProjectID(&entryCopy, cfg, prefix) != projectSlug {
			continue
		}
		out = append(out, e)
	}
	return out
}

// indexSingleRepo indexes a single repo without prefixing for backward compatibility.
func (mi *MultiIndexer) indexSingleRepo(entry config.RepoEntry) (map[string]*IndexResult, error) {
	absPath, err := filepath.Abs(entry.Path)
	if err != nil {
		return nil, fmt.Errorf("resolving path %s: %w", entry.Path, err)
	}

	prefix := config.ResolvePrefix(entry)
	mi.configMgr.LoadWorkspaceConfig(prefix, absPath)
	cfg := mi.configMgr.GetRepoConfig(prefix)

	idx := New(mi.graph, mi.registry, cfg.Index, mi.logger)
	idx.search = mi.search
	if mi.embedder != nil {
		idx.SetEmbedder(mi.embedder)
	}
	if mi.semanticMgr != nil {
		idx.SetSemanticManager(mi.semanticMgr)
	}
	// No repo prefix in single-repo mode.

	result, err := idx.Index(absPath)
	if err != nil {
		return nil, fmt.Errorf("indexing %s: %w", absPath, err)
	}

	identity, _ := DetectIdentity(absPath)

	mi.mu.Lock()
	mi.repos[prefix] = &RepoMetadata{
		RepoPrefix:    prefix,
		RootPath:      absPath,
		Identity:      identity,
		LastIndexTime: time.Now(),
		FileCount:     result.FileCount,
		NodeCount:     result.NodeCount,
		EdgeCount:     result.EdgeCount,
		ParseErrors:   result.Errors,
		FileMtimes:    idx.FileMtimes(),
	}
	mi.indexers[prefix] = idx
	mi.mu.Unlock()

	return map[string]*IndexResult{prefix: result}, nil
}

// readGoModModule reads the module path from a go.mod file.
func readGoModModule(repoPath string) string {
	data, err := os.ReadFile(filepath.Join(repoPath, "go.mod"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "module ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "module"))
		}
	}
	return ""
}

// indexMultiRepo indexes multiple repos concurrently with repo prefixes.
func (mi *MultiIndexer) indexMultiRepo(repos []config.RepoEntry) (map[string]*IndexResult, error) {
	type repoResult struct {
		prefix string
		result *IndexResult
		idx    *Indexer
		meta   *RepoMetadata
		err    error
	}

	// Pre-scan: build tracked repo module map from go.mod files.
	// This enables cross-repo dependency detection via GoModExtractor.
	trackedModules := make(map[string]string)
	for _, entry := range repos {
		prefix := config.ResolvePrefix(entry)
		absPath, _ := filepath.Abs(entry.Path)
		if mod := readGoModModule(absPath); mod != "" {
			trackedModules[prefix] = mod
			mi.logger.Debug("tracked repo module", zap.String("repo", prefix), zap.String("module", mod))
		}
	}

	resultCh := make(chan repoResult, len(repos))
	var wg sync.WaitGroup

	for _, entry := range repos {
		wg.Add(1)
		go func(e config.RepoEntry) {
			defer wg.Done()

			prefix := config.ResolvePrefix(e)
			absPath, err := filepath.Abs(e.Path)
			if err != nil {
				resultCh <- repoResult{prefix: prefix, err: fmt.Errorf("resolving path %s: %w", e.Path, err)}
				return
			}

			mi.configMgr.LoadWorkspaceConfig(prefix, absPath)
			cfg := mi.configMgr.GetRepoConfig(prefix)
			idx := New(mi.graph, mi.registry, cfg.Index, mi.logger)
			idx.search = mi.search
			if mi.embedder != nil {
				idx.SetEmbedder(mi.embedder)
			}
			if mi.semanticMgr != nil {
				idx.SetSemanticManager(mi.semanticMgr)
			}
			idx.SetRepoPrefix(prefix)
			idx.SetTrackedRepoModules(trackedModules)
			// Defer the cross-cutting passes (ResolveAll, InferImplements,
			// semantic enrich, contract extract+commit) so they don't race
			// against each other across goroutines on the shared graph.
			// They run serially below via RunDeferredPasses after wg.Wait().
			idx.SetDeferResolve(true)

			result, err := idx.Index(absPath)
			if err != nil {
				resultCh <- repoResult{prefix: prefix, err: fmt.Errorf("indexing %s: %w", absPath, err)}
				return
			}

			identity, _ := DetectIdentity(absPath)

			meta := &RepoMetadata{
				RepoPrefix:    prefix,
				RootPath:      absPath,
				Identity:      identity,
				LastIndexTime: time.Now(),
				FileCount:     result.FileCount,
				NodeCount:     result.NodeCount,
				EdgeCount:     result.EdgeCount,
				ParseErrors:   result.Errors,
				FileMtimes:    idx.FileMtimes(),
			}

			resultCh <- repoResult{prefix: prefix, result: result, idx: idx, meta: meta}
		}(entry)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	results := make(map[string]*IndexResult)
	var indexErrors []string

	mi.mu.Lock()
	for rr := range resultCh {
		if rr.err != nil {
			mi.logger.Error("failed to index repo", zap.String("prefix", rr.prefix), zap.Error(rr.err))
			indexErrors = append(indexErrors, rr.err.Error())
			continue
		}
		mi.repos[rr.prefix] = rr.meta
		mi.indexers[rr.prefix] = rr.idx
		results[rr.prefix] = rr.result
	}
	mi.mu.Unlock()

	// Run the per-repo passes that the goroutines above deferred. Serial
	// across repos is the simple correctness fix: ResolveAll mutates
	// Edge.Meta on edges in the shared graph, and the contract pass walks
	// every edge — running them in parallel across repos races. Inside a
	// single repo's ResolveAll the resolver still uses its own worker
	// pool, and parsing (the dominant cost on a fresh index) already ran
	// in parallel above, so the wall-time hit is small.
	deferCtx := context.Background()
	for prefix := range results {
		if idx, ok := mi.indexers[prefix]; ok {
			idx.RunDeferredPasses(deferCtx)
		}
	}

	// Run a global cross-repo resolution pass once every repo is
	// indexed, with the cross-workspace boundary check wired in.
	// Without this, repos that import each other have unresolved
	// edges that only resolve when an editor touches a file (the
	// watcher path is the only other place this resolver runs). The
	// boundary lookup means cross-workspace candidates only resolve
	// when an explicit `cross_workspace_deps` declaration covers
	// them.
	cr := resolver.NewCrossRepo(mi.graph)
	cr.SetCrossWorkspaceDepLookup(mi.crossWorkspaceLookup())
	cr.ResolveAll()
	mi.ReconcileContractEdges()

	if len(indexErrors) > 0 && len(results) == 0 {
		return nil, fmt.Errorf("all repos failed to index: %s", strings.Join(indexErrors, "; "))
	}

	return results, nil
}

// IndexRepo re-indexes a single repo by prefix. Evicts existing data first.
func (mi *MultiIndexer) IndexRepo(repoPrefix string) (*IndexResult, error) {
	mi.mu.RLock()
	meta, ok := mi.repos[repoPrefix]
	mi.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("repository not found: %s", repoPrefix)
	}

	// Evict existing data for this repo.
	if mi.IsMultiRepo() {
		mi.graph.EvictRepo(repoPrefix)
	}

	cfg := mi.configMgr.GetRepoConfig(repoPrefix)
	idx := New(mi.graph, mi.registry, cfg.Index, mi.logger)
	idx.search = mi.search
	if mi.embedder != nil {
		idx.SetEmbedder(mi.embedder)
	}
	if mi.IsMultiRepo() {
		idx.SetRepoPrefix(repoPrefix)
	}

	result, err := idx.Index(meta.RootPath)
	if err != nil {
		return nil, fmt.Errorf("indexing %s: %w", meta.RootPath, err)
	}

	mi.mu.Lock()
	mi.repos[repoPrefix] = &RepoMetadata{
		RepoPrefix:    repoPrefix,
		RootPath:      meta.RootPath,
		Identity:      meta.Identity,
		LastIndexTime: time.Now(),
		FileCount:     result.FileCount,
		NodeCount:     result.NodeCount,
		EdgeCount:     result.EdgeCount,
		ParseErrors:   result.Errors,
		FileMtimes:    idx.FileMtimes(),
	}
	mi.indexers[repoPrefix] = idx
	mi.mu.Unlock()

	// TODO: After re-indexing, run CrossRepoResolver.ResolveForRepo(repoPrefix)
	// to update cross-repo edges. This will be implemented in Task 7.1.

	mi.ReconcileContractEdges()

	return result, nil
}

// TrackRepo validates the path, detects identity, indexes, and adds to config.
func (mi *MultiIndexer) TrackRepo(entry config.RepoEntry) (*IndexResult, error) {
	return mi.TrackRepoCtx(context.Background(), entry)
}

// TrackRepoCtx is TrackRepo with a context, allowing callers to pipe progress
// reporters (via progress.WithReporter) through to the underlying Index call.
func (mi *MultiIndexer) TrackRepoCtx(ctx context.Context, entry config.RepoEntry) (*IndexResult, error) {
	absPath, err := filepath.Abs(entry.Path)
	if err != nil {
		return nil, fmt.Errorf("resolving path %s: %w", entry.Path, err)
	}

	// Validate path exists and is a directory.
	info, err := os.Stat(absPath)
	if err != nil {
		return nil, fmt.Errorf("path does not exist: %s", absPath)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", absPath)
	}

	identity, err := DetectIdentity(absPath)
	if err != nil {
		return nil, fmt.Errorf("detecting identity for %s: %w", absPath, err)
	}

	prefix := config.ResolvePrefix(entry)
	if prefix == "" || prefix == "." {
		prefix = identity.RepoPrefix
	}

	// Check if already tracked.
	mi.mu.RLock()
	if _, exists := mi.repos[prefix]; exists {
		mi.mu.RUnlock()
		return nil, nil // already tracked
	}
	mi.mu.RUnlock()

	// Determine if we need to prefix. We must consider both repos already
	// indexed in mi.repos AND the total repos configured — at daemon warmup
	// TrackRepoCtx is called in a loop over all configured repos, so at
	// iteration 0 mi.repos is empty while the config already has N entries.
	// Counting only mi.repos used to leave the first-indexed repo without a
	// prefix while every later repo got one, producing two ID schemes for
	// the same graph and halving cross-file edge density.
	totalConfigured := 1 // ourselves
	if mi.configMgr != nil {
		totalConfigured = len(mi.configMgr.Global().Repos)
	}
	willBeMultiRepo := len(mi.repos)+1 >= 2 || totalConfigured >= 2

	// Lazy-load the per-repo `.gortex.yaml` so GetRepoConfig sees the
	// workspace / project slugs declared inside the repo. Without
	// this the production code path never reads the file and every
	// repo silently falls back to `workspace = repoPrefix`, making
	// shared-workspace cross-repo matching impossible to express. Idempotent
	// — repeated calls just re-cache the parse.
	if mi.configMgr != nil {
		mi.configMgr.LoadWorkspaceConfig(prefix, absPath)
	}

	cfg := mi.configMgr.GetRepoConfig(prefix)
	idx := New(mi.graph, mi.registry, cfg.Index, mi.logger)
	idx.search = mi.search
	if mi.embedder != nil {
		idx.SetEmbedder(mi.embedder)
	}
	if willBeMultiRepo {
		idx.SetRepoPrefix(prefix)
	}
	// Workspace / project slugs stamped on every node. Resolution
	// order (highest priority first): RepoEntry.Workspace from the
	// global config (lets users pin OSS repos without committing a
	// `.gortex.yaml`) → `.gortex.yaml::workspace` → repoPrefix
	// (default). resolveWorkspaceID encodes the precedence; the
	// WorkspaceID-keyed contract registry and the boundary-enforced
	// matcher both consume the result.
	entryCopy := entry
	idx.SetWorkspaceID(resolveWorkspaceID(&entryCopy, cfg, prefix))
	idx.SetProjectID(resolveProjectID(&entryCopy, cfg, prefix))

	result, err := idx.IndexCtx(ctx, absPath)
	if err != nil {
		return nil, fmt.Errorf("indexing %s: %w", absPath, err)
	}

	mi.mu.Lock()
	mi.repos[prefix] = &RepoMetadata{
		RepoPrefix:    prefix,
		RootPath:      absPath,
		Identity:      identity,
		LastIndexTime: time.Now(),
		FileCount:     result.FileCount,
		NodeCount:     result.NodeCount,
		EdgeCount:     result.EdgeCount,
		ParseErrors:   result.Errors,
		FileMtimes:    idx.FileMtimes(),
	}
	mi.indexers[prefix] = idx
	mi.mu.Unlock()

	// Add to global config.
	entry.Path = absPath
	if err := mi.configMgr.Global().AddRepo(entry); err != nil {
		mi.logger.Warn("failed to add repo to config", zap.Error(err))
	}

	mi.ReconcileContractEdges()

	return result, nil
}

// ReconcileRepoCtx registers a repo that already has nodes in the graph
// (typically restored from a daemon snapshot) and brings it back into
// agreement with the filesystem without a full re-index. priorMtimes
// carries the mtimes recorded at the time the snapshot was taken;
// IncrementalReindex uses them to detect files that changed offline
// (re-indexes) and files that were deleted offline (evicts).
//
// Falls back to TrackRepoCtx when the repo is not yet tracked AND no
// prior mtimes are available — in that case there's nothing to
// reconcile against and a full index is the correct path.
func (mi *MultiIndexer) ReconcileRepoCtx(ctx context.Context, entry config.RepoEntry, priorMtimes map[string]int64) (*IndexResult, error) {
	start := time.Now()

	absPath, err := filepath.Abs(entry.Path)
	if err != nil {
		return nil, fmt.Errorf("resolving path %s: %w", entry.Path, err)
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return nil, fmt.Errorf("path does not exist: %s", absPath)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", absPath)
	}
	identity, err := DetectIdentity(absPath)
	if err != nil {
		return nil, fmt.Errorf("detecting identity for %s: %w", absPath, err)
	}

	prefix := config.ResolvePrefix(entry)
	if prefix == "" || prefix == "." {
		prefix = identity.RepoPrefix
	}

	// Already tracked — nothing to do.
	mi.mu.RLock()
	_, exists := mi.repos[prefix]
	mi.mu.RUnlock()
	if exists {
		return nil, nil
	}

	// Fall back to full TrackRepoCtx when we have no prior mtimes:
	// there's nothing meaningful to reconcile against, and
	// IncrementalReindex would treat every file as stale, producing
	// the same duplicate-writes problem we're fixing.
	if len(priorMtimes) == 0 {
		return mi.TrackRepoCtx(ctx, entry)
	}

	totalConfigured := 1
	if mi.configMgr != nil {
		totalConfigured = len(mi.configMgr.Global().Repos)
	}
	willBeMultiRepo := len(mi.repos)+1 >= 2 || totalConfigured >= 2

	// Pick up `.gortex.yaml` workspace/project declarations on
	// reconcile too — config can change between sessions, and warmup-
	// time reconcile is the path that runs after a daemon restart.
	if mi.configMgr != nil {
		mi.configMgr.LoadWorkspaceConfig(prefix, absPath)
	}

	cfg := mi.configMgr.GetRepoConfig(prefix)
	idx := New(mi.graph, mi.registry, cfg.Index, mi.logger)
	idx.search = mi.search
	if mi.embedder != nil {
		idx.SetEmbedder(mi.embedder)
	}
	if willBeMultiRepo {
		idx.SetRepoPrefix(prefix)
	}
	entryCopy := entry
	idx.SetWorkspaceID(resolveWorkspaceID(&entryCopy, cfg, prefix))
	idx.SetProjectID(resolveProjectID(&entryCopy, cfg, prefix))
	idx.SetRootPath(absPath)
	idx.SetFileMtimes(priorMtimes)

	result, err := idx.IncrementalReindex(absPath)
	if err != nil {
		return nil, fmt.Errorf("reconciling %s: %w", absPath, err)
	}

	mi.mu.Lock()
	mi.repos[prefix] = &RepoMetadata{
		RepoPrefix:    prefix,
		RootPath:      absPath,
		Identity:      identity,
		LastIndexTime: time.Now(),
		FileCount:     result.FileCount,
		NodeCount:     result.NodeCount,
		EdgeCount:     result.EdgeCount,
		ParseErrors:   result.Errors,
		FileMtimes:    idx.FileMtimes(),
	}
	mi.indexers[prefix] = idx
	mi.mu.Unlock()

	entry.Path = absPath
	if err := mi.configMgr.Global().AddRepo(entry); err != nil {
		mi.logger.Warn("failed to add repo to config", zap.Error(err))
	}

	mi.ReconcileContractEdges()

	mi.logger.Info("daemon: reconciled repo from snapshot",
		zap.String("prefix", prefix),
		zap.Int("stale_files_reindexed", result.FileCount),
		zap.Duration("elapsed", time.Since(start)))

	return result, nil
}

// ReconcileAll runs IncrementalReindex on every tracked repo. Used by
// the daemon's periodic janitor to catch files whose mutations slipped
// past fsnotify — inotify watch limits, NFS / SMB mounts, kernel event
// queue overflow, or daemon downtime where edits happened while nobody
// was listening. Cheap on steady-state repos (one filepath.WalkDir +
// per-file os.Stat per repo), and correctness is self-healing: whatever
// was missed gets picked up on the next tick.
//
// Returns a map of prefix → IndexResult for logging / metrics. Errors
// per repo are logged and do not abort the rest of the sweep — a broken
// permission bit on one repo should not starve reconciliation on the
// others.
func (mi *MultiIndexer) ReconcileAll() map[string]*IndexResult {
	mi.mu.RLock()
	prefixes := make([]string, 0, len(mi.indexers))
	for p := range mi.indexers {
		prefixes = append(prefixes, p)
	}
	mi.mu.RUnlock()

	results := make(map[string]*IndexResult, len(prefixes))
	for _, prefix := range prefixes {
		mi.mu.RLock()
		idx, ok := mi.indexers[prefix]
		meta, metaOK := mi.repos[prefix]
		mi.mu.RUnlock()
		if !ok || !metaOK || meta == nil || meta.RootPath == "" {
			continue
		}
		result, err := idx.IncrementalReindex(meta.RootPath)
		if err != nil {
			mi.logger.Warn("janitor: reconcile failed",
				zap.String("prefix", prefix), zap.Error(err))
			continue
		}
		if result != nil && result.FileCount > 0 {
			mi.logger.Info("janitor: reconciled repo",
				zap.String("prefix", prefix),
				zap.Int("stale_files_reindexed", result.FileCount))
		}
		results[prefix] = result

		// Keep RepoMetadata.FileMtimes in sync so the next snapshot
		// picks up the reconciled mtimes.
		mi.mu.Lock()
		if m, ok := mi.repos[prefix]; ok && m != nil {
			m.FileMtimes = idx.FileMtimes()
			m.LastIndexTime = time.Now()
		}
		mi.mu.Unlock()
	}

	if len(results) > 0 {
		mi.ReconcileContractEdges()
	}
	return results
}

// UntrackRepo evicts a repo from the graph and removes it from config.
func (mi *MultiIndexer) UntrackRepo(repoPrefix string) (int, int) {
	mi.mu.Lock()
	meta, ok := mi.repos[repoPrefix]
	if !ok {
		mi.mu.Unlock()
		return 0, 0
	}
	delete(mi.repos, repoPrefix)
	delete(mi.indexers, repoPrefix)
	mi.mu.Unlock()

	nodesRemoved, edgesRemoved := mi.graph.EvictRepo(repoPrefix)

	// Remove from global config.
	if meta.RootPath != "" {
		if err := mi.configMgr.Global().RemoveRepo(meta.RootPath); err != nil {
			mi.logger.Warn("failed to remove repo from config",
				zap.String("prefix", repoPrefix), zap.Error(err))
		}
	}

	mi.ReconcileContractEdges()

	return nodesRemoved, edgesRemoved
}

// GetMetadata returns the metadata for a specific repo, or nil if not found.
func (mi *MultiIndexer) GetMetadata(repoPrefix string) *RepoMetadata {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	return mi.repos[repoPrefix]
}

// AllMetadata returns a copy of all repo metadata.
func (mi *MultiIndexer) AllMetadata() map[string]*RepoMetadata {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	out := make(map[string]*RepoMetadata, len(mi.repos))
	for k, v := range mi.repos {
		out[k] = v
	}
	return out
}

// IsMultiRepo returns true when more than one repo is tracked.
func (mi *MultiIndexer) IsMultiRepo() bool {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	return len(mi.repos) > 1
}

// RepoForFile returns the repo prefix for a given file path by checking
// which repo root contains it. Returns empty string if no match.
func (mi *MultiIndexer) RepoForFile(filePath string) string {
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return ""
	}

	mi.mu.RLock()
	defer mi.mu.RUnlock()

	var bestPrefix string
	var bestLen int

	for prefix, meta := range mi.repos {
		if strings.HasPrefix(absPath, meta.RootPath+string(filepath.Separator)) || absPath == meta.RootPath {
			if len(meta.RootPath) > bestLen {
				bestLen = len(meta.RootPath)
				bestPrefix = prefix
			}
		}
	}

	return bestPrefix
}

// GetIndexer returns the Indexer for a specific repo prefix, or nil.
func (mi *MultiIndexer) GetIndexer(repoPrefix string) *Indexer {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	return mi.indexers[repoPrefix]
}

// ResolveFilePath takes a repo-prefixed relative path (e.g. "ade/internal/foo.go")
// and returns the absolute filesystem path by looking up the repo's root directory.
// Returns empty string if the repo prefix is not found.
func (mi *MultiIndexer) ResolveFilePath(prefixedPath string) string {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	for prefix, meta := range mi.repos {
		if strings.HasPrefix(prefixedPath, prefix+"/") {
			relPath := strings.TrimPrefix(prefixedPath, prefix+"/")
			return filepath.Join(meta.RootPath, relPath)
		}
	}
	return ""
}

// RepoRoot returns the local filesystem root for the given repo prefix.
// ok is true only when the prefix is registered AND meta.RootPath is non-empty.
// Caller is responsible for joining repo-relative file paths against the root.
func (mi *MultiIndexer) RepoRoot(repoPrefix string) (string, bool) {
	if repoPrefix == "" {
		return "", false
	}
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	meta, ok := mi.repos[repoPrefix]
	if !ok || meta == nil || meta.RootPath == "" {
		return "", false
	}
	return meta.RootPath, true
}

// MergedContractRegistry combines contract registries from all per-repo
// indexers into a single registry. In multi-repo mode each repo's indexer
// runs extractContracts independently; this merges the results.
func (mi *MultiIndexer) MergedContractRegistry() *contracts.Registry {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	merged := contracts.NewRegistry()
	for repoPrefix, idx := range mi.indexers {
		cr := idx.ContractRegistry()
		if cr == nil {
			continue
		}
		// Re-stamp the workspace/project slugs from the indexer
		// alongside the repo prefix on merge. The contracts already
		// carry these slugs from
		// their source registry, but AddAllScoped is idempotent (skips
		// non-empty existing values) so this stays correct even if a
		// future code path forgets the stamp on first insert.
		merged.AddAllScoped(cr.All(), repoPrefix, idx.WorkspaceID(), idx.ProjectID())
	}
	return merged
}

// attachInlinedShapes folds the field shape of each contract's
// response_type / request_type into the contract's Meta so the
// dashboard can render the expanded field list. Targets contracts
// where the type has been resolved to a graph node ID (contains
// "::") AND the type node has a shape stored in its Meta.
//
// Type-shape extraction normally runs in commitContracts via
// snapshotContractShapes + inlineEnvelopeShapes — but those passes
// run during the initial extract and miss contracts added later by
// InlineWrappers. This is the post-inline equivalent: it doesn't
// re-extract shapes (the type nodes already have them from
// snapshotContractShapes if they were referenced anywhere), it just
// attaches them to the new contract entries.
func (mi *MultiIndexer) attachInlinedShapes(cr *contracts.Registry, g *graph.Graph) {
	idsToTouch := map[string]bool{}
	for _, c := range cr.All() {
		if c.Meta == nil {
			continue
		}
		for _, key := range []string{"response_type", "request_type"} {
			if v, _ := c.Meta[key].(string); v != "" && strings.Contains(v, "::") {
				idsToTouch[c.ID] = true
				break
			}
		}
		if env, ok := c.Meta["response_envelope"].([]map[string]any); ok && len(env) > 0 {
			// Touch any contract that has an envelope, even when
			// the rows still carry bare type names — the loop below
			// upgrades them. Otherwise we skip them and lose the
			// shape attachment for sibling-file types.
			idsToTouch[c.ID] = true
			_ = env
		}
	}
	srcCache := map[string][]byte{}
	resolveShape := func(typeID string) any {
		if typeID == "" || !strings.Contains(typeID, "::") {
			return nil
		}
		node := g.GetNode(typeID)
		if node == nil {
			return nil
		}
		if node.Kind != graph.KindType && node.Kind != graph.KindInterface {
			return nil
		}
		if node.Meta == nil {
			node.Meta = map[string]any{}
		}
		if shape, ok := node.Meta["shape"]; ok && shape != nil {
			return shape
		}
		// Lazy-extract: snapshotContractShapes only walks types
		// referenced by the initial bulk extract. Types referenced
		// ONLY by wrapper-inlined contracts need this fallback or
		// their fields stay unread.
		src := srcCache[node.FilePath]
		if src == nil {
			data, ok := mi.readNodeSource(node)
			if !ok {
				srcCache[node.FilePath] = []byte{}
				return nil
			}
			src = data
			srcCache[node.FilePath] = src
		}
		if len(src) == 0 {
			return nil
		}
		extracted := contracts.ExtractShape(node.FilePath, src, node.StartLine, node.EndLine)
		if extracted == nil {
			return nil
		}
		node.Meta["shape"] = extracted
		return extracted
	}
	for id := range idsToTouch {
		items := cr.ByID(id)
		changed := false
		for i := range items {
			if items[i].Meta == nil {
				continue
			}
			// Top-level request/response type shapes.
			for _, pair := range []struct{ typeKey, shapeKey string }{
				{"response_type", "response_shape"},
				{"request_type", "request_shape"},
			} {
				if _, has := items[i].Meta[pair.shapeKey]; has {
					continue
				}
				typeID, _ := items[i].Meta[pair.typeKey].(string)
				if shape := resolveShape(typeID); shape != nil {
					items[i].Meta[pair.shapeKey] = shape
					changed = true
				}
			}
			// Envelope rows — upgrade bare type names to graph IDs
			// (so the shape lookup hits) and attach shapes.
			if env, ok := items[i].Meta["response_envelope"].([]map[string]any); ok && len(env) > 0 {
				envChanged := false
				for ri, row := range env {
					typeID, _ := row["type"].(string)
					// Upgrade bare type name → graph ID when the
					// in-file resolveTypeInFile pass left it bare
					// (the type lives in a sibling file).
					if typeID != "" && !strings.Contains(typeID, "::") {
						matches := g.FindNodesByName(typeID)
						var resolved string
						for _, n := range matches {
							if n.Kind != graph.KindType && n.Kind != graph.KindInterface {
								continue
							}
							resolved = n.ID
							if items[i].RepoPrefix != "" && strings.HasPrefix(n.ID, items[i].RepoPrefix+"/") {
								break // prefer same-repo
							}
						}
						if resolved != "" {
							env[ri]["type"] = resolved
							typeID = resolved
							envChanged = true
						}
					}
					if _, has := row["shape"]; has {
						continue
					}
					if shape := resolveShape(typeID); shape != nil {
						env[ri]["shape"] = shape
						envChanged = true
					}
				}
				if envChanged {
					items[i].Meta["response_envelope"] = env
					changed = true
				}
			}
		}
		if changed {
			cr.ReplaceByID(id, items)
		}
	}
}

// readNodeSource returns the source bytes of the file the node lives
// in, resolving the repo prefix to a real disk path via tracked-repo
// metadata. Mirrors wrapperSourceReader's path-resolution dance.
func (mi *MultiIndexer) readNodeSource(node *graph.Node) ([]byte, bool) {
	if node == nil || node.FilePath == "" {
		return nil, false
	}
	rel := node.FilePath
	if node.RepoPrefix != "" {
		meta := mi.GetMetadata(node.RepoPrefix)
		if meta == nil || meta.RootPath == "" {
			return nil, false
		}
		rel = strings.TrimPrefix(rel, node.RepoPrefix+"/")
		data, err := os.ReadFile(filepath.Join(meta.RootPath, rel))
		if err != nil {
			return nil, false
		}
		return data, true
	}
	for _, m := range mi.AllMetadata() {
		data, err := os.ReadFile(filepath.Join(m.RootPath, rel))
		if err == nil {
			return data, true
		}
	}
	return nil, false
}

// wrapperSourceReader returns a SourceReader closure that maps a graph
// node back to its on-disk bytes by joining the node's repo-relative
// FilePath with the repo's RootPath from MultiIndexer metadata. In
// single-repo mode (no RepoPrefix on nodes) the indexer's sole root is
// used. Read results are memoized inside the closure so multi-caller
// wrappers don't trigger N disk reads per file.
func (mi *MultiIndexer) wrapperSourceReader() contracts.SourceReader {
	cache := make(map[string][]byte)
	miss := make(map[string]struct{})

	readFile := func(absPath string) ([]byte, bool) {
		if b, ok := cache[absPath]; ok {
			return b, true
		}
		if _, skipped := miss[absPath]; skipped {
			return nil, false
		}
		b, err := os.ReadFile(absPath)
		if err != nil {
			miss[absPath] = struct{}{}
			return nil, false
		}
		cache[absPath] = b
		return b, true
	}

	return func(n *graph.Node) ([]byte, bool) {
		if n == nil || n.FilePath == "" {
			return nil, false
		}
		// Multi-repo case: strip "<repo-prefix>/" from the FilePath and
		// join with the repo's recorded root.
		if n.RepoPrefix != "" {
			meta := mi.GetMetadata(n.RepoPrefix)
			if meta == nil || meta.RootPath == "" {
				return nil, false
			}
			rel := strings.TrimPrefix(n.FilePath, n.RepoPrefix+"/")
			return readFile(filepath.Join(meta.RootPath, rel))
		}
		// Single-repo fallback: a node without a RepoPrefix carries its
		// path relative to the sole indexer root. Try each known repo
		// root since the single-repo path wraps through indexSingleRepo.
		for _, meta := range mi.AllMetadata() {
			if b, ok := readFile(filepath.Join(meta.RootPath, n.FilePath)); ok {
				return b, true
			}
		}
		// Last resort: treat FilePath as already absolute (tests).
		return readFile(n.FilePath)
	}
}

// ReconcileContractEdges walks the merged contract registry, runs the
// consumer↔provider matcher, and writes the results into the graph as
// EdgeMatches edges pointing from consumer-contract nodes to their matched
// provider-contract nodes. Every call evicts the prior set of EdgeMatches
// first so the edges stay in sync with the current contract view; that's
// correct after track / untrack / re-index / watcher re-scan. Returns the
// number of match edges added so callers can log or test the effect.
//
// This is what makes get_call_chain traverse service boundaries: without a
// persisted contract→contract edge, the matcher's result is only visible
// via the `contracts check` tool and traversals stop at each service's
// boundary.
func (mi *MultiIndexer) ReconcileContractEdges() int {
	g := mi.Graph()
	if g == nil {
		return 0
	}

	// Evict any prior EdgeMatches so the graph reflects only the current
	// registry. Collect first, remove second — we're mutating the same
	// out-edge lists we'd be iterating otherwise.
	type edgeKey struct{ from, to string }
	var stale []edgeKey
	for _, e := range g.AllEdges() {
		if e.Kind == graph.EdgeMatches {
			stale = append(stale, edgeKey{e.From, e.To})
		}
	}
	for _, k := range stale {
		g.RemoveEdge(k.from, k.to, graph.EdgeMatches)
	}

	merged := mi.MergedContractRegistry()
	if merged == nil {
		return 0
	}

	// Inline HTTP wrapper callers (T2.4). Codebases that route every
	// endpoint through a helper like `request(path, ...)` produce one
	// parametric consumer contract per wrapper at extraction time —
	// useless for matching. InlineWrappers walks incoming call edges
	// of each wrapper, re-reads the caller's source, and emits a
	// specific consumer contract per literal path. The returned
	// contracts are also persisted into their owning repo's per-repo
	// registry so subsequent `contracts list`/`check` calls see them
	// (MergedContractRegistry rebuilds fresh each call and would
	// otherwise lose them).
	inlined := contracts.InlineWrappers(merged, g, mi.wrapperSourceReader())
	if len(inlined) > 0 {
		mi.mu.RLock()
		for _, c := range inlined {
			if c.RepoPrefix == "" {
				continue
			}
			idx, ok := mi.indexers[c.RepoPrefix]
			if !ok {
				continue
			}
			cr := idx.ContractRegistry()
			if cr == nil {
				continue
			}
			// Skip if the same contract is already persisted —
			// ReconcileContractEdges runs on every repo change, and
			// appending the same inlined contract on every pass would
			// blow up the registry with duplicates. Compare on the
			// Registry.All() dedupe key.
			alreadyPersisted := false
			for _, existing := range cr.ByID(c.ID) {
				if existing.SymbolID == c.SymbolID &&
					existing.FilePath == c.FilePath &&
					existing.Role == c.Role {
					alreadyPersisted = true
					break
				}
			}
			if !alreadyPersisted {
				cr.Add(c)
			}
		}
		mi.mu.RUnlock()

		// Wrapper-inlined contracts arrive AFTER commitContracts ran
		// its UpgradeBareTypeRefs pass, so their response_type /
		// request_type still carries bare names like "ToolInfo".
		// Re-run the upgrade against the merged graph so downstream
		// snapshotContractShapes finds the type node and the
		// dashboard sees fields instead of a string.
		mi.mu.RLock()
		lookup := func(name, repoHint string) []string {
			matches := mi.graph.FindNodesByName(name)
			if len(matches) == 0 {
				return nil
			}
			ids := make([]string, 0, len(matches))
			for _, n := range matches {
				if n.Kind != graph.KindType && n.Kind != graph.KindInterface {
					continue
				}
				ids = append(ids, n.ID)
			}
			// Prefer same-repo when multiple match.
			if len(ids) > 1 && repoHint != "" {
				var sameRepo []string
				for _, id := range ids {
					if strings.HasPrefix(id, repoHint+"/") {
						sameRepo = append(sameRepo, id)
					}
				}
				if len(sameRepo) > 0 {
					return sameRepo
				}
			}
			return ids
		}
		for _, idx := range mi.indexers {
			cr := idx.ContractRegistry()
			if cr != nil {
				cr.UpgradeBareTypeRefs(lookup)
			}
		}
		// UpgradeBareTypeRefs leaves names with ≥2 candidates alone
		// (e.g. a TS app declaring `DashboardSnapshot` in both
		// `lib/schema.ts` and `lib/types.ts`). disambiguateBareTypesViaImports
		// re-reads the consumer's source, parses its `import` lines,
		// and picks the candidate whose graph FilePath matches an
		// imported module. Runs before attachInlinedShapes so the
		// shape attachment sees fully-qualified IDs.
		for _, idx := range mi.indexers {
			cr := idx.ContractRegistry()
			if cr != nil {
				mi.disambiguateBareTypesViaImports(cr, mi.graph)
			}
		}
		// Now that response_type / request_type point at real graph
		// nodes, fold each referenced type's shape (struct fields)
		// into the contract's Meta so the dashboard renders the
		// expanded field list instead of just the type name. Mirrors
		// what snapshotContractShapes + inlineEnvelopeShapes do for
		// initially-extracted contracts.
		for _, idx := range mi.indexers {
			cr := idx.ContractRegistry()
			if cr == nil {
				continue
			}
			mi.attachInlinedShapes(cr, mi.graph)
		}
		mi.mu.RUnlock()
	}

	// Bind provider-contract SymbolIDs that came from spec files
	// (.proto for gRPC, OpenAPI YAML/JSON for HTTP). Without this
	// step the matcher finds pairs but the bridge-emission check
	// below skips them because provider SymbolID is empty. Must run
	// before Match so Match sees the updated records.
	contracts.BindProviderSymbols(merged, g)

	result := contracts.Match(merged)
	added := 0
	for _, m := range result.Matched {
		// Connect the consumer's enclosing symbol directly to the
		// provider's enclosing symbol. Contract nodes in the graph are
		// deduped by Contract.ID, so a provider and a consumer that share
		// the same ID collapse into one node — a contract→contract edge
		// would be a self-loop. Symbol→symbol bypasses that and gives
		// get_call_chain the traversal it needs: saveTuck (extension) →
		// Handler.CreateTuck (core-api).
		if m.Provider.SymbolID == "" || m.Consumer.SymbolID == "" {
			continue
		}
		if m.Provider.SymbolID == m.Consumer.SymbolID {
			continue
		}
		g.AddEdge(&graph.Edge{
			From:            m.Consumer.SymbolID,
			To:              m.Provider.SymbolID,
			Kind:            graph.EdgeMatches,
			FilePath:        m.Consumer.FilePath,
			Line:            m.Consumer.Line,
			Confidence:      1.0,
			ConfidenceLabel: "EXTRACTED",
			Origin:          graph.OriginASTResolved,
			CrossRepo:       m.CrossRepo,
		})
		added++
	}
	return added
}

// Graph returns the underlying shared graph.
func (mi *MultiIndexer) Graph() *graph.Graph {
	return mi.graph
}

// Search returns the shared search backend.
func (mi *MultiIndexer) Search() search.Backend {
	return mi.search
}

// AutoDetectRepos walks immediate subdirectories of parentPath looking for
// .git directories. If parentPath itself is a Git repo, it returns a single
// entry (the caller should index it as single-repo). If zero Git repos are
// found, it returns nil so the caller can fall back to single-repo mode.
// This is gated by the workspace.auto_detect config flag.
func (mi *MultiIndexer) AutoDetectRepos(parentPath string) []config.RepoEntry {
	absPath, err := filepath.Abs(parentPath)
	if err != nil {
		mi.logger.Warn("auto-detect: failed to resolve path", zap.String("path", parentPath), zap.Error(err))
		return nil
	}

	// If the path itself is a Git repo, return it as a single repo.
	if isGitRepo(absPath) {
		return []config.RepoEntry{{
			Path: absPath,
			Name: filepath.Base(absPath),
		}}
	}

	// Walk immediate subdirectories (not recursive) for .git dirs.
	entries, err := os.ReadDir(absPath)
	if err != nil {
		mi.logger.Warn("auto-detect: failed to read directory", zap.String("path", absPath), zap.Error(err))
		return nil
	}

	var repos []config.RepoEntry
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		subDir := filepath.Join(absPath, entry.Name())
		if isGitRepo(subDir) {
			repos = append(repos, config.RepoEntry{
				Path: subDir,
				Name: entry.Name(), // Derive RepoPrefix from subdirectory name.
			})
		}
	}

	// If zero Git repos found, return nil — caller falls back to single-repo.
	if len(repos) == 0 {
		return nil
	}

	return repos
}

// isGitRepo checks whether the given directory contains a .git subdirectory.
func isGitRepo(dir string) bool {
	info, err := os.Stat(filepath.Join(dir, ".git"))
	if err != nil {
		return false
	}
	return info.IsDir()
}
