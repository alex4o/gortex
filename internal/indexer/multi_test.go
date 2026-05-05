package indexer

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"pgregory.net/rapid"

	"github.com/zzet/gortex/internal/config"
	"github.com/zzet/gortex/internal/graph"
	"github.com/zzet/gortex/internal/parser"
	"github.com/zzet/gortex/internal/parser/languages"
	"github.com/zzet/gortex/internal/search"
)

// newTestConfigManager creates a ConfigManager with an empty GlobalConfig
// pointing at a temp file so Save() doesn't touch the real config.
func newTestConfigManager(t *testing.T) *config.ConfigManager {
	t.Helper()
	tmpFile := filepath.Join(t.TempDir(), "config.yaml")
	cm, err := config.NewConfigManager(tmpFile)
	require.NoError(t, err)
	return cm
}

// newTestRegistry returns a parser registry with Go support.
func newTestRegistry() *parser.Registry {
	reg := parser.NewRegistry()
	reg.Register(languages.NewGoExtractor())
	return reg
}

// setupRepoDir creates a temp directory with a Go file for testing.
func setupRepoDir(t *testing.T, name string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), name)
	require.NoError(t, os.MkdirAll(dir, 0o755))
	writeFile(t, filepath.Join(dir, "main.go"), `package main

func Hello() {}
`)
	return dir
}

func TestNewMultiIndexer(t *testing.T) {
	g := graph.New()
	reg := newTestRegistry()
	s := search.NewBM25()
	cm := newTestConfigManager(t)

	mi := NewMultiIndexer(g, reg, s, cm, zap.NewNop())
	require.NotNil(t, mi)
	assert.False(t, mi.IsMultiRepo())
	assert.Empty(t, mi.AllMetadata())
	assert.Same(t, g, mi.Graph())
	assert.Same(t, s, mi.Search())
}

func TestMultiIndexer_IndexAll_SingleRepo(t *testing.T) {
	dir := setupRepoDir(t, "myrepo")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{{Path: dir, Name: "myrepo"}},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	results, err := mi.IndexAll()
	require.NoError(t, err)
	require.Contains(t, results, "myrepo")

	res := results["myrepo"]
	assert.Greater(t, res.NodeCount, 0)
	assert.Greater(t, res.FileCount, 0)

	// Single repo: should NOT be multi-repo mode.
	assert.False(t, mi.IsMultiRepo())

	// Nodes should NOT have repo prefix in single-repo mode (backward compat).
	for _, n := range g.AllNodes() {
		assert.Empty(t, n.RepoPrefix, "single-repo nodes should not have RepoPrefix")
	}

	// Metadata should be populated.
	meta := mi.GetMetadata("myrepo")
	require.NotNil(t, meta)
	assert.Equal(t, "myrepo", meta.RepoPrefix)
	assert.Equal(t, dir, meta.RootPath)
	assert.Greater(t, meta.FileCount, 0)
}

func TestMultiIndexer_IndexAll_MultiRepo(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	results, err := mi.IndexAll()
	require.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Contains(t, results, "repo-a")
	assert.Contains(t, results, "repo-b")

	// Multi-repo mode.
	assert.True(t, mi.IsMultiRepo())

	// Nodes should have repo prefix set.
	for _, n := range g.AllNodes() {
		assert.NotEmpty(t, n.RepoPrefix, "multi-repo nodes should have RepoPrefix")
		assert.Contains(t, []string{"repo-a", "repo-b"}, n.RepoPrefix)
	}

	// Both repos should have metadata.
	assert.NotNil(t, mi.GetMetadata("repo-a"))
	assert.NotNil(t, mi.GetMetadata("repo-b"))
	assert.Len(t, mi.AllMetadata(), 2)
}

func TestMultiIndexer_IndexAll_SingleRepoLoadsWorkspaceExclude(t *testing.T) {
	dir := setupRepoDir(t, "myrepo")
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "ignored"), 0o755))
	writeFile(t, filepath.Join(dir, "ignored", "ignored.go"), "package main\nfunc Ignored() {}\n")
	writeFile(t, filepath.Join(dir, ".gortex.yaml"), "exclude:\n  - ignored/**\n")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{Repos: []config.RepoEntry{{Path: dir, Name: "myrepo"}}}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	for _, n := range g.AllNodes() {
		assert.NotContains(t, n.FilePath, "ignored/ignored.go")
		assert.NotContains(t, n.ID, "Ignored")
	}
}

func TestMultiIndexer_IndexAll_MultiRepoLoadsWorkspaceExclude(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")
	require.NoError(t, os.MkdirAll(filepath.Join(repoA, "ignored"), 0o755))
	writeFile(t, filepath.Join(repoA, "ignored", "ignored.go"), "package main\nfunc Ignored() {}\n")
	writeFile(t, filepath.Join(repoA, ".gortex.yaml"), "exclude:\n  - ignored/**\n")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	for _, n := range g.AllNodes() {
		assert.NotContains(t, n.FilePath, "ignored/ignored.go")
		assert.NotContains(t, n.ID, "Ignored")
	}
}

func TestMultiIndexer_IndexRepo(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	nodesBefore := g.NodeCount()
	repoBNodes := len(g.GetRepoNodes("repo-b"))

	// Re-index repo-a.
	result, err := mi.IndexRepo("repo-a")
	require.NoError(t, err)
	assert.Greater(t, result.NodeCount, 0)

	// Repo B should be unchanged.
	assert.Equal(t, repoBNodes, len(g.GetRepoNodes("repo-b")))
	// Total should be roughly the same (re-indexed, not duplicated).
	assert.InDelta(t, nodesBefore, g.NodeCount(), 2)
}

func TestMultiIndexer_IndexRepo_NotFound(t *testing.T) {
	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err := mi.IndexRepo("nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "repository not found")
}

func TestMultiIndexer_TrackRepo(t *testing.T) {
	dir := setupRepoDir(t, "tracked")

	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	result, err := mi.TrackRepo(config.RepoEntry{Path: dir, Name: "tracked"})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Greater(t, result.NodeCount, 0)

	// Should be in metadata.
	meta := mi.GetMetadata("tracked")
	require.NotNil(t, meta)
	assert.Equal(t, "tracked", meta.RepoPrefix)
	assert.Equal(t, dir, meta.RootPath)

	// Track again — should return nil (already tracked).
	result2, err := mi.TrackRepo(config.RepoEntry{Path: dir, Name: "tracked"})
	require.NoError(t, err)
	assert.Nil(t, result2)
}

// TestMultiIndexer_TrackRepo_SearchSpansAllRepos verifies that scoping
// buildSearchIndex to the current repo (the perf fix that drops the
// O(N²) re-index of every prior repo's nodes on every TrackRepo call)
// does not regress search recall. After three repos are tracked, the
// shared search backend must still find symbols defined in the first,
// second, and third repos — earlier-tracked entries are not lost
// because each repo's TrackRepo pass already added its own nodes.
func TestMultiIndexer_TrackRepo_SearchSpansAllRepos(t *testing.T) {
	mkRepo := func(name, symbol string) string {
		dir := filepath.Join(t.TempDir(), name)
		require.NoError(t, os.MkdirAll(dir, 0o755))
		writeFile(t, filepath.Join(dir, "main.go"), fmt.Sprintf("package main\n\nfunc %s() {}\n", symbol))
		return dir
	}
	dirA := mkRepo("repo-aaa", "AlphaSymbolUnique")
	dirB := mkRepo("repo-bbb", "BetaSymbolUnique")
	dirC := mkRepo("repo-ccc", "GammaSymbolUnique")

	// Pre-populate the global config so willBeMultiRepo trips on the
	// first TrackRepo too — that's the production warmup-loop path.
	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{Repos: []config.RepoEntry{
		{Path: dirA, Name: "repo-aaa"},
		{Path: dirB, Name: "repo-bbb"},
		{Path: dirC, Name: "repo-ccc"},
	}}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())
	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	for _, e := range []config.RepoEntry{
		{Path: dirA, Name: "repo-aaa"},
		{Path: dirB, Name: "repo-bbb"},
		{Path: dirC, Name: "repo-ccc"},
	} {
		_, err := mi.TrackRepo(e)
		require.NoError(t, err)
	}

	// Query the camelCase-split tokens individually — that's how the
	// BM25 backend stores them at Add time, and Search.TokenizeQuery
	// doesn't perform the same camelCase split.
	for _, want := range []struct{ query, prefix string }{
		{"alpha", "repo-aaa"},
		{"beta", "repo-bbb"},
		{"gamma", "repo-ccc"},
	} {
		hits := mi.Search().Search(want.query, 10)
		require.NotEmpty(t, hits, "shared search backend must find %q after all repos tracked", want.query)
		assert.True(t, strings.HasPrefix(hits[0].ID, want.prefix+"/"),
			"top hit for %q should belong to %s, got %s", want.query, want.prefix, hits[0].ID)
	}
}

// TestMultiIndexer_TrackRepo_EmptyAfterPopulated regresses the bug
// where IndexResult/RepoMetadata stamped the *whole multi-repo graph*
// counts onto an empty repo at TrackRepo time. A second tracked repo
// that contributed zero source files used to come back with the same
// node count as the populated repo because IndexResult.NodeCount was
// graph.NodeCount() rather than the per-repo contribution. Downstream
// daemon-status code multiplied search backend bytes by share = 1 for
// every empty row, attributing the entire workspace search budget to
// each empty repo. The fix: IndexResult counts come from
// graph.RepoMemoryEstimate(prefix) when repoPrefix is non-empty, and
// daemon_controller.Status no longer falls back to meta when
// RepoStats has no entry for the prefix.
func TestMultiIndexer_TrackRepo_EmptyAfterPopulated(t *testing.T) {
	populated := setupRepoDir(t, "populated")
	empty := filepath.Join(t.TempDir(), "empty")
	require.NoError(t, os.MkdirAll(empty, 0o755))

	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	first, err := mi.TrackRepo(config.RepoEntry{Path: populated, Name: "populated"})
	require.NoError(t, err)
	require.NotNil(t, first)
	require.Greater(t, first.NodeCount, 0, "populated repo must contribute nodes")

	globalNodesBefore := g.NodeCount()
	require.Greater(t, globalNodesBefore, 0)

	second, err := mi.TrackRepo(config.RepoEntry{Path: empty, Name: "empty"})
	require.NoError(t, err)
	require.NotNil(t, second)

	// Per-repo IndexResult must not echo the workspace-wide graph total.
	assert.Equal(t, 0, second.NodeCount,
		"empty repo IndexResult.NodeCount must be 0, got %d (global graph has %d)",
		second.NodeCount, globalNodesBefore)
	assert.Equal(t, 0, second.EdgeCount,
		"empty repo IndexResult.EdgeCount must be 0, got %d", second.EdgeCount)

	// RepoMetadata is stamped from IndexResult, so it must agree.
	emptyMeta := mi.GetMetadata("empty")
	require.NotNil(t, emptyMeta)
	assert.Equal(t, 0, emptyMeta.NodeCount, "empty repo metadata must record 0 nodes")
	assert.Equal(t, 0, emptyMeta.EdgeCount, "empty repo metadata must record 0 edges")

	// And the populated repo's metadata must reflect only its own
	// contribution, not its size + everything that came after.
	populatedMeta := mi.GetMetadata("populated")
	require.NotNil(t, populatedMeta)
	assert.Greater(t, populatedMeta.NodeCount, 0)
	assert.LessOrEqual(t, populatedMeta.NodeCount, globalNodesBefore,
		"populated repo metadata must not exceed graph total at its track time")
}

func TestMultiIndexer_TrackRepo_InvalidPath(t *testing.T) {
	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err := mi.TrackRepo(config.RepoEntry{Path: "/nonexistent/path/xyz"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "path does not exist")
}

func TestMultiIndexer_TrackRepo_NotADirectory(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "file.txt")
	require.NoError(t, os.WriteFile(tmpFile, []byte("hello"), 0o644))

	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err := mi.TrackRepo(config.RepoEntry{Path: tmpFile})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not a directory")
}

func TestMultiIndexer_UntrackRepo(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	repoBNodesBefore := len(g.GetRepoNodes("repo-b"))
	assert.Greater(t, len(g.GetRepoNodes("repo-a")), 0)

	// Untrack repo-a.
	nodesRemoved, edgesRemoved := mi.UntrackRepo("repo-a")
	assert.Greater(t, nodesRemoved, 0)
	_ = edgesRemoved

	// Repo-a should be gone.
	assert.Nil(t, mi.GetMetadata("repo-a"))
	assert.Empty(t, g.GetRepoNodes("repo-a"))

	// Repo-b should be unchanged.
	assert.Equal(t, repoBNodesBefore, len(g.GetRepoNodes("repo-b")))
}

func TestMultiIndexer_UntrackRepo_NotFound(t *testing.T) {
	g := graph.New()
	cm := newTestConfigManager(t)
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	nodesRemoved, edgesRemoved := mi.UntrackRepo("nonexistent")
	assert.Equal(t, 0, nodesRemoved)
	assert.Equal(t, 0, edgesRemoved)
}

func TestMultiIndexer_RepoForFile(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	// File in repo-a.
	assert.Equal(t, "repo-a", mi.RepoForFile(filepath.Join(repoA, "main.go")))
	// File in repo-b.
	assert.Equal(t, "repo-b", mi.RepoForFile(filepath.Join(repoB, "main.go")))
	// Unknown file.
	assert.Empty(t, mi.RepoForFile("/some/random/path.go"))
}

func TestMultiIndexer_GetIndexer(t *testing.T) {
	dir := setupRepoDir(t, "myrepo")

	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{{Path: dir, Name: "myrepo"}},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	idx := mi.GetIndexer("myrepo")
	assert.NotNil(t, idx)
	assert.Nil(t, mi.GetIndexer("nonexistent"))
}

func TestMultiIndexer_IndexAll_EmptyRepos(t *testing.T) {
	cm := newTestConfigManager(t)
	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	results, err := mi.IndexAll()
	require.NoError(t, err)
	assert.Nil(t, results)
}

// Feature: multi-repo-support, Property 6: Node ID format by mode
//
// TestPropertyNodeIDFormatByMode verifies that:
//   - In multi-repo mode (2+ repos), node IDs match <repo_prefix>/<path>::<Symbol>
//     and RepoPrefix is non-empty.
//   - In single-repo mode (1 repo), node IDs match <path>::<Symbol>
//     and RepoPrefix is empty.
func TestPropertyNodeIDFormatByMode(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		// Generate random function names for uniqueness per iteration.
		funcNameA := "Func" + rapid.StringMatching(`[A-Z][a-z]{2,6}`).Draw(rt, "funcNameA")
		funcNameB := "Func" + rapid.StringMatching(`[A-Z][a-z]{2,6}`).Draw(rt, "funcNameB")

		// Decide mode: single-repo or multi-repo.
		multiRepo := rapid.Bool().Draw(rt, "multiRepo")

		tmpBase := t.TempDir()

		if multiRepo {
			// --- Multi-repo mode: 2 repos ---
			repoA := filepath.Join(tmpBase, "repo-a")
			repoB := filepath.Join(tmpBase, "repo-b")
			require.NoError(t, os.MkdirAll(repoA, 0o755))
			require.NoError(t, os.MkdirAll(repoB, 0o755))

			writeFile(t, filepath.Join(repoA, "main.go"),
				fmt.Sprintf("package main\n\nfunc %s() {}\n", funcNameA))
			writeFile(t, filepath.Join(repoB, "main.go"),
				fmt.Sprintf("package main\n\nfunc %s() {}\n", funcNameB))

			tmpCfg := filepath.Join(tmpBase, "config.yaml")
			gc := &config.GlobalConfig{
				Repos: []config.RepoEntry{
					{Path: repoA, Name: "repo-a"},
					{Path: repoB, Name: "repo-b"},
				},
			}
			gc.SetConfigPath(tmpCfg)
			require.NoError(t, gc.Save())

			cm, err := config.NewConfigManager(tmpCfg)
			require.NoError(t, err)

			g := graph.New()
			mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

			results, err := mi.IndexAll()
			require.NoError(t, err)
			require.Len(t, results, 2)

			// Must be multi-repo mode.
			if !mi.IsMultiRepo() {
				rt.Error("expected IsMultiRepo() == true for 2 repos")
			}

			for _, n := range g.AllNodes() {
				// RepoPrefix must be non-empty.
				if n.RepoPrefix == "" {
					rt.Errorf("multi-repo node %q has empty RepoPrefix", n.ID)
				}

				// ID must start with "<repo_prefix>/".
				prefix := n.RepoPrefix + "/"
				if !strings.HasPrefix(n.ID, prefix) {
					rt.Errorf("multi-repo node ID %q does not start with prefix %q", n.ID, prefix)
				}

				// ID must contain "::" separating path from symbol (for non-file nodes).
				if n.Kind != graph.KindFile && n.Kind != graph.KindPackage && n.Kind != graph.KindImport {
					if !strings.Contains(n.ID, "::") {
						rt.Errorf("multi-repo node ID %q missing '::' separator", n.ID)
					}
				}
			}
		} else {
			// --- Single-repo mode: 1 repo ---
			repoDir := filepath.Join(tmpBase, "solo-repo")
			require.NoError(t, os.MkdirAll(repoDir, 0o755))

			writeFile(t, filepath.Join(repoDir, "main.go"),
				fmt.Sprintf("package main\n\nfunc %s() {}\n", funcNameA))

			tmpCfg := filepath.Join(tmpBase, "config.yaml")
			gc := &config.GlobalConfig{
				Repos: []config.RepoEntry{
					{Path: repoDir, Name: "solo-repo"},
				},
			}
			gc.SetConfigPath(tmpCfg)
			require.NoError(t, gc.Save())

			cm, err := config.NewConfigManager(tmpCfg)
			require.NoError(t, err)

			g := graph.New()
			mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

			results, err := mi.IndexAll()
			require.NoError(t, err)
			require.Len(t, results, 1)

			// Must NOT be multi-repo mode.
			if mi.IsMultiRepo() {
				rt.Error("expected IsMultiRepo() == false for 1 repo")
			}

			for _, n := range g.AllNodes() {
				// RepoPrefix must be empty in single-repo mode.
				if n.RepoPrefix != "" {
					rt.Errorf("single-repo node %q has non-empty RepoPrefix %q", n.ID, n.RepoPrefix)
				}

				// ID must NOT contain a repo prefix slash before the path.
				// In single-repo mode, IDs are <path>::<Symbol>, not <prefix>/<path>::<Symbol>.
				// We verify by checking no known prefix appears.
				if strings.HasPrefix(n.ID, "solo-repo/") {
					rt.Errorf("single-repo node ID %q should not have repo prefix", n.ID)
				}
			}
		}
	})
}

// Feature: multi-repo-support, Property 9: Re-index isolation
//
// TestPropertyReindexIsolation verifies that re-indexing repo A does not
// modify, remove, or add any nodes or edges belonging to repo B.
// The node count and edge count for repo B before and after re-indexing A
// are identical.
func TestPropertyReindexIsolation(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		// Generate random function names for uniqueness per iteration.
		funcNameA := "Func" + rapid.StringMatching(`[A-Z][a-z]{2,6}`).Draw(rt, "funcNameA")
		funcNameB := "Func" + rapid.StringMatching(`[A-Z][a-z]{2,6}`).Draw(rt, "funcNameB")

		tmpBase := t.TempDir()

		repoA := filepath.Join(tmpBase, "repo-a")
		repoB := filepath.Join(tmpBase, "repo-b")
		require.NoError(t, os.MkdirAll(repoA, 0o755))
		require.NoError(t, os.MkdirAll(repoB, 0o755))

		writeFile(t, filepath.Join(repoA, "main.go"),
			fmt.Sprintf("package main\n\nfunc %s() {}\n", funcNameA))
		writeFile(t, filepath.Join(repoB, "main.go"),
			fmt.Sprintf("package main\n\nfunc %s() {}\n", funcNameB))

		tmpCfg := filepath.Join(tmpBase, "config.yaml")
		gc := &config.GlobalConfig{
			Repos: []config.RepoEntry{
				{Path: repoA, Name: "repo-a"},
				{Path: repoB, Name: "repo-b"},
			},
		}
		gc.SetConfigPath(tmpCfg)
		require.NoError(t, gc.Save())

		cm, err := config.NewConfigManager(tmpCfg)
		require.NoError(t, err)

		g := graph.New()
		mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

		_, err = mi.IndexAll()
		require.NoError(t, err)

		// Record repo B's state before re-indexing A.
		repoBNodesBefore := len(g.GetRepoNodes("repo-b"))
		repoBEdgesBefore := countRepoEdges(g, "repo-b")
		// Capture repo B node IDs for identity check.
		repoBNodeIDsBefore := make(map[string]bool)
		for _, n := range g.GetRepoNodes("repo-b") {
			repoBNodeIDsBefore[n.ID] = true
		}

		// Re-index repo A.
		_, err = mi.IndexRepo("repo-a")
		require.NoError(t, err)

		// Verify repo B node count unchanged.
		repoBNodesAfter := len(g.GetRepoNodes("repo-b"))
		if repoBNodesBefore != repoBNodesAfter {
			rt.Errorf("repo-b node count changed: before=%d after=%d", repoBNodesBefore, repoBNodesAfter)
		}

		// Verify repo B edge count unchanged.
		repoBEdgesAfter := countRepoEdges(g, "repo-b")
		if repoBEdgesBefore != repoBEdgesAfter {
			rt.Errorf("repo-b edge count changed: before=%d after=%d", repoBEdgesBefore, repoBEdgesAfter)
		}

		// Verify repo B node IDs are exactly the same.
		repoBNodeIDsAfter := make(map[string]bool)
		for _, n := range g.GetRepoNodes("repo-b") {
			repoBNodeIDsAfter[n.ID] = true
		}
		for id := range repoBNodeIDsBefore {
			if !repoBNodeIDsAfter[id] {
				rt.Errorf("repo-b node %q disappeared after re-indexing repo-a", id)
			}
		}
		for id := range repoBNodeIDsAfter {
			if !repoBNodeIDsBefore[id] {
				rt.Errorf("repo-b node %q appeared after re-indexing repo-a", id)
			}
		}
	})
}

// countRepoEdges counts edges where at least one endpoint belongs to the given repo prefix.
func countRepoEdges(g *graph.Graph, repoPrefix string) int {
	prefix := repoPrefix + "/"
	count := 0
	for _, e := range g.AllEdges() {
		if strings.HasPrefix(e.From, prefix) || strings.HasPrefix(e.To, prefix) {
			count++
		}
	}
	return count
}

// --- Task 17.1: Backward compatibility verification ---

// TestSingleRepoMode_BackwardCompat verifies that single --index with no repos
// config behaves identically to the current implementation.
func TestSingleRepoMode_BackwardCompat(t *testing.T) {
	t.Run("node_ids_use_existing_format", func(t *testing.T) {
		dir := setupRepoDir(t, "myrepo")

		tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
		gc := &config.GlobalConfig{
			Repos: []config.RepoEntry{{Path: dir, Name: "myrepo"}},
		}
		gc.SetConfigPath(tmpCfg)
		require.NoError(t, gc.Save())

		cm, err := config.NewConfigManager(tmpCfg)
		require.NoError(t, err)

		g := graph.New()
		mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

		_, err = mi.IndexAll()
		require.NoError(t, err)

		// Single-repo mode: IsMultiRepo should be false.
		assert.False(t, mi.IsMultiRepo())

		// Node IDs should use existing format without prefix.
		for _, n := range g.AllNodes() {
			assert.Empty(t, n.RepoPrefix, "single-repo nodes should not have RepoPrefix")
			// IDs should NOT start with "myrepo/" prefix.
			assert.False(t, strings.HasPrefix(n.ID, "myrepo/"),
				"single-repo node ID %q should not have repo prefix", n.ID)
		}
	})

	t.Run("existing_gortex_yaml_loads_without_errors", func(t *testing.T) {
		// Create a minimal .gortex.yaml without repos/workspace sections.
		tmpDir := t.TempDir()
		cfgContent := `index:
  exclude:
    - "vendor/**"
  workers: 4
watch:
  enabled: true
  debounce_ms: 200
guards:
  rules:
    - name: test-rule
      kind: co-change
      source: "internal/parser"
      target: "internal/parser/languages"
      message: "Parser changes require language extractor test updates"
`
		cfgPath := filepath.Join(tmpDir, ".gortex.yaml")
		require.NoError(t, os.WriteFile(cfgPath, []byte(cfgContent), 0o644))

		cfg, err := config.Load(cfgPath)
		require.NoError(t, err)
		require.NotNil(t, cfg)

		// New fields should have defaults.
		assert.False(t, cfg.Multi.AutoDetect)
		// Existing fields should be loaded.
		assert.Equal(t, 4, cfg.Index.Workers)
		assert.True(t, cfg.Watch.Enabled)
		assert.Len(t, cfg.Guards.Rules, 1)
	})

	t.Run("no_global_config_required", func(t *testing.T) {
		// Loading a non-existent GlobalConfig should not error.
		gc, err := config.LoadGlobal("/nonexistent/path/config.yaml")
		require.NoError(t, err)
		require.NotNil(t, gc)
		assert.Empty(t, gc.Repos)
		assert.Empty(t, gc.Projects)
		assert.Empty(t, gc.ActiveProject)
	})

	t.Run("mcp_tools_work_without_repo_param", func(t *testing.T) {
		// In single-repo mode, the graph should be queryable without repo filtering.
		dir := setupRepoDir(t, "solo")

		tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
		gc := &config.GlobalConfig{
			Repos: []config.RepoEntry{{Path: dir, Name: "solo"}},
		}
		gc.SetConfigPath(tmpCfg)
		require.NoError(t, gc.Save())

		cm, err := config.NewConfigManager(tmpCfg)
		require.NoError(t, err)

		g := graph.New()
		mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

		_, err = mi.IndexAll()
		require.NoError(t, err)

		// Query engine should work without repo scoping.
		nodes := g.AllNodes()
		assert.Greater(t, len(nodes), 0, "should have indexed nodes")

		// All nodes should be accessible without repo prefix filtering.
		for _, n := range nodes {
			found := g.GetNode(n.ID)
			assert.NotNil(t, found, "node %q should be findable by ID", n.ID)
		}
	})
}

// --- Task 17.2: Single-to-multi-repo transition ---

// TestSingleToMultiRepoTransition verifies that when upgrading from single-repo
// to multi-repo, a full re-index generates Qualified_Node_IDs.
func TestSingleToMultiRepoTransition(t *testing.T) {
	repoA := setupRepoDir(t, "repo-a")
	repoB := setupRepoDir(t, "repo-b")

	// Step 1: Start in single-repo mode with just repo-a.
	tmpCfg := filepath.Join(t.TempDir(), "config.yaml")
	gc := &config.GlobalConfig{
		Repos: []config.RepoEntry{{Path: repoA, Name: "repo-a"}},
	}
	gc.SetConfigPath(tmpCfg)
	require.NoError(t, gc.Save())

	cm, err := config.NewConfigManager(tmpCfg)
	require.NoError(t, err)

	g := graph.New()
	mi := NewMultiIndexer(g, newTestRegistry(), search.NewBM25(), cm, zap.NewNop())

	_, err = mi.IndexAll()
	require.NoError(t, err)

	// Verify single-repo mode: no prefixes.
	assert.False(t, mi.IsMultiRepo())
	for _, n := range g.AllNodes() {
		assert.Empty(t, n.RepoPrefix)
		assert.False(t, strings.HasPrefix(n.ID, "repo-a/"))
	}

	// Step 2: Track a second repo to transition to multi-repo mode.
	result, err := mi.TrackRepo(config.RepoEntry{Path: repoB, Name: "repo-b"})
	require.NoError(t, err)
	require.NotNil(t, result)

	// Now we're in multi-repo mode.
	assert.True(t, mi.IsMultiRepo())

	// Step 3: Full re-index to generate Qualified_Node_IDs.
	// Create a new graph and multi-indexer to simulate full re-index.
	gc2 := &config.GlobalConfig{
		Repos: []config.RepoEntry{
			{Path: repoA, Name: "repo-a"},
			{Path: repoB, Name: "repo-b"},
		},
	}
	tmpCfg2 := filepath.Join(t.TempDir(), "config2.yaml")
	gc2.SetConfigPath(tmpCfg2)
	require.NoError(t, gc2.Save())

	cm2, err := config.NewConfigManager(tmpCfg2)
	require.NoError(t, err)

	g2 := graph.New()
	mi2 := NewMultiIndexer(g2, newTestRegistry(), search.NewBM25(), cm2, zap.NewNop())

	results, err := mi2.IndexAll()
	require.NoError(t, err)
	require.Len(t, results, 2)

	// All nodes should now have Qualified_Node_IDs with repo prefix.
	assert.True(t, mi2.IsMultiRepo())
	for _, n := range g2.AllNodes() {
		assert.NotEmpty(t, n.RepoPrefix, "multi-repo node %q should have RepoPrefix", n.ID)
		prefix := n.RepoPrefix + "/"
		assert.True(t, strings.HasPrefix(n.ID, prefix),
			"multi-repo node ID %q should start with %q", n.ID, prefix)
	}
}
