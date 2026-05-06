package semantic

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/zzet/gortex/internal/graph"
)

// Manager orchestrates multiple semantic providers and coordinates enrichment.
type Manager struct {
	providers []Provider
	config    Config
	logger    *zap.Logger

	mu          sync.RWMutex
	lastResults map[string]*EnrichResult // provider name → last result
}

// NewManager creates a Manager from configuration.
// It registers providers based on config, probes availability, and logs results.
func NewManager(cfg Config, logger *zap.Logger) *Manager {
	m := &Manager{
		config:      cfg,
		logger:      logger,
		lastResults: make(map[string]*EnrichResult),
	}
	return m
}

// RegisterProvider adds a provider to the manager.
func (m *Manager) RegisterProvider(p Provider) {
	m.providers = append(m.providers, p)
	m.logger.Info("semantic provider registered",
		zap.String("name", p.Name()),
		zap.Strings("languages", p.Languages()),
		zap.Bool("available", p.Available()),
	)
}

// EnrichAll runs all available providers against the graph.
// For each language, only the highest-priority available provider runs.
func (m *Manager) EnrichAll(g *graph.Graph, roots map[string]string) ([]*EnrichResult, error) {
	if !m.config.Enabled {
		return nil, nil
	}

	// Build a map of language → sorted providers (by priority from config).
	langProviders := m.selectProviders()

	var results []*EnrichResult
	runOnce := make(map[string]bool)

	for lang, provider := range langProviders {
		if !provider.Available() {
			m.logger.Debug("semantic provider unavailable, skipping",
				zap.String("provider", provider.Name()),
				zap.String("language", lang),
			)
			continue
		}

		// Run enrichment for each repo root. A provider may advertise multiple
		// languages; run it once per repo, not once per language, or LSP/SCIP
		// providers do duplicate full-graph work.
		for repoName, repoRoot := range roots {
			runKey := provider.Name() + "\x00" + repoName
			if runOnce[runKey] {
				continue
			}
			runOnce[runKey] = true

			start := time.Now()
			m.logger.Info("semantic enrichment starting",
				zap.String("provider", provider.Name()),
				zap.String("language", lang),
				zap.String("repo", repoName),
			)

			result, err := provider.Enrich(g, repoRoot)
			if err != nil {
				m.logger.Warn("semantic enrichment failed",
					zap.String("provider", provider.Name()),
					zap.String("language", lang),
					zap.Error(err),
				)
				continue
			}

			if result != nil {
				result.DurationMs = time.Since(start).Milliseconds()
				results = append(results, result)

				m.mu.Lock()
				m.lastResults[provider.Name()] = result
				m.mu.Unlock()

				m.logger.Info("semantic enrichment complete",
					zap.String("provider", provider.Name()),
					zap.String("language", lang),
					zap.Int("confirmed", result.EdgesConfirmed),
					zap.Int("added", result.EdgesAdded),
					zap.Int("refuted", result.EdgesRefuted),
					zap.Int("nodes_enriched", result.NodesEnriched),
					zap.Float64("coverage", result.CoveragePercent),
					zap.Int64("duration_ms", result.DurationMs),
				)
			}
		}
	}

	return results, nil
}

// EnrichFile runs incremental enrichment for a single file change.
func (m *Manager) EnrichFile(g *graph.Graph, repoRoot, filePath string) (*EnrichResult, error) {
	if !m.config.Enabled || !m.config.EnrichOnWatch {
		return nil, nil
	}

	langProviders := m.selectProviders()

	// Determine language from file nodes.
	nodes := g.GetFileNodes(filePath)
	if len(nodes) == 0 {
		return nil, nil
	}
	lang := nodes[0].Language

	provider, ok := langProviders[lang]
	if !ok || !provider.Available() {
		return nil, nil
	}

	return provider.EnrichFile(g, repoRoot, filePath)
}

// selectProviders returns the highest-priority available provider per language.
func (m *Manager) selectProviders() map[string]Provider {
	// Build priority map from config.
	type configEntry struct {
		name     string
		priority int
		enabled  bool
	}
	configMap := make(map[string]configEntry)
	for _, pc := range m.config.Providers {
		configMap[pc.Name] = configEntry{
			name:     pc.Name,
			priority: pc.Priority,
			enabled:  pc.Enabled,
		}
	}

	// Group providers by language with priority.
	type langCandidate struct {
		provider Provider
		priority int
	}
	langCandidates := make(map[string][]langCandidate)

	for _, p := range m.providers {
		ce, ok := configMap[p.Name()]
		if ok && !ce.enabled {
			continue
		}
		priority := 99
		if ok {
			priority = ce.priority
		}
		for _, lang := range p.Languages() {
			langCandidates[lang] = append(langCandidates[lang], langCandidate{
				provider: p,
				priority: priority,
			})
		}
	}

	// Select highest-priority (lowest number) per language.
	result := make(map[string]Provider)
	for lang, candidates := range langCandidates {
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].priority < candidates[j].priority
		})
		result[lang] = candidates[0].provider
	}

	return result
}

// Stats returns the current status of all providers.
func (m *Manager) Stats() []ProviderStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var statuses []ProviderStatus
	for _, p := range m.providers {
		for _, lang := range p.Languages() {
			status := "unavailable"
			if p.Available() {
				status = "ready"
			}

			ps := ProviderStatus{
				Name:     p.Name(),
				Language: lang,
				Status:   status,
			}

			if lr, ok := m.lastResults[p.Name()]; ok {
				ps.CoveragePercent = lr.CoveragePercent
				ps.LastResult = lr
			}

			statuses = append(statuses, ps)
		}
	}
	return statuses
}

// Close shuts down all providers.
func (m *Manager) Close() error {
	var errs []error
	for _, p := range m.providers {
		if err := p.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing %s: %w", p.Name(), err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("semantic manager close errors: %v", errs)
	}
	return nil
}

// Enabled returns whether semantic enrichment is enabled.
func (m *Manager) Enabled() bool {
	return m.config.Enabled
}

// HasProviders returns whether any providers are registered and available.
func (m *Manager) HasProviders() bool {
	for _, p := range m.providers {
		if p.Available() {
			return true
		}
	}
	return false
}
