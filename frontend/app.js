/**
 * MUST++ Frontend Application
 * 
 * Operator-grade UI for multilingual hate speech detection.
 * 
 * Design Principles:
 * - Never hide uncertainty
 * - Never oversimplify harm
 * - Safety explanations must be inspectable
 * - Latency must feel intentional
 */

(function() {
    'use strict';

    // ==================== Configuration ====================
    const CONFIG = {
        API_BASE: window.location.origin.includes('localhost') 
            ? 'http://localhost:8080' 
            : window.location.origin,
        SCRIPT_DETECT_DEBOUNCE_MS: 300,
        PROCESSING_DELAY_THRESHOLD_MS: 300,
        LABELS: ['neutral', 'offensive', 'hate'],
        HISTORY_MAX_ITEMS: 50,
        TOAST_DURATION_MS: 3000
    };

    // ==================== State ====================
    let state = {
        lastResponse: null,
        isProcessing: false,
        systemHealthy: true,
        degradedMode: false,
        scriptDetectTimeout: null,
        history: [],
        historyPanelOpen: false,
        shortcutsPanelOpen: false,
        theme: 'system'
    };

    // ==================== DOM Elements ====================
    const elements = {
        // System
        systemStatus: document.getElementById('systemStatus'),
        statusIndicator: document.getElementById('statusIndicator'),
        statusText: document.getElementById('statusText'),
        systemBanner: document.getElementById('systemBanner'),
        bannerText: document.getElementById('bannerText'),
        
        // Input Layer
        textInput: document.getElementById('textInput'),
        charCountValue: document.getElementById('charCountValue'),
        scriptValue: document.getElementById('scriptValue'),
        languageHint: document.getElementById('languageHint'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        
        // Processing
        processingIndicator: document.getElementById('processingIndicator'),
        
        // Decision Layer
        decisionLayer: document.getElementById('decisionLayer'),
        labelDisplay: document.getElementById('labelDisplay'),
        labelText: document.getElementById('labelText'),
        confidenceBar: document.getElementById('confidenceBar'),
        confidenceValue: document.getElementById('confidenceValue'),
        safetyBadge: document.getElementById('safetyBadge'),
        badgeLabel: document.getElementById('badgeLabel'),
        safetyTooltip: document.getElementById('safetyTooltip'),
        tooltipText: document.getElementById('tooltipText'),
        
        // Explanation Layer
        explanationLayer: document.getElementById('explanationLayer'),
        explanationToggle: document.getElementById('explanationToggle'),
        explanationContent: document.getElementById('explanationContent'),
        highlightedText: document.getElementById('highlightedText'),
        labelJustification: document.getElementById('labelJustification'),
        rejectedLabelsSection: document.getElementById('rejectedLabelsSection'),
        rejectedLabelsList: document.getElementById('rejectedLabelsList'),
        harmTokensSection: document.getElementById('harmTokensSection'),
        harmTokensContainer: document.getElementById('harmTokensContainer'),
        identityGroupsSection: document.getElementById('identityGroupsSection'),
        identityGroupsContainer: document.getElementById('identityGroupsContainer'),
        copyExplanationBtn: document.getElementById('copyExplanationBtn'),
        
        // System Trace Layer
        traceLayer: document.getElementById('traceLayer'),
        traceToggle: document.getElementById('traceToggle'),
        traceContent: document.getElementById('traceContent'),
        traceLanguages: document.getElementById('traceLanguages'),
        traceScripts: document.getElementById('traceScripts'),
        traceCodeMixed: document.getElementById('traceCodeMixed'),
        traceTransformer: document.getElementById('traceTransformer'),
        traceTransformerPred: document.getElementById('traceTransformerPred'),
        traceTransformerConf: document.getElementById('traceTransformerConf'),
        traceGateDecision: document.getElementById('traceGateDecision'),
        traceFallback: document.getElementById('traceFallback'),
        traceFallbackTier: document.getElementById('traceFallbackTier'),
        traceEscalation: document.getElementById('traceEscalation'),
        traceEntropy: document.getElementById('traceEntropy'),
        traceCoverage: document.getElementById('traceCoverage'),
        traceDegraded: document.getElementById('traceDegraded'),
        traceProcessingTime: document.getElementById('traceProcessingTime'),
        copyJsonBtn: document.getElementById('copyJsonBtn'),
        
        // Modal
        jsonModal: document.getElementById('jsonModal'),
        jsonOutput: document.getElementById('jsonOutput'),
        closeJsonModal: document.getElementById('closeJsonModal'),
        copyModalJsonBtn: document.getElementById('copyModalJsonBtn'),
        
        // Theme
        themeToggle: document.getElementById('themeToggle'),
        themeIcon: document.getElementById('themeIcon'),
        
        // History
        historyToggle: document.getElementById('historyToggle'),
        historyBadge: document.getElementById('historyBadge'),
        historyPanel: document.getElementById('historyPanel'),
        historyList: document.getElementById('historyList'),
        historyEmpty: document.getElementById('historyEmpty'),
        clearHistoryBtn: document.getElementById('clearHistoryBtn'),
        closeHistoryBtn: document.getElementById('closeHistoryBtn'),
        
        // Shortcuts
        shortcutsPanel: document.getElementById('shortcutsPanel'),
        shortcutsHint: document.getElementById('shortcutsHint'),
        closeShortcuts: document.getElementById('closeShortcuts'),
        
        // Toast
        toastContainer: document.getElementById('toastContainer')
    };

    // ==================== Initialization ====================
    function init() {
        loadTheme();
        loadHistory();
        checkSystemHealth();
        bindEvents();
        updateCharCount();
        updateHistoryBadge();
    }

    function bindEvents() {
        // Input events
        elements.textInput.addEventListener('input', handleTextInput);
        elements.textInput.addEventListener('paste', handlePaste);
        elements.analyzeBtn.addEventListener('click', handleAnalyze);
        
        // Layer toggles
        elements.explanationToggle.addEventListener('click', () => toggleLayer('explanation'));
        elements.traceToggle.addEventListener('click', () => toggleLayer('trace'));
        
        // Safety badge tooltip
        elements.safetyBadge.addEventListener('mouseenter', showTooltip);
        elements.safetyBadge.addEventListener('mouseleave', hideTooltip);
        elements.safetyBadge.addEventListener('focus', showTooltip);
        elements.safetyBadge.addEventListener('blur', hideTooltip);
        
        // Copy buttons
        elements.copyExplanationBtn.addEventListener('click', copyExplanation);
        elements.copyJsonBtn.addEventListener('click', openJsonModal);
        elements.closeJsonModal.addEventListener('click', closeJsonModal);
        elements.copyModalJsonBtn.addEventListener('click', copyRawJson);
        
        // Modal close on backdrop click
        elements.jsonModal.addEventListener('click', (e) => {
            if (e.target === elements.jsonModal) {
                closeJsonModal();
            }
        });
        
        // Theme toggle
        elements.themeToggle.addEventListener('click', toggleTheme);
        
        // History panel
        elements.historyToggle.addEventListener('click', toggleHistoryPanel);
        elements.closeHistoryBtn.addEventListener('click', closeHistoryPanel);
        elements.clearHistoryBtn.addEventListener('click', clearHistory);
        
        // Shortcuts panel
        elements.shortcutsHint.addEventListener('click', toggleShortcutsPanel);
        elements.closeShortcuts.addEventListener('click', closeShortcutsPanel);
        
        // Global keyboard shortcuts
        document.addEventListener('keydown', handleGlobalKeydown);
    }

    // ==================== Keyboard Shortcuts ====================
    function handleGlobalKeydown(e) {
        // Don't trigger shortcuts when typing in input
        const isTyping = document.activeElement === elements.textInput;
        
        // Escape - Clear or close
        if (e.key === 'Escape') {
            if (!elements.jsonModal.classList.contains('hidden')) {
                closeJsonModal();
            } else if (state.shortcutsPanelOpen) {
                closeShortcutsPanel();
            } else if (state.historyPanelOpen) {
                closeHistoryPanel();
            } else if (isTyping && elements.textInput.value) {
                elements.textInput.value = '';
                updateCharCount();
                elements.scriptValue.textContent = '—';
                showToast('Input cleared');
            }
            return;
        }
        
        // Ctrl+Enter - Analyze
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleAnalyze();
            return;
        }
        
        // Only process these shortcuts when not typing
        if (!isTyping) {
            // ? - Show shortcuts
            if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
                e.preventDefault();
                toggleShortcutsPanel();
                return;
            }
            
            // / - Focus input
            if (e.key === '/') {
                e.preventDefault();
                elements.textInput.focus();
                return;
            }
            
            // Ctrl+D - Toggle dark mode
            if (e.key === 'd' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                toggleTheme();
                return;
            }
            
            // Ctrl+H - Toggle history
            if (e.key === 'h' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                toggleHistoryPanel();
                return;
            }
            
            // Ctrl+C with selection or result
            if (e.key === 'c' && (e.ctrlKey || e.metaKey)) {
                const selection = window.getSelection().toString();
                if (!selection && state.lastResponse) {
                    // Copy result summary
                    const summary = `[${state.lastResponse.label.toUpperCase()}] ${state.lastResponse.confidence.toFixed(2)}`;
                    navigator.clipboard.writeText(summary);
                    showToast('Result copied');
                }
            }
        }
    }

    // ==================== Theme Management ====================
    function loadTheme() {
        const savedTheme = localStorage.getItem('must-theme') || 'system';
        state.theme = savedTheme;
        applyTheme();
    }

    function applyTheme() {
        const root = document.documentElement;
        
        if (state.theme === 'dark') {
            root.setAttribute('data-theme', 'dark');
            elements.themeIcon.textContent = '☀️';
        } else if (state.theme === 'light') {
            root.setAttribute('data-theme', 'light');
            elements.themeIcon.textContent = '🌙';
        } else {
            root.removeAttribute('data-theme');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            elements.themeIcon.textContent = prefersDark ? '☀️' : '🌙';
        }
    }

    function toggleTheme() {
        // Cycle: system -> light -> dark -> system
        if (state.theme === 'system') {
            state.theme = 'light';
        } else if (state.theme === 'light') {
            state.theme = 'dark';
        } else {
            state.theme = 'system';
        }
        
        localStorage.setItem('must-theme', state.theme);
        applyTheme();
        showToast(`Theme: ${state.theme}`);
    }

    // ==================== History Management ====================
    function loadHistory() {
        try {
            const saved = localStorage.getItem('must-history');
            if (saved) {
                state.history = JSON.parse(saved);
                renderHistory();
            }
        } catch (e) {
            console.error('Failed to load history:', e);
            state.history = [];
        }
    }

    function saveHistory() {
        try {
            localStorage.setItem('must-history', JSON.stringify(state.history));
        } catch (e) {
            console.error('Failed to save history:', e);
        }
    }

    function addToHistory(text, result) {
        const entry = {
            id: Date.now(),
            text: text.substring(0, 100),
            fullText: text,
            label: result.label,
            confidence: result.confidence,
            timestamp: new Date().toISOString()
        };
        
        state.history.unshift(entry);
        
        // Keep only last N items
        if (state.history.length > CONFIG.HISTORY_MAX_ITEMS) {
            state.history = state.history.slice(0, CONFIG.HISTORY_MAX_ITEMS);
        }
        
        saveHistory();
        renderHistory();
        updateHistoryBadge();
    }

    function renderHistory() {
        if (state.history.length === 0) {
            elements.historyEmpty.classList.remove('hidden');
            elements.historyList.querySelectorAll('.history-item').forEach(el => el.remove());
            return;
        }
        
        elements.historyEmpty.classList.add('hidden');
        
        // Clear existing items
        elements.historyList.querySelectorAll('.history-item').forEach(el => el.remove());
        
        // Render items
        state.history.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.dataset.id = entry.id;
            
            const timeAgo = getTimeAgo(new Date(entry.timestamp));
            
            item.innerHTML = `
                <div class="history-item-text">${escapeHtml(entry.text)}</div>
                <div class="history-item-meta">
                    <span class="history-item-label ${entry.label}">${entry.label}</span>
                    <span class="history-item-time">${timeAgo}</span>
                </div>
            `;
            
            item.addEventListener('click', () => loadFromHistory(entry));
            elements.historyList.appendChild(item);
        });
    }

    function loadFromHistory(entry) {
        elements.textInput.value = entry.fullText;
        updateCharCount();
        detectScriptRealtime();
        closeHistoryPanel();
        showToast('Loaded from history');
    }

    function clearHistory() {
        if (confirm('Clear all history? This cannot be undone.')) {
            state.history = [];
            saveHistory();
            renderHistory();
            updateHistoryBadge();
            showToast('History cleared');
        }
    }

    function updateHistoryBadge() {
        const count = state.history.length;
        if (count > 0) {
            elements.historyBadge.textContent = count > 99 ? '99+' : count;
            elements.historyBadge.classList.remove('hidden');
        } else {
            elements.historyBadge.classList.add('hidden');
        }
    }

    function toggleHistoryPanel() {
        state.historyPanelOpen = !state.historyPanelOpen;
        elements.historyPanel.classList.toggle('open', state.historyPanelOpen);
    }

    function closeHistoryPanel() {
        state.historyPanelOpen = false;
        elements.historyPanel.classList.remove('open');
    }

    // ==================== Shortcuts Panel ====================
    function toggleShortcutsPanel() {
        state.shortcutsPanelOpen = !state.shortcutsPanelOpen;
        elements.shortcutsPanel.classList.toggle('hidden', !state.shortcutsPanelOpen);
        elements.shortcutsHint.classList.toggle('hidden', state.shortcutsPanelOpen);
    }

    function closeShortcutsPanel() {
        state.shortcutsPanelOpen = false;
        elements.shortcutsPanel.classList.add('hidden');
        elements.shortcutsHint.classList.remove('hidden');
    }

    // ==================== Toast Notifications ====================
    function showToast(message, type = 'default') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        elements.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fadeout');
            setTimeout(() => toast.remove(), 200);
        }, CONFIG.TOAST_DURATION_MS);
    }

    // ==================== System Health ====================
    async function checkSystemHealth() {
        try {
            const response = await fetch(`${CONFIG.API_BASE}/health`);
            const data = await response.json();
            
            state.systemHealthy = data.status === 'healthy';
            state.degradedMode = data.degraded_mode;
            
            updateSystemStatus(data);
        } catch (error) {
            state.systemHealthy = false;
            updateSystemStatus({ status: 'error', degraded_mode: true });
        }
    }

    function updateSystemStatus(data) {
        const indicator = elements.statusIndicator;
        const text = elements.statusText;
        
        if (data.status === 'healthy' && !data.degraded_mode) {
            indicator.className = 'status-indicator healthy';
            text.textContent = 'System ready';
            elements.systemBanner.classList.add('hidden');
        } else if (data.status === 'healthy' && data.degraded_mode) {
            indicator.className = 'status-indicator degraded';
            text.textContent = 'Safety-only mode';
            elements.bannerText.textContent = 'Running in safety-only mode. Transformer unavailable.';
            elements.systemBanner.classList.remove('hidden');
        } else {
            indicator.className = 'status-indicator error';
            text.textContent = 'System error';
            elements.systemBanner.classList.add('hidden');
        }
    }

    // ==================== Input Handling ====================
    function handleTextInput() {
        updateCharCount();
        debouncedScriptDetect();
    }

    function handlePaste(e) {
        // No auto-submit on paste - just update counters
        setTimeout(() => {
            updateCharCount();
            detectScriptRealtime();
        }, 0);
    }

    function updateCharCount() {
        const count = elements.textInput.value.length;
        elements.charCountValue.textContent = count.toLocaleString();
    }

    function debouncedScriptDetect() {
        if (state.scriptDetectTimeout) {
            clearTimeout(state.scriptDetectTimeout);
        }
        state.scriptDetectTimeout = setTimeout(detectScriptRealtime, CONFIG.SCRIPT_DETECT_DEBOUNCE_MS);
    }

    async function detectScriptRealtime() {
        const text = elements.textInput.value.trim();
        
        if (!text) {
            elements.scriptValue.textContent = '—';
            elements.scriptValue.classList.remove('mixed');
            return;
        }
        
        try {
            const response = await fetch(`${CONFIG.API_BASE}/detect-script`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            const data = await response.json();
            
            if (data.is_mixed) {
                const scripts = Object.keys(data.scripts).join(' + ');
                elements.scriptValue.textContent = scripts || 'Mixed';
                elements.scriptValue.classList.add('mixed');
            } else if (data.primary_script) {
                elements.scriptValue.textContent = formatScriptName(data.primary_script);
                elements.scriptValue.classList.remove('mixed');
            } else {
                elements.scriptValue.textContent = '—';
                elements.scriptValue.classList.remove('mixed');
            }
        } catch (error) {
            elements.scriptValue.textContent = '—';
            elements.scriptValue.classList.remove('mixed');
        }
    }

    function formatScriptName(script) {
        const names = {
            'latin': 'Latin',
            'devanagari': 'Devanagari',
            'tamil': 'Tamil',
            'unknown': 'Unknown'
        };
        return names[script.toLowerCase()] || script;
    }

    // ==================== Analysis ====================
    async function handleAnalyze() {
        const text = elements.textInput.value.trim();
        
        if (!text) {
            showToast('Please enter text to analyze', 'error');
            elements.textInput.focus();
            return;
        }
        
        if (state.isProcessing) {
            return;
        }
        
        state.isProcessing = true;
        elements.analyzeBtn.disabled = true;
        elements.analyzeBtn.classList.add('analyzing');
        
        const startTime = performance.now();
        
        // Show processing indicator after threshold
        const processingTimeout = setTimeout(() => {
            elements.processingIndicator.classList.remove('hidden');
        }, CONFIG.PROCESSING_DELAY_THRESHOLD_MS);
        
        try {
            const languageHint = elements.languageHint.value || null;
            
            const response = await fetch(`${CONFIG.API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, language_hint: languageHint })
            });
            
            const data = await response.json();
            
            clearTimeout(processingTimeout);
            elements.processingIndicator.classList.add('hidden');
            
            if (response.ok) {
                state.lastResponse = data;
                renderResults(data, text);
                addToHistory(text, data);
                showToast(`Analyzed: ${data.label.toUpperCase()}`, data.label === 'hate' ? 'error' : 'success');
            } else {
                // Handle error with safe default
                if (data.safe_default) {
                    state.lastResponse = data;
                    renderSafeDefault(data, text);
                } else {
                    showToast('Analysis failed. Please try again.', 'error');
                }
            }
        } catch (error) {
            clearTimeout(processingTimeout);
            elements.processingIndicator.classList.add('hidden');
            showToast('Connection error. Please check your network.', 'error');
        } finally {
            state.isProcessing = false;
            elements.analyzeBtn.disabled = false;
            elements.analyzeBtn.classList.remove('analyzing');
        }
    }

    // ==================== Rendering ====================
    function renderResults(data, originalText) {
        // Show result layers
        elements.decisionLayer.classList.remove('hidden');
        elements.explanationLayer.classList.remove('hidden');
        elements.traceLayer.classList.remove('hidden');
        
        // Render Decision Layer
        renderDecisionLayer(data);
        
        // Render Explanation Layer
        renderExplanationLayer(data, originalText);
        
        // Render System Trace Layer
        renderTraceLayer(data);
    }

    function renderDecisionLayer(data) {
        const label = data.label;
        const confidence = data.confidence;
        const safetyBadge = data.safety_badge;
        
        // Label
        elements.labelDisplay.className = `label-display ${label}`;
        elements.labelText.textContent = label.toUpperCase();
        
        // Confidence bar
        elements.confidenceBar.className = `confidence-bar ${label}`;
        elements.confidenceBar.style.width = `${confidence * 100}%`;
        elements.confidenceValue.textContent = confidence.toFixed(2);
        
        // Safety badge
        elements.safetyBadge.className = `safety-badge ${safetyBadge.type}`;
        elements.badgeLabel.textContent = safetyBadge.label;
        elements.tooltipText.textContent = safetyBadge.tooltip;
    }

    function renderExplanationLayer(data, originalText) {
        const explanation = data.explanation;
        
        // Highlighted text
        const highlightedHtml = highlightHarmTokens(originalText, explanation.key_harm_tokens);
        elements.highlightedText.innerHTML = highlightedHtml;
        
        // Label justification
        elements.labelJustification.textContent = explanation.label_justification;
        
        // Rejected labels
        if (explanation.weaker_labels_rejected && explanation.weaker_labels_rejected.length > 0) {
            elements.rejectedLabelsSection.classList.remove('hidden');
            elements.rejectedLabelsList.innerHTML = '';
            
            explanation.weaker_labels_rejected.forEach(item => {
                const li = document.createElement('li');
                li.className = 'rejected-label-item';
                li.innerHTML = `
                    <span class="rejected-label-name">${item.label}</span>
                    <span class="rejected-label-reason">${item.reason}</span>
                `;
                elements.rejectedLabelsList.appendChild(li);
            });
        } else {
            elements.rejectedLabelsSection.classList.add('hidden');
        }
        
        // Harm tokens
        if (explanation.key_harm_tokens && explanation.key_harm_tokens.length > 0) {
            elements.harmTokensSection.classList.remove('hidden');
            elements.harmTokensContainer.innerHTML = '';
            
            explanation.key_harm_tokens.forEach(token => {
                const chip = document.createElement('span');
                chip.className = 'token-chip';
                chip.textContent = token;
                elements.harmTokensContainer.appendChild(chip);
            });
        } else {
            elements.harmTokensSection.classList.add('hidden');
        }
        
        // Identity groups
        if (explanation.identity_groups && explanation.identity_groups.length > 0) {
            elements.identityGroupsSection.classList.remove('hidden');
            elements.identityGroupsContainer.innerHTML = '';
            
            explanation.identity_groups.forEach(group => {
                const chip = document.createElement('span');
                chip.className = 'group-chip';
                chip.textContent = group;
                elements.identityGroupsContainer.appendChild(chip);
            });
        } else {
            elements.identityGroupsSection.classList.add('hidden');
        }
    }

    function highlightHarmTokens(text, tokens) {
        if (!tokens || tokens.length === 0) {
            return escapeHtml(text);
        }
        
        // Escape HTML first
        let result = escapeHtml(text);
        
        // Sort tokens by length (longest first) to avoid partial matches
        const sortedTokens = [...tokens].sort((a, b) => b.length - a.length);
        
        // Highlight each token
        sortedTokens.forEach(token => {
            const escapedToken = escapeHtml(token);
            // Case-insensitive replacement that preserves original case
            const regex = new RegExp(`(${escapeRegex(escapedToken)})`, 'gi');
            result = result.replace(regex, '<span class="harm-highlight">$1</span>');
        });
        
        return result;
    }

    function renderTraceLayer(data) {
        const trace = data.system_trace;
        const metadata = data.metadata;
        
        // Languages
        elements.traceLanguages.textContent = formatLanguages(trace.languages_detected);
        
        // Scripts
        elements.traceScripts.textContent = formatScripts(trace.script_distribution);
        
        // Code mixed
        elements.traceCodeMixed.textContent = trace.is_code_mixed ? 'Yes' : 'No';
        elements.traceCodeMixed.className = `trace-value ${trace.is_code_mixed ? 'warning' : ''}`;
        
        // Transformer
        elements.traceTransformer.textContent = trace.transformer_used ? 'Yes' : 'No (bypassed)';
        elements.traceTransformer.className = `trace-value ${trace.transformer_used ? 'true' : 'warning'}`;
        
        // Transformer prediction
        elements.traceTransformerPred.textContent = trace.transformer_prediction || '—';
        
        // Transformer confidence
        elements.traceTransformerConf.textContent = trace.transformer_confidence 
            ? trace.transformer_confidence.toFixed(4) 
            : '—';
        
        // Gate decision
        elements.traceGateDecision.textContent = formatGateDecision(trace.confidence_gate_decision);
        
        // Fallback
        elements.traceFallback.textContent = trace.fallback_used ? 'Yes' : 'No';
        elements.traceFallback.className = `trace-value ${trace.fallback_used ? 'warning' : ''}`;
        
        // Fallback tier
        elements.traceFallbackTier.textContent = trace.fallback_tier 
            ? `Tier ${trace.fallback_tier}` 
            : '—';
        
        // Escalation
        elements.traceEscalation.textContent = trace.escalation_triggered ? 'Yes' : 'No';
        elements.traceEscalation.className = `trace-value ${trace.escalation_triggered ? 'error' : ''}`;
        
        // Entropy
        elements.traceEntropy.textContent = trace.entropy.toFixed(4);
        
        // Coverage
        elements.traceCoverage.textContent = `${(trace.tokenization_coverage * 100).toFixed(1)}%`;
        
        // Degraded mode
        elements.traceDegraded.textContent = trace.degraded_mode ? 'Yes' : 'No';
        elements.traceDegraded.className = `trace-value ${trace.degraded_mode ? 'warning' : ''}`;
        
        // Processing time
        elements.traceProcessingTime.textContent = `${metadata.processing_time_ms.toFixed(0)}ms`;
    }

    function renderSafeDefault(data, originalText) {
        // Show decision layer with safe default
        elements.decisionLayer.classList.remove('hidden');
        elements.explanationLayer.classList.remove('hidden');
        elements.traceLayer.classList.add('hidden');
        
        const safeDefault = data.safe_default;
        
        // Render with safe default values
        elements.labelDisplay.className = 'label-display neutral';
        elements.labelText.textContent = safeDefault.label.toUpperCase();
        elements.confidenceBar.className = 'confidence-bar neutral';
        elements.confidenceBar.style.width = `${safeDefault.confidence * 100}%`;
        elements.confidenceValue.textContent = safeDefault.confidence.toFixed(2);
        
        // Show error badge
        elements.safetyBadge.className = 'safety-badge rule_escalation';
        elements.badgeLabel.textContent = 'System Error';
        elements.tooltipText.textContent = 'Analysis failed. Returning safe default.';
        
        // Show error in explanation
        elements.highlightedText.innerHTML = escapeHtml(originalText);
        elements.labelJustification.textContent = safeDefault.explanation;
        elements.rejectedLabelsSection.classList.add('hidden');
        elements.harmTokensSection.classList.add('hidden');
        elements.identityGroupsSection.classList.add('hidden');
    }

    // ==================== Layer Toggles ====================
    function toggleLayer(layer) {
        if (layer === 'explanation') {
            const isExpanded = elements.explanationToggle.getAttribute('aria-expanded') === 'true';
            elements.explanationToggle.setAttribute('aria-expanded', !isExpanded);
            elements.explanationContent.classList.toggle('collapsed', isExpanded);
            elements.explanationContent.classList.toggle('expanded', !isExpanded);
        } else if (layer === 'trace') {
            const isExpanded = elements.traceToggle.getAttribute('aria-expanded') === 'true';
            elements.traceToggle.setAttribute('aria-expanded', !isExpanded);
            elements.traceContent.classList.toggle('collapsed', isExpanded);
            elements.traceContent.classList.toggle('expanded', !isExpanded);
        }
    }

    // ==================== Tooltip ====================
    function showTooltip() {
        elements.safetyTooltip.classList.remove('hidden');
    }

    function hideTooltip() {
        elements.safetyTooltip.classList.add('hidden');
    }

    // ==================== Copy Functions ====================
    function copyExplanation() {
        if (!state.lastResponse) return;
        
        const explanation = state.lastResponse.explanation;
        const text = [
            `Label: ${state.lastResponse.label}`,
            `Confidence: ${state.lastResponse.confidence}`,
            ``,
            `Justification: ${explanation.label_justification}`,
            ``,
            `Key Harm Tokens: ${explanation.key_harm_tokens.join(', ') || 'None'}`,
            `Identity Groups: ${explanation.identity_groups.join(', ') || 'None'}`
        ].join('\n');
        
        copyToClipboard(text, elements.copyExplanationBtn);
        showToast('Explanation copied');
    }

    function openJsonModal() {
        if (!state.lastResponse) return;
        
        elements.jsonOutput.textContent = JSON.stringify(state.lastResponse, null, 2);
        elements.jsonModal.classList.remove('hidden');
        elements.closeJsonModal.focus();
    }

    function closeJsonModal() {
        elements.jsonModal.classList.add('hidden');
        elements.copyJsonBtn.focus();
    }

    function copyRawJson() {
        if (!state.lastResponse) return;
        
        const json = JSON.stringify(state.lastResponse, null, 2);
        copyToClipboard(json, elements.copyModalJsonBtn);
        showToast('JSON copied');
    }

    async function copyToClipboard(text, button) {
        try {
            await navigator.clipboard.writeText(text);
            
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.classList.add('copied');
            
            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('copied');
            }, 2000);
        } catch (error) {
            console.error('Failed to copy:', error);
            showToast('Failed to copy', 'error');
        }
    }

    // ==================== Banner ====================
    window.dismissBanner = function() {
        elements.systemBanner.classList.add('hidden');
    };

    // ==================== Utility Functions ====================
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function formatLanguages(languages) {
        if (!languages || Object.keys(languages).length === 0) {
            return '—';
        }
        
        return Object.entries(languages)
            .filter(([_, proportion]) => proportion > 0.01)
            .map(([lang, proportion]) => `${lang}: ${(proportion * 100).toFixed(0)}%`)
            .join(', ');
    }

    function formatScripts(scripts) {
        if (!scripts || Object.keys(scripts).length === 0) {
            return '—';
        }
        
        return Object.entries(scripts)
            .filter(([_, proportion]) => proportion > 0.01)
            .map(([script, proportion]) => `${script}: ${(proportion * 100).toFixed(0)}%`)
            .join(', ');
    }

    function formatGateDecision(decision) {
        const labels = {
            'accepted': 'Accepted',
            'bypassed_degraded_mode': 'Bypassed (degraded mode)',
            'uncertain_high_entropy': 'Uncertain (high entropy)',
            'low_confidence': 'Low confidence',
            'fallback_override': 'Fallback override'
        };
        return labels[decision] || decision;
    }

    function getTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);
        
        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
        
        return date.toLocaleDateString();
    }

    // ==================== Initialize ====================
    document.addEventListener('DOMContentLoaded', init);
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (state.theme === 'system') {
            applyTheme();
        }
    });
})();
