/**
 * Decomposition View
 * ==================
 *
 * Persistent right sidebar displaying real-time motor unit decomposition data:
 *  - Classification label badge (REST / MOVE 1 / MOVE 2)
 *  - Per-MU source signal waveform (continuous trace)
 *  - Per-MU spike raster (vertical ticks)
 *  - Per-MU firing rate bar + SIL score label
 *
 * Follows the same canvas + RAF pattern as EMGChannelView.js.
 * Width: 320px, fixed to the right edge of the viewport.
 */
export default class DecompositionView {
	constructor(emgClient, options = {}) {
		this.emgClient = emgClient;
		this.isVisible = false;

		// Buffer config — ~2s at 2kHz
		this.bufferSize = options.bufferSize || 4096;
		this.bufferSizeFocused = options.bufferSizeFocused || 16384; // ~8s at 2kHz
		this.silThreshold = options.silThreshold ?? 0;

		// Per-MU state
		this.nMUs = 0;
		this.sourceBuffers = [];   // Float32Array per MU
		this.spikeHistory = [];    // array of buffer-relative indices per MU
		this.firingRates = [];     // latest Hz per MU
		this.silScores = [];       // latest SIL per MU
		this.classificationLabel = null; // null | 0 | 1 | 2

		// Classification focus mode: show only the 2 classified MUs
		this.focusMode = false;
		this.classificationMU1 = null; // MU index for label 1
		this.classificationMU2 = null; // MU index for label 2
		this.classificationThreshold = 10.0; // Hz

		// Layout constants
		this.width = 320;
		this.muRowHeight = 80;  // waveform (48) + raster (20) + label (12)
		this.muRowHeightFocused = 160; // larger rows in focus mode
		this.waveformHeight = 40;
		this.waveformHeightFocused = 90;
		this.rasterHeight = 18;
		this.rasterHeightFocused = 30;
		this.labelHeight = 16;
		this.badgeHeight = 40;
		this.padding = 8;
		this.frBarWidth = 40; // width for FR bars area

		// MU colors (one per MU, cycling)
		this.muColors = [
			'#00ff88', '#ff8800', '#00aaff', '#ff44aa',
			'#aaff00', '#aa44ff', '#ff4444', '#44ffff',
		];

		// Colors (dark theme matching EMGChannelView)
		this.colors = {
			background: '#1a1a2e',
			panel: '#0f0f1e',
			border: '#3a3a5e',
			grid: '#2a2a4e',
			text: '#ffffff',
			textDim: '#888899',
			badgeRest: '#555566',
			badgeMove1: '#00cc66',
			badgeMove2: '#ff8800',
		};

		// Animation
		this.animationId = null;

		this._createElements();
		this._subscribeToClient();
	}

	// -------------------------------------------------------------------------
	// DOM setup
	// -------------------------------------------------------------------------

	_createElements() {
		// Sidebar container
		this.sidebar = document.createElement('div');
		this.sidebar.className = 'decomp-sidebar';
		this.sidebar.style.cssText = `
			position: fixed;
			right: 0;
			top: 0;
			width: ${this.width}px;
			height: 100vh;
			background: ${this.colors.background};
			border-left: 1px solid ${this.colors.border};
			display: none;
			flex-direction: column;
			z-index: 500;
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			overflow: hidden;
		`;

		// Header
		const header = document.createElement('div');
		header.style.cssText = `
			display: flex;
			justify-content: space-between;
			align-items: center;
			padding: 10px 12px 8px;
			border-bottom: 1px solid ${this.colors.border};
			flex-shrink: 0;
		`;

		const title = document.createElement('span');
		title.textContent = 'DECOMPOSITION';
		title.style.cssText = `
			color: ${this.colors.text};
			font-size: 13px;
			font-weight: 600;
			letter-spacing: 0.05em;
		`;

		const closeBtn = document.createElement('button');
		closeBtn.textContent = '\u00d7';
		closeBtn.style.cssText = `
			background: none;
			border: none;
			color: ${this.colors.textDim};
			font-size: 22px;
			cursor: pointer;
			padding: 0 4px;
			line-height: 1;
		`;
		closeBtn.onclick = () => this.hide();

		header.appendChild(title);
		header.appendChild(closeBtn);

		// Focus mode toggle button
		this.focusToggle = document.createElement('button');
		this.focusToggle.textContent = 'FOCUS';
		this.focusToggle.style.cssText = `
			margin: 6px 12px 2px;
			padding: 5px 12px;
			border-radius: 4px;
			background: ${this.colors.border};
			border: 1px solid ${this.colors.border};
			color: ${this.colors.textDim};
			font-size: 11px;
			font-weight: 600;
			letter-spacing: 0.06em;
			cursor: pointer;
			flex-shrink: 0;
			transition: background 0.15s, color 0.15s;
		`;
		this.focusToggle.onclick = () => this._toggleFocusMode();

		// Classification badge
		this.badge = document.createElement('div');
		this.badge.style.cssText = `
			margin: 8px 12px;
			padding: 8px 12px;
			border-radius: 8px;
			background: ${this.colors.badgeRest};
			text-align: center;
			font-size: 14px;
			font-weight: 700;
			letter-spacing: 0.08em;
			color: white;
			flex-shrink: 0;
			transition: background 0.15s;
		`;
		this.badge.textContent = 'NO MODEL';

		// FR bars container (drawn on its own canvas, to the right)
		this.frBarContainer = document.createElement('div');
		this.frBarContainer.style.cssText = `
			display: none;
			flex-shrink: 0;
			padding: 8px 12px;
			border-bottom: 1px solid ${this.colors.border};
			background: ${this.colors.panel};
		`;
		this.frBarCanvas = document.createElement('canvas');
		this.frBarCanvas.style.cssText = 'display: block; width: 100%;';
		this.frBarCtx = this.frBarCanvas.getContext('2d');
		this.frBarContainer.appendChild(this.frBarCanvas);

		// Canvas container (scrollable)
		this.canvasContainer = document.createElement('div');
		this.canvasContainer.style.cssText = `
			flex: 1;
			overflow-y: auto;
			overflow-x: hidden;
			background: ${this.colors.panel};
		`;

		// Canvas
		this.canvas = document.createElement('canvas');
		this.canvas.style.cssText = 'display: block; width: 100%;';
		this.ctx = this.canvas.getContext('2d');

		this.canvasContainer.appendChild(this.canvas);

		// Assemble
		this.sidebar.appendChild(header);
		this.sidebar.appendChild(this.focusToggle);
		this.sidebar.appendChild(this.badge);
		this.sidebar.appendChild(this.frBarContainer);
		this.sidebar.appendChild(this.canvasContainer);

		document.body.appendChild(this.sidebar);

		// Resize canvas when container size changes
		this._resizeObserver = new ResizeObserver(() => this._resizeCanvas());
		this._resizeObserver.observe(this.canvasContainer);
	}

	_resizeCanvas() {
		const musToShow = this._getMUsToShow();
		const rowH = this.focusMode ? this.muRowHeightFocused : this.muRowHeight;
		const h = musToShow.length > 0
			? musToShow.length * rowH + this.padding * 2
			: 100;
		const w = this.canvasContainer.clientWidth || this.width;
		this.canvas.width = w;
		this.canvas.height = h;
		this.canvas.style.height = h + 'px';

		// FR bar canvas
		this._resizeFrBarCanvas();
	}

	_resizeFrBarCanvas() {
		const w = this.canvasContainer.clientWidth || this.width;
		const barH = 120;
		this.frBarCanvas.width = w;
		this.frBarCanvas.height = barH;
		this.frBarCanvas.style.height = barH + 'px';
	}

	// -------------------------------------------------------------------------
	// EMG client subscriptions
	// -------------------------------------------------------------------------

	_subscribeToClient() {
		this.emgClient.onDecompositionStatus((status) => {
			if (status.active) {
				this._initBuffers(status.nMUs);
				this._updateBadge(null); // reset to "REST" state while inactive
			} else {
				this.nMUs = 0;
				this.sourceBuffers = [];
				this.spikeHistory = [];
				this.firingRates = [];
				this.silScores = [];
				this.classificationLabel = null;
				this._updateBadge(null);
				this._resizeCanvas();
			}
		});

		this.emgClient.onDecompositionData((data) => {
			if (this.nMUs === 0) return;
			this.firingRates = data.firingRates || [];
			this.silScores = data.silScores || [];
			this._updateBuffers(data);
		});

		this.emgClient.onClassification((result) => {
			this.classificationLabel = result.label;
			this._updateBadge(result.label);
		});

		this.emgClient.onClassificationStatus((status) => {
			if (status.active) {
				this.classificationMU1 = status.mu1Idx;
				this.classificationMU2 = status.mu2Idx;
				this.classificationThreshold = status.threshold ?? 10.0;
				this.frBarContainer.style.display = 'block';
				this._resizeFrBarCanvas();
				this._resizeCanvas();
			} else {
				this.classificationMU1 = null;
				this.classificationMU2 = null;
				this.frBarContainer.style.display = 'none';
				if (this.focusMode) this._toggleFocusMode(); // exit focus mode
			}
		});
	}

	_initBuffers(nMUs) {
		this.nMUs = nMUs;
		this.sourceBuffers = [];
		this.spikeHistory = [];
		this.firingRates = new Array(nMUs).fill(0);
		this.silScores = new Array(nMUs).fill(0);
		this.classificationLabel = null;

		// Allocate with larger buffer for focus mode
		const maxBuf = Math.max(this.bufferSize, this.bufferSizeFocused);
		for (let i = 0; i < nMUs; i++) {
			this.sourceBuffers.push(new Float32Array(maxBuf));
			this.spikeHistory.push([]);
		}

		this._updateBadge(null);
		this._resizeCanvas();
	}

	_updateBuffers(data) {
		const nNew = data.nSamples;
		const maxBuf = Math.max(this.bufferSize, this.bufferSizeFocused);

		for (let mu = 0; mu < this.nMUs && mu < data.sources.length; mu++) {
			const buf = this.sourceBuffers[mu];

			// Shift left and append (same pattern as EMGClient._handleEMGChunk)
			if (nNew < maxBuf) {
				buf.copyWithin(0, nNew);
				const src = data.sources[mu];
				for (let i = 0; i < nNew; i++) {
					buf[maxBuf - nNew + i] = src[i];
				}
			} else {
				const src = data.sources[mu];
				for (let i = 0; i < maxBuf; i++) {
					buf[i] = src[nNew - maxBuf + i];
				}
			}

			// Shift spike history indices
			this.spikeHistory[mu] = this.spikeHistory[mu]
				.map(t => t - nNew)
				.filter(t => t >= 0);

			// Add new spikes (only if SIL is above threshold)
			if ((this.silScores[mu] ?? 0) >= this.silThreshold) {
				const newSpikes = data.spikes[String(mu)] || [];
				for (const idx of newSpikes) {
					this.spikeHistory[mu].push(maxBuf - nNew + idx);
				}
			}
		}
	}

	// -------------------------------------------------------------------------
	// Badge
	// -------------------------------------------------------------------------

	_updateBadge(label) {
		if (this.nMUs === 0) {
			this.badge.textContent = 'NO MODEL';
			this.badge.style.background = this.colors.badgeRest;
			return;
		}
		if (label === null || label === undefined) {
			this.badge.textContent = 'REST';
			this.badge.style.background = this.colors.badgeRest;
			return;
		}
		switch (label) {
			case 0:
				this.badge.textContent = 'REST';
				this.badge.style.background = this.colors.badgeRest;
				break;
			case 1:
				this.badge.textContent = 'MOVE 1';
				this.badge.style.background = this.colors.badgeMove1;
				break;
			case 2:
				this.badge.textContent = 'MOVE 2';
				this.badge.style.background = this.colors.badgeMove2;
				break;
			default:
				this.badge.textContent = `LABEL ${label}`;
				this.badge.style.background = this.colors.badgeRest;
		}
	}

	// -------------------------------------------------------------------------
	// Show / hide / toggle
	// -------------------------------------------------------------------------

	show() {
		this.isVisible = true;
		this.sidebar.style.display = 'flex';
		this._startAnimation();
	}

	hide() {
		this.isVisible = false;
		this.sidebar.style.display = 'none';
		this._stopAnimation();
	}

	toggle() {
		if (this.isVisible) {
			this.hide();
		} else {
			this.show();
		}
	}

	// -------------------------------------------------------------------------
	// RAF draw loop
	// -------------------------------------------------------------------------

	_startAnimation() {
		const draw = () => {
			if (!this.isVisible) return;
			this._draw();
			this.animationId = requestAnimationFrame(draw);
		};
		draw();
	}

	_stopAnimation() {
		if (this.animationId) {
			cancelAnimationFrame(this.animationId);
			this.animationId = null;
		}
	}

	// -------------------------------------------------------------------------
	// Canvas rendering
	// -------------------------------------------------------------------------

	_draw() {
		const ctx = this.ctx;
		const W = this.canvas.width;
		const H = this.canvas.height;

		if (!W || !H) return;

		// Background
		ctx.fillStyle = this.colors.panel;
		ctx.fillRect(0, 0, W, H);

		if (this.nMUs === 0) {
			ctx.fillStyle = this.colors.textDim;
			ctx.font = '13px monospace';
			ctx.textAlign = 'center';
			ctx.fillText('Waiting for model...', W / 2, 50);
			ctx.textAlign = 'left';
			return;
		}

		const musToShow = this._getMUsToShow();
		const rowH = this.focusMode ? this.muRowHeightFocused : this.muRowHeight;
		let y = this.padding;

		for (const mu of musToShow) {
			this._drawMU(ctx, mu, W, y);
			y += rowH;
		}

		// Draw FR bars
		if (this.classificationMU1 !== null && this.classificationMU2 !== null) {
			this._drawFRBars();
		}
	}

	_drawMU(ctx, mu, W, yBase) {
		const color = this.muColors[mu % this.muColors.length];
		const fr = (this.firingRates[mu] ?? 0).toFixed(1);
		const sil = (this.silScores[mu] ?? 0).toFixed(2);
		const reliable = (this.silScores[mu] ?? 0) >= this.silThreshold;
		const px = this.padding;
		const innerW = W - px * 2;

		const rowH = this.focusMode ? this.muRowHeightFocused : this.muRowHeight;
		const wfH = this.focusMode ? this.waveformHeightFocused : this.waveformHeight;
		const rsH = this.focusMode ? this.rasterHeightFocused : this.rasterHeight;
		const activeBuf = this.focusMode ? this.bufferSizeFocused : this.bufferSize;

		// Highlight if this MU is one of the classification MUs
		const isClassMU = mu === this.classificationMU1 || mu === this.classificationMU2;
		if (isClassMU && !this.focusMode) {
			ctx.fillStyle = 'rgba(255,255,255,0.03)';
			ctx.fillRect(0, yBase, W, rowH);
		}

		// --- Label row ---
		const labelY = yBase + 12;
		ctx.fillStyle = reliable ? color : this.colors.textDim;
		ctx.font = `bold 11px monospace`;
		ctx.fillText(`MU ${mu}`, px, labelY);

		// Firing rate + SIL on the right
		const stats = `${fr} Hz  SIL ${sil}${reliable ? '' : ' (low)'}`;
		ctx.font = '10px monospace';
		ctx.fillStyle = reliable ? this.colors.text : this.colors.textDim;
		ctx.textAlign = 'right';
		ctx.fillText(stats, W - px, labelY);
		ctx.textAlign = 'left';

		// Firing rate bar (thin, below label)
		const barY = labelY + 3;
		const barH = this.focusMode ? 5 : 3;
		const maxFR = 40; // Hz scale max
		ctx.fillStyle = this.colors.grid;
		ctx.fillRect(px, barY, innerW, barH);
		const barW = Math.min(1, (this.firingRates[mu] ?? 0) / maxFR) * innerW;
		ctx.fillStyle = reliable ? color : '#555566';
		ctx.fillRect(px, barY, barW, barH);

		// --- Waveform ---
		const waveY = yBase + 22 + (this.focusMode ? 4 : 0);
		this._drawWaveform(ctx, mu, px, waveY, innerW, wfH, color, reliable, activeBuf);

		// --- Spike raster ---
		const rasterY = waveY + wfH + 2;
		this._drawRaster(ctx, mu, px, rasterY, innerW, rsH, color, reliable, activeBuf);

		// Divider line
		ctx.strokeStyle = this.colors.border;
		ctx.lineWidth = 0.5;
		ctx.beginPath();
		ctx.moveTo(0, yBase + rowH - 1);
		ctx.lineTo(W, yBase + rowH - 1);
		ctx.stroke();
	}

	_drawWaveform(ctx, mu, x, y, w, h, color, reliable, activeBuf) {
		const buf = this.sourceBuffers[mu];
		if (!buf) return;

		const maxBuf = Math.max(this.bufferSize, this.bufferSizeFocused);
		// Use only the last activeBuf samples from the full buffer
		const startIdx = maxBuf - activeBuf;

		// Background
		ctx.fillStyle = this.colors.background;
		ctx.fillRect(x, y, w, h);

		// Center line
		ctx.strokeStyle = this.colors.grid;
		ctx.lineWidth = 0.5;
		ctx.beginPath();
		ctx.moveTo(x, y + h / 2);
		ctx.lineTo(x + w, y + h / 2);
		ctx.stroke();

		// Find range for normalization (use active portion of buffer)
		let min = Infinity, max = -Infinity;
		for (let i = startIdx; i < maxBuf; i++) {
			if (buf[i] < min) min = buf[i];
			if (buf[i] > max) max = buf[i];
		}
		const range = max - min || 1;
		const midY = y + h / 2;
		const amplitude = (h / 2) * 0.85;

		ctx.strokeStyle = reliable ? color : this.colors.textDim;
		ctx.lineWidth = 1.2;
		ctx.beginPath();

		for (let px = 0; px < w; px++) {
			const sampleIdx = startIdx + Math.floor((px / w) * activeBuf);
			const sample = buf[sampleIdx] || 0;
			const normalized = ((sample - min) / range) * 2 - 1;
			const drawY = midY - normalized * amplitude;

			if (px === 0) {
				ctx.moveTo(x + px, drawY);
			} else {
				ctx.lineTo(x + px, drawY);
			}
		}
		ctx.stroke();
	}

	_drawRaster(ctx, mu, x, y, w, h, color, reliable, activeBuf) {
		// Raster background
		ctx.fillStyle = this.colors.background;
		ctx.fillRect(x, y, w, h);

		if (!reliable) {
			ctx.fillStyle = this.colors.textDim;
			ctx.font = '9px monospace';
			ctx.fillText('SIL below threshold', x + 4, y + h / 2 + 3);
			return;
		}

		const spikes = this.spikeHistory[mu];
		if (!spikes || spikes.length === 0) return;

		const maxBuf = Math.max(this.bufferSize, this.bufferSizeFocused);
		const startIdx = maxBuf - activeBuf;
		const tickHalf = h * 0.4;
		const midY = y + h / 2;

		ctx.strokeStyle = color;
		ctx.lineWidth = 1.5;

		for (const bufIdx of spikes) {
			if (bufIdx < startIdx) continue; // outside visible range
			const xPos = x + ((bufIdx - startIdx) / activeBuf) * w;
			ctx.beginPath();
			ctx.moveTo(xPos, midY - tickHalf);
			ctx.lineTo(xPos, midY + tickHalf);
			ctx.stroke();
		}
	}

	// -------------------------------------------------------------------------
	// Focus mode & FR bars
	// -------------------------------------------------------------------------

	/**
	 * Get the list of MU indices to render.
	 * In focus mode, only the two classification MUs; otherwise all.
	 */
	_getMUsToShow() {
		if (this.nMUs === 0) return [];
		if (this.focusMode && this.classificationMU1 !== null && this.classificationMU2 !== null) {
			const set = new Set([this.classificationMU1, this.classificationMU2]);
			return [...set].sort((a, b) => a - b);
		}
		return Array.from({ length: this.nMUs }, (_, i) => i);
	}

	_toggleFocusMode() {
		if (this.classificationMU1 === null || this.classificationMU2 === null) return;
		this.focusMode = !this.focusMode;

		if (this.focusMode) {
			this.focusToggle.style.background = this.colors.badgeMove1;
			this.focusToggle.style.color = '#fff';
			this.focusToggle.textContent = 'FOCUS: ON';
		} else {
			this.focusToggle.style.background = this.colors.border;
			this.focusToggle.style.color = this.colors.textDim;
			this.focusToggle.textContent = 'FOCUS';
		}
		this._resizeCanvas();
	}

	/**
	 * Draw the firing rate bars for the two classification MUs.
	 */
	_drawFRBars() {
		const ctx = this.frBarCtx;
		const W = this.frBarCanvas.width;
		const H = this.frBarCanvas.height;
		if (!W || !H) return;

		ctx.fillStyle = this.colors.panel;
		ctx.fillRect(0, 0, W, H);

		const mu1 = this.classificationMU1;
		const mu2 = this.classificationMU2;
		const fr1 = this.firingRates[mu1] ?? 0;
		const fr2 = this.firingRates[mu2] ?? 0;
		const threshold = this.classificationThreshold;
		const maxFR = Math.max(40, threshold * 2);

		const barAreaW = (W - 60) / 2; // two bars side by side
		const barH = H - 40; // vertical space for the bars
		const topY = 20;

		const bars = [
			{ label: `MU${mu1}`, fr: fr1, color: this.muColors[mu1 % this.muColors.length], x: 30 },
			{ label: `MU${mu2}`, fr: fr2, color: this.muColors[mu2 % this.muColors.length], x: 30 + barAreaW + 10 },
		];

		for (const bar of bars) {
			const barW = Math.min(barAreaW, 60);
			const bx = bar.x + (barAreaW - barW) / 2;

			// Background
			ctx.fillStyle = this.colors.background;
			ctx.fillRect(bx, topY, barW, barH);

			// Fill level
			const fillH = Math.min(1, bar.fr / maxFR) * barH;
			const active = bar.fr >= threshold;
			ctx.fillStyle = active ? bar.color : '#555566';
			ctx.fillRect(bx, topY + barH - fillH, barW, fillH);

			// Threshold line
			const threshY = topY + barH - (threshold / maxFR) * barH;
			ctx.strokeStyle = '#ff4444';
			ctx.lineWidth = 2;
			ctx.setLineDash([4, 3]);
			ctx.beginPath();
			ctx.moveTo(bx - 4, threshY);
			ctx.lineTo(bx + barW + 4, threshY);
			ctx.stroke();
			ctx.setLineDash([]);

			// FR value text
			ctx.fillStyle = this.colors.text;
			ctx.font = 'bold 12px monospace';
			ctx.textAlign = 'center';
			ctx.fillText(`${bar.fr.toFixed(1)}`, bx + barW / 2, topY - 5);

			// Label
			ctx.fillStyle = this.colors.textDim;
			ctx.font = '10px monospace';
			ctx.fillText(bar.label, bx + barW / 2, topY + barH + 14);
		}

		// Threshold label
		ctx.fillStyle = '#ff4444';
		ctx.font = '9px monospace';
		ctx.textAlign = 'left';
		const threshY = topY + barH - (threshold / maxFR) * barH;
		ctx.fillText(`${threshold}Hz`, W - 40, threshY + 3);
		ctx.textAlign = 'left';
	}
}
