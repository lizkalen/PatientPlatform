/**
 * EMG Channel Visualization View
 * ===============================
 *
 * Displays EMG channel data in a modal/overlay with:
 * - 8 channels per page
 * - Pagination controls for 64 channels (8 pages)
 * - Real-time waveform visualization
 * - Recording status indicator
 */
export default class EMGChannelView {
	constructor(emgClient) {
		this.emgClient = emgClient;
		this.isVisible = false;

		// Pagination
		this.channelsPerPage = 8;
		this.currentPage = 0;
		this.totalPages = 8; // 64 channels / 8 per page

		// Canvas settings
		this.canvasWidth = 800;
		this.canvasHeight = 600;
		this.channelHeight = 70;
		this.padding = 10;

		// Animation
		this.animationId = null;

		// Colors
		this.colors = {
			background: '#1a1a2e',
			grid: '#2a2a4e',
			waveform: '#00ff88',
			waveformRecording: '#ff4444',
			text: '#ffffff',
			channelBg: '#0f0f1e',
			border: '#3a3a5e',
		};

		// Create DOM elements
		this._createElements();
	}

	_createElements() {
		// Overlay container
		this.overlay = document.createElement('div');
		this.overlay.className = 'emg-overlay';
		this.overlay.style.cssText = `
			position: fixed;
			top: 0;
			left: 0;
			width: 100vw;
			height: 100vh;
			background: rgba(0, 0, 0, 0.85);
			display: none;
			justify-content: center;
			align-items: center;
			z-index: 1000;
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
		`;

		// Modal container
		this.modal = document.createElement('div');
		this.modal.className = 'emg-modal';
		this.modal.style.cssText = `
			background: ${this.colors.background};
			border-radius: 12px;
			padding: 20px;
			box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
			border: 1px solid ${this.colors.border};
			max-width: 90vw;
			max-height: 90vh;
			overflow: hidden;
		`;

		// Header
		this.header = document.createElement('div');
		this.header.style.cssText = `
			display: flex;
			justify-content: space-between;
			align-items: center;
			margin-bottom: 15px;
			padding-bottom: 10px;
			border-bottom: 1px solid ${this.colors.border};
		`;

		// Title
		const title = document.createElement('h2');
		title.textContent = 'EMG Channels';
		title.style.cssText = `
			margin: 0;
			color: ${this.colors.text};
			font-size: 20px;
			font-weight: 500;
		`;

		// Connection status
		this.statusBadge = document.createElement('span');
		this.statusBadge.textContent = 'Disconnected';
		this.statusBadge.style.cssText = `
			padding: 4px 12px;
			border-radius: 12px;
			font-size: 12px;
			font-weight: 500;
			background: #ff4444;
			color: white;
		`;

		// Close button
		const closeBtn = document.createElement('button');
		closeBtn.textContent = '\u00d7';
		closeBtn.style.cssText = `
			background: none;
			border: none;
			color: ${this.colors.text};
			font-size: 28px;
			cursor: pointer;
			padding: 0 10px;
			opacity: 0.7;
			transition: opacity 0.2s;
		`;
		closeBtn.onmouseover = () => (closeBtn.style.opacity = '1');
		closeBtn.onmouseout = () => (closeBtn.style.opacity = '0.7');
		closeBtn.onclick = () => this.hide();

		const titleContainer = document.createElement('div');
		titleContainer.style.cssText = 'display: flex; align-items: center; gap: 15px;';
		titleContainer.appendChild(title);
		titleContainer.appendChild(this.statusBadge);

		this.header.appendChild(titleContainer);
		this.header.appendChild(closeBtn);

		// Recording status indicator
		this.recordingIndicator = document.createElement('div');
		this.recordingIndicator.style.cssText = `
			display: none;
			align-items: center;
			gap: 8px;
			padding: 8px 16px;
			background: #ff4444;
			border-radius: 6px;
			margin-bottom: 15px;
			animation: pulse 1s infinite;
		`;
		this.recordingIndicator.innerHTML = `
			<span style="width: 10px; height: 10px; background: white; border-radius: 50%;"></span>
			<span style="color: white; font-weight: 500;">RECORDING</span>
		`;

		// Live-EMG-disabled notice (shown when the server isn't streaming EMG for
		// plotting, e.g. Quattrocento; recording still works).
		this.liveEmgNotice = document.createElement('div');
		this.liveEmgNotice.style.cssText = `
			display: none;
			align-items: center;
			gap: 8px;
			padding: 8px 16px;
			background: #33334e;
			border: 1px solid ${this.colors.border};
			border-radius: 6px;
			margin-bottom: 15px;
		`;
		this.liveEmgNotice.innerHTML = `
			<span style="color: #ffcc55; font-weight: 600;">Live EMG disabled</span>
			<span style="color: #b0b0c0;">— waveforms are off for performance. Recording is unaffected.</span>
		`;

		// Add pulse animation
		const style = document.createElement('style');
		style.textContent = `
			@keyframes pulse {
				0%, 100% { opacity: 1; }
				50% { opacity: 0.7; }
			}
		`;
		document.head.appendChild(style);

		// Canvas container
		this.canvasContainer = document.createElement('div');
		this.canvasContainer.style.cssText = `
			background: ${this.colors.channelBg};
			border-radius: 8px;
			padding: 10px;
			margin-bottom: 15px;
		`;

		// Canvas
		this.canvas = document.createElement('canvas');
		this.canvas.width = this.canvasWidth;
		this.canvas.height = this.canvasHeight;
		this.canvas.style.cssText = `
			display: block;
			width: 100%;
			height: auto;
		`;
		this.ctx = this.canvas.getContext('2d');

		this.canvasContainer.appendChild(this.canvas);

		// Pagination controls
		this.pagination = document.createElement('div');
		this.pagination.style.cssText = `
			display: flex;
			justify-content: center;
			align-items: center;
			gap: 10px;
		`;

		this.prevBtn = this._createButton('\u2190 Previous', () => this._prevPage());
		this.pageInfo = document.createElement('span');
		this.pageInfo.style.cssText = `
			color: ${this.colors.text};
			font-size: 14px;
			min-width: 150px;
			text-align: center;
		`;
		this.nextBtn = this._createButton('Next \u2192', () => this._nextPage());

		this.pagination.appendChild(this.prevBtn);
		this.pagination.appendChild(this.pageInfo);
		this.pagination.appendChild(this.nextBtn);

		// Assemble
		this.modal.appendChild(this.header);
		this.modal.appendChild(this.recordingIndicator);
		this.modal.appendChild(this.liveEmgNotice);
		this.modal.appendChild(this.canvasContainer);
		this.modal.appendChild(this.pagination);
		this.overlay.appendChild(this.modal);

		// Close on overlay click
		this.overlay.addEventListener('click', (e) => {
			if (e.target === this.overlay) {
				this.hide();
			}
		});

		// Close on Escape
		document.addEventListener('keydown', (e) => {
			if (e.key === 'Escape' && this.isVisible) {
				this.hide();
			}
		});

		document.body.appendChild(this.overlay);

		// Update pagination display
		this._updatePagination();

		// Subscribe to EMG client events
		this._subscribeToClient();
	}

	_createButton(text, onClick) {
		const btn = document.createElement('button');
		btn.textContent = text;
		btn.style.cssText = `
			padding: 8px 16px;
			background: ${this.colors.border};
			border: none;
			border-radius: 6px;
			color: ${this.colors.text};
			font-size: 14px;
			cursor: pointer;
			transition: background 0.2s;
		`;
		btn.onmouseover = () => (btn.style.background = '#4a4a6e');
		btn.onmouseout = () => (btn.style.background = this.colors.border);
		btn.onclick = onClick;
		return btn;
	}

	_subscribeToClient() {
		this.emgClient.onConnect((info) => {
			this.totalPages = Math.ceil(info.nChannels / this.channelsPerPage);
			this._updateStatus(true);
			this._updateLiveEmgNotice(info.liveEmgEnabled);
			this._updatePagination();
		});

		this.emgClient.onDisconnect(() => {
			this._updateStatus(false);
		});

		this.emgClient.onRecordingStatus((status) => {
			this._updateRecordingStatus(status.recording);
		});
	}

	_updateStatus(connected) {
		if (connected) {
			this.statusBadge.textContent = 'Connected';
			this.statusBadge.style.background = '#00cc66';
		} else {
			this.statusBadge.textContent = 'Disconnected';
			this.statusBadge.style.background = '#ff4444';
		}
	}

	_updateRecordingStatus(recording) {
		this.recordingIndicator.style.display = recording ? 'flex' : 'none';
	}

	_updateLiveEmgNotice(liveEmgEnabled) {
		// Undefined (older server) is treated as enabled -> no notice.
		this.liveEmgNotice.style.display = liveEmgEnabled === false ? 'flex' : 'none';
	}

	_updatePagination() {
		const startCh = this.currentPage * this.channelsPerPage + 1;
		const endCh = Math.min(startCh + this.channelsPerPage - 1, (this.emgClient.nChannels || 64));
		this.pageInfo.textContent = `Channels ${startCh}-${endCh} (Page ${this.currentPage + 1}/${this.totalPages})`;

		this.prevBtn.disabled = this.currentPage === 0;
		this.prevBtn.style.opacity = this.currentPage === 0 ? '0.5' : '1';

		this.nextBtn.disabled = this.currentPage >= this.totalPages - 1;
		this.nextBtn.style.opacity = this.currentPage >= this.totalPages - 1 ? '0.5' : '1';
	}

	_prevPage() {
		if (this.currentPage > 0) {
			this.currentPage--;
			this._updatePagination();
		}
	}

	_nextPage() {
		if (this.currentPage < this.totalPages - 1) {
			this.currentPage++;
			this._updatePagination();
		}
	}

	show() {
		this.isVisible = true;
		this.overlay.style.display = 'flex';
		this._updateStatus(this.emgClient.isConnected);
		this._updateRecordingStatus(this.emgClient.isRecording);
		this._updateLiveEmgNotice(this.emgClient.liveEmgEnabled);
		this._startAnimation();
	}

	hide() {
		this.isVisible = false;
		this.overlay.style.display = 'none';
		this._stopAnimation();
	}

	toggle() {
		if (this.isVisible) {
			this.hide();
		} else {
			this.show();
		}
	}

	_startAnimation() {
		const draw = () => {
			if (!this.isVisible) return;
			this._drawChannels();
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

	_drawChannels() {
		const ctx = this.ctx;
		const width = this.canvasWidth;
		const height = this.canvasHeight;

		// Clear canvas
		ctx.fillStyle = this.colors.channelBg;
		ctx.fillRect(0, 0, width, height);

		const startChannel = this.currentPage * this.channelsPerPage;
		const nChannels = this.emgClient.nChannels || 64;

		for (let i = 0; i < this.channelsPerPage; i++) {
			const channelIndex = startChannel + i;
			if (channelIndex >= nChannels) break;

			const y = i * this.channelHeight + this.padding;
			this._drawChannel(channelIndex, y);
		}
	}

	_drawChannel(channelIndex, y) {
		const ctx = this.ctx;
		const width = this.canvasWidth - 2 * this.padding;
		const height = this.channelHeight - 10;
		const x = this.padding;

		// Channel background
		ctx.fillStyle = this.colors.background;
		ctx.fillRect(x, y, width, height);

		// Grid lines
		ctx.strokeStyle = this.colors.grid;
		ctx.lineWidth = 0.5;
		ctx.beginPath();
		// Horizontal center line
		ctx.moveTo(x, y + height / 2);
		ctx.lineTo(x + width, y + height / 2);
		ctx.stroke();

		// Channel label
		ctx.fillStyle = this.colors.text;
		ctx.font = '12px monospace';
		ctx.fillText(`CH ${channelIndex + 1}`, x + 5, y + 15);

		// Get channel data
		const data = this.emgClient.getChannelData(channelIndex);
		if (!data || data.length === 0) return;

		// Draw waveform
		ctx.strokeStyle = this.emgClient.isRecording
			? this.colors.waveformRecording
			: this.colors.waveform;
		ctx.lineWidth = 1.5;
		ctx.beginPath();

		const samplesPerPixel = Math.ceil(data.length / width);
		const midY = y + height / 2;
		const amplitude = (height / 2) * 0.8;

		// Find data range for normalization
		let min = Infinity;
		let max = -Infinity;
		for (let i = 0; i < data.length; i++) {
			if (data[i] < min) min = data[i];
			if (data[i] > max) max = data[i];
		}
		const range = max - min || 1;

		for (let px = 0; px < width; px++) {
			const sampleIndex = Math.floor((px / width) * data.length);
			const sample = data[sampleIndex] || 0;

			// Normalize to [-1, 1] and scale
			const normalized = ((sample - min) / range) * 2 - 1;
			const drawY = midY - normalized * amplitude;

			if (px === 0) {
				ctx.moveTo(x + px, drawY);
			} else {
				ctx.lineTo(x + px, drawY);
			}
		}
		ctx.stroke();

		// Border
		ctx.strokeStyle = this.colors.border;
		ctx.lineWidth = 1;
		ctx.strokeRect(x, y, width, height);
	}
}
