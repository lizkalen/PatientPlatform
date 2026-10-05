/**
 * Trigger Monitor
 * ===============
 *
 * A small live waveform of one channel (the stim trigger), for verifying the
 * trigger during setup. Draws a rolling ~10 s min/max envelope so oscillating
 * regions (stim active during a movement) show as tall bars and rest shows as a
 * flat line, plus the mid-line threshold the bout detector uses. This makes it
 * obvious whether the selected channel actually marks the movements.
 */
export default class TriggerMonitor {
	constructor(emgClient) {
		this.client = emgClient;
		this.channel = null;
		this.W = 460; this.H = 130;
		this.bufN = 24000;                 // ~10 s at 120 pts/chunk x 20 chunks/s
		this.buf = new Float32Array(this.bufN);
		this.filled = 0;
		this._build();
		this.client.onTriggerChunk?.((msg) => this._onChunk(msg));
	}

	// -- lifecycle ----------------------------------------------------------------
	start(channel) {
		this.channel = channel;
		this.filled = 0; this.buf.fill(0);
		this.title.textContent = `Trigger — channel ${channel}`;
		this.range.textContent = '';
		this.el.style.display = 'block';
		this.client.monitorTrigger(channel);
	}
	stop() {
		this.el.style.display = 'none';
		this.client.monitorTrigger(null);
	}
	isOpen() { return this.el.style.display !== 'none'; }

	// -- build --------------------------------------------------------------------
	_build() {
		const el = document.createElement('div');
		Object.assign(el.style, {
			position: 'fixed', bottom: '20px', right: '20px', zIndex: 940, display: 'none',
			padding: '10px 12px', borderRadius: '10px', background: 'rgba(16,19,26,0.94)',
			color: '#e6e9f0', font: '12px system-ui, sans-serif',
			boxShadow: '0 8px 26px rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)',
		});
		const head = document.createElement('div');
		Object.assign(head.style, { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px', gap: '10px' });
		this.title = document.createElement('span');
		this.title.style.fontWeight = '600';
		const close = document.createElement('button');
		close.textContent = 'Close';
		Object.assign(close.style, { border: '1px solid #3a4152', background: 'transparent', color: '#c3ccdd', borderRadius: '6px', padding: '3px 8px', cursor: 'pointer', fontSize: '11px' });
		close.onclick = () => this.stop();
		head.append(this.title, close);

		this.canvas = document.createElement('canvas');
		this.canvas.width = this.W; this.canvas.height = this.H;
		this.canvas.style.cssText = `display:block; background:#0e1117; border-radius:6px;`;
		this.ctx = this.canvas.getContext('2d');

		this.range = document.createElement('div');
		Object.assign(this.range.style, { opacity: '0.6', fontSize: '11px', marginTop: '4px' });

		el.append(head, this.canvas, this.range);
		document.body.appendChild(el);
		this.el = el;
	}

	// -- data + draw --------------------------------------------------------------
	_onChunk(msg) {
		if (this.channel == null || msg.channel !== this.channel) return;
		const d = msg.data || [];
		const n = d.length;
		if (!n) return;
		if (n >= this.bufN) {
			this.buf.set(d.slice(n - this.bufN));
			this.filled = this.bufN;
		} else {
			this.buf.copyWithin(0, n);
			this.buf.set(d, this.bufN - n);
			this.filled = Math.min(this.bufN, this.filled + n);
		}
		this._draw();
	}

	_draw() {
		const { ctx, W, H } = this;
		ctx.clearRect(0, 0, W, H);
		const start = this.bufN - this.filled;
		if (this.filled < 2) return;

		let gmin = Infinity, gmax = -Infinity;
		for (let i = start; i < this.bufN; i++) { const v = this.buf[i]; if (v < gmin) gmin = v; if (v > gmax) gmax = v; }
		if (gmax - gmin < 1e-6) { gmax = gmin + 1; }
		const y = (v) => H - 4 - ((v - gmin) / (gmax - gmin)) * (H - 8);

		// mid-line threshold used by the bout detector
		const mid = 0.5 * (gmin + gmax);
		ctx.strokeStyle = 'rgba(224,165,63,0.7)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
		ctx.beginPath(); ctx.moveTo(0, y(mid)); ctx.lineTo(W, y(mid)); ctx.stroke(); ctx.setLineDash([]);

		// min/max envelope, one vertical bar per pixel column
		ctx.strokeStyle = '#5aa9ff'; ctx.lineWidth = 1;
		const span = this.filled / W;
		ctx.beginPath();
		for (let c = 0; c < W; c++) {
			const a = start + Math.floor(c * span);
			const b = Math.min(this.bufN, start + Math.floor((c + 1) * span));
			let mn = Infinity, mx = -Infinity;
			for (let i = a; i < b; i++) { const v = this.buf[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
			if (mn === Infinity) continue;
			ctx.moveTo(c + 0.5, y(mx)); ctx.lineTo(c + 0.5, y(mn));
		}
		ctx.stroke();

		const osc = gmax - gmin;
		this.range.textContent = `range [${gmin.toFixed(0)}, ${gmax.toFixed(0)}]` +
			(osc < 50 ? ' — looks FLAT (no stim?)' : ' — tall bars = active/movement, flat = rest');
	}
}
