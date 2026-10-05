/**
 * Mode Menu
 * =========
 *
 * A single dropdown (top-right) to choose what the operator is doing, replacing
 * the separate floating Mode toggle and the movement-session launcher:
 *
 *   • Config          — full admin interface
 *   • Patient         — patient recording view
 *   • Movement Session — the guided record → train → calibrate → online flow
 *
 * Selecting a mode switches PatientModeController (or opens/closes the movement
 * session). One control, no floating buttons stacking on each other.
 */
import { MODE } from './PatientModeController';

const ITEMS = [
	{ key: 'config', label: 'Config', hint: 'Setup & configuration' },
	{ key: 'patient', label: 'Patient', hint: 'Patient recording view' },
	{ key: 'movement', label: 'Movement Session', hint: 'Guided train → calibrate → live' },
];

export default class ModeMenu {
	constructor(app) {
		this.app = app;
		this.modeController = app.modeController;
		this.session = app.movementSessionView;
		this._open = false;
		this._build();
		this.modeController?.onModeChange(() => this._sync());
		document.addEventListener('click', (e) => {
			if (this._open && !this.el.contains(e.target)) this._toggle(false);
		});
		this._sync();
	}

	_build() {
		this.el = document.createElement('div');
		Object.assign(this.el.style, { position: 'fixed', top: '20px', right: '20px', zIndex: 960, fontFamily: 'system-ui, sans-serif' });

		this.button = document.createElement('button');
		Object.assign(this.button.style, {
			display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 16px',
			background: 'rgba(0,0,0,0.78)', border: '1px solid rgba(255,255,255,0.2)',
			borderRadius: '22px', color: '#fff', cursor: 'pointer', backdropFilter: 'blur(10px)',
			font: '12px system-ui', letterSpacing: '0.5px',
		});
		this.dot = document.createElement('span');
		Object.assign(this.dot.style, { width: '10px', height: '10px', borderRadius: '50%', background: '#ff9900' });
		this.label = document.createElement('span');
		this.caret = document.createElement('span');
		this.caret.textContent = '▾';
		this.caret.style.opacity = '0.6';
		this.button.append(this.dot, this.label, this.caret);
		this.button.onclick = (e) => { e.stopPropagation(); this._toggle(); };

		this.menu = document.createElement('div');
		Object.assign(this.menu.style, {
			display: 'none', marginTop: '8px', background: 'rgba(20,23,30,0.97)',
			border: '1px solid rgba(255,255,255,0.14)', borderRadius: '12px', overflow: 'hidden',
			boxShadow: '0 10px 30px rgba(0,0,0,0.5)', backdropFilter: 'blur(10px)', minWidth: '210px',
		});
		this._itemEls = {};
		for (const it of ITEMS) {
			const row = document.createElement('button');
			Object.assign(row.style, {
				display: 'block', width: '100%', textAlign: 'left', padding: '10px 14px',
				background: 'transparent', border: 'none', color: '#e6e9f0', cursor: 'pointer', font: '13px system-ui',
			});
			row.innerHTML = `<div style="font-weight:600">${it.label}</div>` +
				`<div style="font-size:11px;opacity:0.55">${it.hint}</div>`;
			row.onmouseenter = () => { row.style.background = 'rgba(255,255,255,0.06)'; };
			row.onmouseleave = () => { row.style.background = this._activeKey === it.key ? 'rgba(90,169,255,0.14)' : 'transparent'; };
			row.onclick = (e) => { e.stopPropagation(); this._select(it.key); };
			this.menu.appendChild(row);
			this._itemEls[it.key] = row;
		}

		this.el.append(this.button, this.menu);
		document.body.appendChild(this.el);
	}

	_toggle(v = !this._open) {
		this._open = v;
		this.menu.style.display = v ? 'block' : 'none';
		this.caret.textContent = v ? '▴' : '▾';
	}

	_select(key) {
		if (key === 'movement') {
			this.session?.open();
		} else {
			if (this.session?.isOpen()) this.session.close();   // reset() also sets config mode
			this.modeController?.setMode(key);
		}
		this._toggle(false);
		this._sync();
	}

	_sync() {
		const sessionOpen = !!this.session?.isOpen?.();
		this._activeKey = sessionOpen ? 'movement' : this.modeController?.getMode();
		const active = ITEMS.find((i) => i.key === this._activeKey) || ITEMS[0];
		this.label.textContent = active.label;
		this.dot.style.background = this._activeKey === 'movement' ? '#5aa9ff'
			: (this._activeKey === MODE.PATIENT ? '#00cc66' : '#ff9900');
		for (const it of ITEMS) {
			this._itemEls[it.key].style.background = it.key === this._activeKey ? 'rgba(90,169,255,0.14)' : 'transparent';
		}
	}
}
