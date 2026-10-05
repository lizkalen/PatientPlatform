/**
 * Sequence Configuration Panel
 * ============================
 *
 * A modal panel for editing sequence configuration:
 * - Metadata (subjectId, sessionId, notes)
 * - Settings (rest times, prep time, playback speed, etc.)
 * - Items (add/remove/edit sequence items)
 */

import WebGLView from '../webgl/WebGLView';
import sequenceConfig from '../config/sequenceConfig';
import { loadSavedConfig, saveConfig, normalizeConfig, clearSavedConfig } from '../config/configStore';
import {
	CONFIG_SECTIONS, STIM_CHANNEL_FIELDS, defaultStimChannel,
	getAtPath, setAtPath, parseFieldValue,
} from '../config/sequenceSchema';

export default class SequenceConfigPanel {
	constructor(app) {
		this.app = app;
		this.isVisible = false;

		// Working copy of config (edited, not yet applied)
		this.workingConfig = null;

		// Schema-driven widgets, keyed by config path: { [path]: { field, input } }.
		// Populated by _createSchemaSection; drives _loadConfigToUI/_readConfigFromUI.
		this.fields = {};

		// Available models for dropdown
		this.availableModels = WebGLView.AVAILABLE_MODELS;

		// Colors
		this.colors = {
			background: '#1a1a2e',
			sectionBg: '#0f0f1e',
			border: '#3a3a5e',
			text: '#ffffff',
			textMuted: '#888899',
			accent: '#4a9eff',
			danger: '#ff4444',
			success: '#00cc66',
			inputBg: '#252540',
		};

		this._createElements();
	}

	_createElements() {
		// Overlay
		this.overlay = document.createElement('div');
		this.overlay.className = 'config-overlay';
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

		// Modal
		this.modal = document.createElement('div');
		this.modal.className = 'config-modal';
		this.modal.style.cssText = `
			background: ${this.colors.background};
			border-radius: 12px;
			padding: 0;
			box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
			border: 1px solid ${this.colors.border};
			width: 700px;
			max-width: 90vw;
			max-height: 90vh;
			display: flex;
			flex-direction: column;
			overflow: hidden;
		`;

		// Header
		this.header = this._createHeader();

		// Content (scrollable)
		this.content = document.createElement('div');
		this.content.style.cssText = `
			flex: 1;
			overflow-y: auto;
			padding: 20px;
		`;

		// Sections
		this.metadataSection = this._createMetadataSection();
		this.settingsSection = this._createSettingsSection();
		this.stimulationSection = this._createStimulationSection();
		this.itemsSection = this._createItemsSection();

		this.content.appendChild(this.metadataSection);
		this.content.appendChild(this.settingsSection);
		this.content.appendChild(this.stimulationSection);
		this.content.appendChild(this.itemsSection);

		// Footer
		this.footer = this._createFooter();

		// Assemble
		this.modal.appendChild(this.header);
		this.modal.appendChild(this.content);
		this.modal.appendChild(this.footer);
		this.overlay.appendChild(this.modal);

		// Close on overlay click
		this.overlay.addEventListener('click', (e) => {
			if (e.target === this.overlay) this.hide();
		});

		// Close on Escape
		document.addEventListener('keydown', (e) => {
			if (e.key === 'Escape' && this.isVisible) this.hide();
		});

		document.body.appendChild(this.overlay);
	}

	_createHeader() {
		const header = document.createElement('div');
		header.style.cssText = `
			display: flex;
			justify-content: space-between;
			align-items: center;
			padding: 15px 20px;
			border-bottom: 1px solid ${this.colors.border};
			background: ${this.colors.sectionBg};
		`;

		const title = document.createElement('h2');
		title.textContent = 'Sequence Configuration';
		title.style.cssText = `
			margin: 0;
			color: ${this.colors.text};
			font-size: 18px;
			font-weight: 500;
		`;

		const closeBtn = document.createElement('button');
		closeBtn.textContent = '\u00d7';
		closeBtn.style.cssText = `
			background: none;
			border: none;
			color: ${this.colors.text};
			font-size: 24px;
			cursor: pointer;
			padding: 0 5px;
			opacity: 0.7;
		`;
		closeBtn.onmouseover = () => (closeBtn.style.opacity = '1');
		closeBtn.onmouseout = () => (closeBtn.style.opacity = '0.7');
		closeBtn.onclick = () => this.hide();

		header.appendChild(title);
		header.appendChild(closeBtn);
		return header;
	}

	_createFooter() {
		const footer = document.createElement('div');
		footer.style.cssText = `
			display: flex;
			justify-content: space-between;
			align-items: center;
			padding: 15px 20px;
			border-top: 1px solid ${this.colors.border};
			background: ${this.colors.sectionBg};
		`;

		// Left side - Import/Export
		const leftBtns = document.createElement('div');
		leftBtns.style.cssText = 'display: flex; gap: 10px;';

		const exportBtn = this._createButton('Export JSON', () => this._exportConfig(), 'secondary');
		const importBtn = this._createButton('Import JSON', () => this._importConfig(), 'secondary');
		const resetBtn = this._createButton('Reset to Defaults', () => this._resetToDefaults(), 'secondary');
		resetBtn.title = 'Discard the saved config and reload the built-in default sequence';

		leftBtns.appendChild(exportBtn);
		leftBtns.appendChild(importBtn);
		leftBtns.appendChild(resetBtn);

		// Right side - Cancel/Apply
		const rightBtns = document.createElement('div');
		rightBtns.style.cssText = 'display: flex; gap: 10px;';

		const cancelBtn = this._createButton('Cancel', () => this.hide(), 'secondary');
		const applyBtn = this._createButton('Apply', () => this._applyConfig(), 'primary');

		rightBtns.appendChild(cancelBtn);
		rightBtns.appendChild(applyBtn);

		footer.appendChild(leftBtns);
		footer.appendChild(rightBtns);
		return footer;
	}

	_createButton(text, onClick, type = 'secondary') {
		const btn = document.createElement('button');
		btn.textContent = text;

		const isPrimary = type === 'primary';
		btn.style.cssText = `
			padding: 8px 16px;
			background: ${isPrimary ? this.colors.accent : this.colors.inputBg};
			border: 1px solid ${isPrimary ? this.colors.accent : this.colors.border};
			border-radius: 6px;
			color: ${this.colors.text};
			font-size: 14px;
			cursor: pointer;
			transition: all 0.2s;
		`;
		btn.onmouseover = () => {
			btn.style.background = isPrimary ? '#5aafff' : this.colors.border;
		};
		btn.onmouseout = () => {
			btn.style.background = isPrimary ? this.colors.accent : this.colors.inputBg;
		};
		btn.onclick = onClick;
		return btn;
	}

	_createSection(title) {
		const section = document.createElement('div');
		section.style.cssText = `
			background: ${this.colors.sectionBg};
			border-radius: 8px;
			padding: 15px;
			margin-bottom: 15px;
			border: 1px solid ${this.colors.border};
		`;

		const header = document.createElement('h3');
		header.textContent = title;
		header.style.cssText = `
			margin: 0 0 15px 0;
			color: ${this.colors.text};
			font-size: 14px;
			font-weight: 600;
			text-transform: uppercase;
			letter-spacing: 1px;
		`;

		section.appendChild(header);
		return section;
	}

	_createInputRow(label, inputElement) {
		const row = document.createElement('div');
		row.style.cssText = `
			display: flex;
			align-items: center;
			margin-bottom: 10px;
		`;

		const labelEl = document.createElement('label');
		labelEl.textContent = label;
		labelEl.style.cssText = `
			width: 140px;
			color: ${this.colors.textMuted};
			font-size: 13px;
		`;

		row.appendChild(labelEl);
		row.appendChild(inputElement);
		return row;
	}

	_createTextInput(placeholder = '') {
		const input = document.createElement('input');
		input.type = 'text';
		input.placeholder = placeholder;
		input.style.cssText = `
			flex: 1;
			padding: 8px 12px;
			background: ${this.colors.inputBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			font-size: 13px;
			outline: none;
		`;
		input.onfocus = () => (input.style.borderColor = this.colors.accent);
		input.onblur = () => (input.style.borderColor = this.colors.border);
		return input;
	}

	_createNumberInput(min = 0, max = 100, step = 0.1) {
		const input = document.createElement('input');
		input.type = 'number';
		input.min = min;
		input.max = max;
		input.step = step;
		input.style.cssText = `
			width: 100px;
			padding: 8px 12px;
			background: ${this.colors.inputBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			font-size: 13px;
			outline: none;
		`;
		input.onfocus = () => (input.style.borderColor = this.colors.accent);
		input.onblur = () => (input.style.borderColor = this.colors.border);
		return input;
	}

	_createCheckbox() {
		const checkbox = document.createElement('input');
		checkbox.type = 'checkbox';
		checkbox.style.cssText = `
			width: 18px;
			height: 18px;
			cursor: pointer;
		`;
		return checkbox;
	}

	/**
	 * Build the input widget for one schema field.
	 * @param {import('../config/sequenceSchema').Field} field
	 */
	_createFieldWidget(field) {
		switch (field.type) {
			case 'number':
				return this._createNumberInput(field.min, field.max, field.step);

			case 'checkbox':
				return this._createCheckbox();

			case 'select': {
				const select = document.createElement('select');
				select.style.cssText = `
					flex: 1;
					padding: 8px 12px;
					background: ${this.colors.inputBg};
					border: 1px solid ${this.colors.border};
					border-radius: 4px;
					color: ${this.colors.text};
					font-size: 13px;
					outline: none;
				`;
				for (const [value, label] of field.options || []) {
					const opt = document.createElement('option');
					opt.value = value;
					opt.textContent = label;
					select.appendChild(opt);
				}
				return select;
			}

			case 'textarea': {
				const area = document.createElement('textarea');
				area.placeholder = field.placeholder || '';
				area.style.cssText = `
					flex: 1;
					padding: 8px 12px;
					background: ${this.colors.inputBg};
					border: 1px solid ${this.colors.border};
					border-radius: 4px;
					color: ${this.colors.text};
					font-size: 13px;
					outline: none;
					resize: vertical;
					min-height: 60px;
					font-family: inherit;
				`;
				return area;
			}

			case 'text':
			default:
				return this._createTextInput(field.placeholder || '');
		}
	}

	/**
	 * Build a section from its schema entry, registering each widget in `this.fields`
	 * so load/read can drive them all without naming any of them individually.
	 */
	_createSchemaSection(title) {
		const spec = CONFIG_SECTIONS.find((s) => s.title === title);
		const section = this._createSection(title);

		for (const field of spec.fields) {
			const input = this._createFieldWidget(field);
			this.fields[field.path] = { field, input };

			const row = this._createInputRow(field.label, input);
			if (field.unit) {
				const unit = document.createElement('span');
				unit.textContent = field.unit;
				unit.style.cssText = `margin-left: 8px; color: ${this.colors.textMuted}; font-size: 12px;`;
				row.appendChild(unit);
			}
			section.appendChild(row);
		}

		return section;
	}

	_createMetadataSection() {
		return this._createSchemaSection('Trial Metadata');
	}

	_createSettingsSection() {
		return this._createSchemaSection('Timing Settings');
	}

	/** Schema-driven scalars, plus the repeating channel editor. */
	_createStimulationSection() {
		const section = this._createSchemaSection('Stimulation');

		// Channels label
		const channelsLabel = document.createElement('div');
		channelsLabel.textContent = 'Channels';
		channelsLabel.style.cssText = `
			color: ${this.colors.textMuted};
			font-size: 13px;
			margin: 10px 0 8px 0;
		`;
		section.appendChild(channelsLabel);

		// Channels container + add button
		this.stimChannelsContainer = document.createElement('div');
		section.appendChild(this.stimChannelsContainer);

		const addChannelBtn = this._createButton('+ Add Channel', () => this._addStimChannel(), 'secondary');
		addChannelBtn.style.width = '100%';
		section.appendChild(addChannelBtn);

		const hint = document.createElement('div');
		hint.textContent = 'Duration is set to infinite automatically; the train runs until the movement ends.';
		hint.style.cssText = `color: ${this.colors.textMuted}; font-size: 11px; margin-top: 8px;`;
		section.appendChild(hint);

		return section;
	}

	/**
	 * One editable stimulation channel. Used by BOTH channel editors: the roomy
	 * global list under `stimulation.channels`, and the compact per-item override
	 * under `items[i].stim.channels`.
	 *
	 * Writes straight into the `channels` array it is handed, so the caller decides
	 * which list is being edited. Deliberately does NOT create or backfill entries —
	 * an EMPTY per-item list means "inherit the global channels" (see
	 * SequencePlayer._handleStimulationTransition and
	 * MovementSessionController._resolvePatterns), and quietly adding a default
	 * channel here would break that inheritance without any visible error.
	 *
	 * Values are not clamped to min/max, matching both editors' previous behaviour;
	 * unparseable input falls back to the value already in the config.
	 *
	 * @param {object[]} channels - the array being edited (global or per-item)
	 * @param {number} index      - which channel in that array
	 * @param {object} opts
	 * @param {boolean} [opts.compact] - narrow layout + short labels (per-item rows)
	 * @param {Function} opts.onRemove - called after this channel is spliced out
	 */
	_createChannelRow(channels, index, { compact = false, onRemove } = {}) {
		const channel = channels[index];

		const row = document.createElement('div');
		row.style.cssText = compact
			? 'display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:3px 0;'
			: `display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 10px;
			   background: ${this.colors.inputBg}; border-radius: 6px; margin-bottom: 8px;
			   border: 1px solid ${this.colors.border};`;

		const labelCss = `color: ${this.colors.textMuted}; font-size: 11px;`;
		const inputCss = compact
			? `width:58px; padding:4px 6px; background:${this.colors.sectionBg};
			   border:1px solid ${this.colors.border}; border-radius:4px;
			   color:${this.colors.text}; font-size:11px;`
			: `width: 80px; padding: 6px 8px; background: ${this.colors.sectionBg};
			   border: 1px solid ${this.colors.border}; border-radius: 4px;
			   color: ${this.colors.text}; font-size: 12px;`;

		for (const field of STIM_CHANNEL_FIELDS) {
			const labelEl = document.createElement('span');
			labelEl.textContent = compact ? field.short : field.label;
			labelEl.style.cssText = labelCss;

			let input;
			if (field.type === 'checkbox') {
				input = this._createCheckbox();
				input.checked = channel[field.key] ?? field.default;
				input.onchange = () => { channel[field.key] = input.checked; };
			} else {
				input = document.createElement('input');
				input.type = 'number';
				input.min = field.min;
				input.max = field.max;
				input.step = field.step;
				input.value = channel[field.key] ?? field.default;
				input.style.cssText = inputCss;
				input.onchange = () => {
					channel[field.key] = parseFieldValue(field, input.value, channel[field.key]);
				};
			}

			if (compact) {
				row.appendChild(labelEl);
				row.appendChild(input);
			} else {
				const wrap = document.createElement('div');
				wrap.style.cssText = 'display: flex; flex-direction: column; gap: 2px;'
					+ (field.type === 'checkbox' ? ' align-items: center;' : '');
				wrap.appendChild(labelEl);
				wrap.appendChild(input);
				row.appendChild(wrap);
			}
		}

		const removeBtn = document.createElement('button');
		removeBtn.textContent = compact ? 'Remove' : '×';
		removeBtn.title = 'Remove channel';
		removeBtn.style.cssText = compact
			? `padding:3px 8px; cursor:pointer; background:${this.colors.sectionBg};
			   border:1px solid ${this.colors.border}; border-radius:4px;
			   color:${this.colors.text}; font-size:11px;`
			: `margin-left: auto; padding: 4px 8px; background: ${this.colors.danger}33;
			   border: 1px solid ${this.colors.danger}; border-radius: 4px;
			   color: ${this.colors.danger}; cursor: pointer; font-size: 14px;`;
		removeBtn.onclick = () => {
			channels.splice(index, 1);
			onRemove();
		};
		row.appendChild(removeBtn);

		return row;
	}

	_renderStimChannels() {
		this.stimChannelsContainer.innerHTML = '';
		const channels = this.workingConfig?.stimulation?.channels || [];
		channels.forEach((_, index) => {
			this.stimChannelsContainer.appendChild(
				this._createChannelRow(channels, index, { onRemove: () => this._renderStimChannels() }),
			);
		});
	}

	_addStimChannel() {
		if (!this.workingConfig.stimulation) this.workingConfig.stimulation = {};
		if (!this.workingConfig.stimulation.channels) this.workingConfig.stimulation.channels = [];
		const channels = this.workingConfig.stimulation.channels;
		const maxId = channels.reduce((m, ch) => Math.max(m, ch.id || 0), 0);
		channels.push(defaultStimChannel(Math.min(maxId + 1, 8)));
		this._renderStimChannels();
	}

	_createItemsSection() {
		const section = this._createSection('Sequence Items');

		// Items container
		this.itemsContainer = document.createElement('div');
		this.itemsContainer.style.cssText = `
			margin-bottom: 15px;
		`;
		section.appendChild(this.itemsContainer);

		// Add item button
		const addBtn = this._createButton('+ Add Item', () => this._addItem(), 'secondary');
		addBtn.style.width = '100%';
		section.appendChild(addBtn);

		return section;
	}

	_createItemRow(item, index) {
		const row = document.createElement('div');
		row.style.cssText = `
			display: flex;
			align-items: center;
			gap: 10px;
			padding: 10px;
			background: ${this.colors.inputBg};
			border-radius: 6px;
			margin-bottom: 8px;
			border: 1px solid ${this.colors.border};
		`;

		// Index badge
		const indexBadge = document.createElement('span');
		indexBadge.textContent = index + 1;
		indexBadge.style.cssText = `
			width: 24px;
			height: 24px;
			background: ${this.colors.border};
			border-radius: 50%;
			display: flex;
			align-items: center;
			justify-content: center;
			font-size: 12px;
			color: ${this.colors.text};
			flex-shrink: 0;
		`;
		row.appendChild(indexBadge);

		// Model dropdown
		const modelSelect = document.createElement('select');
		modelSelect.style.cssText = `
			padding: 6px 10px;
			background: ${this.colors.sectionBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			font-size: 12px;
			flex: 1;
			min-width: 120px;
		`;
		this.availableModels.forEach((model) => {
			const option = document.createElement('option');
			option.value = model.file;
			option.textContent = model.name;
			if (model.file === item.model) option.selected = true;
			modelSelect.appendChild(option);
		});
		modelSelect.onchange = () => {
			this.workingConfig.items[index].model = modelSelect.value;
			// Reset animation to 0 when model changes
			this.workingConfig.items[index].animation = 0;
			animInput.value = 0;
		};
		row.appendChild(modelSelect);

		// Animation input
		const animLabel = document.createElement('span');
		animLabel.textContent = 'Anim:';
		animLabel.style.cssText = `color: ${this.colors.textMuted}; font-size: 12px;`;
		row.appendChild(animLabel);

		const animInput = document.createElement('input');
		animInput.type = 'number';
		animInput.min = 0;
		animInput.max = 10;
		animInput.value = item.animation || 0;
		animInput.style.cssText = `
			width: 50px;
			padding: 6px 8px;
			background: ${this.colors.sectionBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			font-size: 12px;
		`;
		animInput.onchange = () => {
			this.workingConfig.items[index].animation = parseInt(animInput.value) || 0;
		};
		row.appendChild(animInput);

		// Repetitions input
		const repsLabel = document.createElement('span');
		repsLabel.textContent = 'Reps:';
		repsLabel.style.cssText = `color: ${this.colors.textMuted}; font-size: 12px;`;
		row.appendChild(repsLabel);

		const repsInput = document.createElement('input');
		repsInput.type = 'number';
		repsInput.min = 1;
		repsInput.max = 50;
		repsInput.value = item.repetitions || 1;
		repsInput.style.cssText = `
			width: 50px;
			padding: 6px 8px;
			background: ${this.colors.sectionBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			font-size: 12px;
		`;
		repsInput.onchange = () => {
			this.workingConfig.items[index].repetitions = parseInt(repsInput.value) || 1;
		};
		row.appendChild(repsInput);

		// Stimulate toggle (only meaningful when stimulation is enabled globally)
		const stimLabel = document.createElement('span');
		stimLabel.textContent = 'Stim:';
		stimLabel.title = 'Deliver a stimulation train during this item’s movement (requires stimulation enabled)';
		stimLabel.style.cssText = `color: ${this.colors.textMuted}; font-size: 12px;`;
		row.appendChild(stimLabel);

		const stimCheckbox = this._createCheckbox();
		stimCheckbox.checked = !!item.stimulate;
		stimCheckbox.onchange = () => {
			this.workingConfig.items[index].stimulate = stimCheckbox.checked;
		};
		row.appendChild(stimCheckbox);

		// Rest-label toggle: this stimulated bout is cued like a movement but LABELED
		// rest — it feeds the stim head genuine "stim on + at rest" examples so it can
		// recognize return-to-rest under stimulation. Requires Stim on (the trigger must
		// fire for the bout to be segmented). Excluded from the movement classes + calib.
		const restLabel = document.createElement('span');
		restLabel.textContent = 'Rest:';
		restLabel.title = 'Label this STIMULATED bout as rest (stimulate, but the patient holds still). '
			+ 'Trains the stim head to recognize rest under stimulation. Requires Stim on.';
		restLabel.style.cssText = `color: ${this.colors.textMuted}; font-size: 12px;`;
		row.appendChild(restLabel);

		const restCheckbox = this._createCheckbox();
		restCheckbox.checked = !!item.restClass;
		restCheckbox.onchange = () => {
			this.workingConfig.items[index].restClass = restCheckbox.checked;
		};
		row.appendChild(restCheckbox);

		// Move up button
		const upBtn = document.createElement('button');
		upBtn.textContent = '\u2191';
		upBtn.title = 'Move up';
		upBtn.style.cssText = `
			padding: 4px 8px;
			background: ${this.colors.sectionBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			cursor: pointer;
			font-size: 14px;
		`;
		upBtn.onclick = () => this._moveItem(index, -1);
		upBtn.disabled = index === 0;
		if (index === 0) upBtn.style.opacity = '0.3';
		row.appendChild(upBtn);

		// Move down button
		const downBtn = document.createElement('button');
		downBtn.textContent = '\u2193';
		downBtn.title = 'Move down';
		downBtn.style.cssText = `
			padding: 4px 8px;
			background: ${this.colors.sectionBg};
			border: 1px solid ${this.colors.border};
			border-radius: 4px;
			color: ${this.colors.text};
			cursor: pointer;
			font-size: 14px;
		`;
		downBtn.onclick = () => this._moveItem(index, 1);
		row.appendChild(downBtn);

		// Delete button
		const deleteBtn = document.createElement('button');
		deleteBtn.textContent = '\u00d7';
		deleteBtn.title = 'Remove item';
		deleteBtn.style.cssText = `
			padding: 4px 8px;
			background: ${this.colors.danger}33;
			border: 1px solid ${this.colors.danger};
			border-radius: 4px;
			color: ${this.colors.danger};
			cursor: pointer;
			font-size: 14px;
		`;
		deleteBtn.onclick = () => this._removeItem(index);
		row.appendChild(deleteBtn);

		// Wrap the main row + a per-movement stim-channel list (used for BOTH the
		// recording phases and the online closed-loop). Shown when "Stim" is on.
		row.style.marginBottom = '0';
		const wrapper = document.createElement('div');
		wrapper.style.cssText = 'display:flex; flex-direction:column; gap:4px; margin-bottom:8px;';
		wrapper.appendChild(row);

		const stimSection = document.createElement('div');
		stimSection.style.cssText = `display:${item.stimulate ? 'block' : 'none'}; padding-left:42px;`;
		this._renderItemStimChannels(stimSection, index);
		wrapper.appendChild(stimSection);

		stimCheckbox.onchange = () => {
			this.workingConfig.items[index].stimulate = stimCheckbox.checked;
			if (stimCheckbox.checked) this._ensureItemStim(index);
			stimSection.style.display = stimCheckbox.checked ? 'block' : 'none';
			this._renderItemStimChannels(stimSection, index);
		};

		return wrapper;
	}

	/** Per-movement stim channels (recording + online). A stacked list of channel
	 * rows + an "Add channel" button; empty means "use the global channels". */
	_renderItemStimChannels(container, index) {
		container.innerHTML = '';
		const title = document.createElement('div');
		title.textContent = 'Stim channels for this movement (recording + online) — empty = use global channels';
		title.style.cssText = `color:${this.colors.textMuted}; font-size:11px; margin:2px 0 4px 0;`;
		container.appendChild(title);

		const chans = this.workingConfig.items[index].stim?.channels || [];
		chans.forEach((_, ci) => container.appendChild(
			this._createChannelRow(chans, ci, {
				compact: true,
				onRemove: () => this._renderItemStimChannels(container, index),
			}),
		));

		const addBtn = this._createButton('+ Add channel', () => {
			const stim = this._ensureItemStim(index);
			stim.channels.push(defaultStimChannel(stim.channels.length + 1));
			this._renderItemStimChannels(container, index);
		}, 'secondary');
		addBtn.style.cssText += 'font-size:11px; padding:4px 10px; margin-top:2px;';
		container.appendChild(addBtn);
	}

	_ensureItemStim(index) {
		const it = this.workingConfig.items[index];
		if (!it.stim || !Array.isArray(it.stim.channels)) it.stim = { channels: [] };
		return it.stim;
	}

	_renderItems() {
		this.itemsContainer.innerHTML = '';

		if (!this.workingConfig?.items?.length) {
			const empty = document.createElement('div');
			empty.textContent = 'No items. Click "Add Item" to create one.';
			empty.style.cssText = `
				text-align: center;
				color: ${this.colors.textMuted};
				padding: 20px;
				font-size: 13px;
			`;
			this.itemsContainer.appendChild(empty);
			return;
		}

		this.workingConfig.items.forEach((item, index) => {
			const row = this._createItemRow(item, index);
			this.itemsContainer.appendChild(row);
		});

		// Update down button states
		const rows = this.itemsContainer.children;
		if (rows.length > 0) {
			const lastRow = rows[rows.length - 1];
			const downBtn = lastRow.querySelector('button[title="Move down"]');
			if (downBtn) {
				downBtn.disabled = true;
				downBtn.style.opacity = '0.3';
			}
		}
	}

	_addItem() {
		if (!this.workingConfig.items) {
			this.workingConfig.items = [];
		}

		this.workingConfig.items.push({
			model: this.availableModels[0].file,
			animation: 0,
			repetitions: 3,
		});

		this._renderItems();
	}

	_removeItem(index) {
		this.workingConfig.items.splice(index, 1);
		this._renderItems();
	}

	_moveItem(index, direction) {
		const newIndex = index + direction;
		if (newIndex < 0 || newIndex >= this.workingConfig.items.length) return;

		const items = this.workingConfig.items;
		[items[index], items[newIndex]] = [items[newIndex], items[index]];
		this._renderItems();
	}

	_loadConfigToUI() {
		const config = this.workingConfig;

		for (const { field, input } of Object.values(this.fields)) {
			const value = getAtPath(config, field.path);

			if (field.type === 'checkbox') {
				input.checked = value ?? field.default;
			} else if (field.type === 'number') {
				input.value = value ?? field.default;
			} else {
				// Text-ish: empty falls back to the default, matching the previous
				// `config.x || 'default'` reads (notably stimulator type and URL).
				input.value = value || field.default || '';
			}
		}

		this._renderStimChannels();
		this._renderItems();
	}

	_readConfigFromUI() {
		// setAtPath writes each field in place, so anything the panel has no widget for
		// — nostimTrigger, onlineStimulation, items[].restBetweenReps, hand-added keys —
		// is left untouched rather than being dropped by a wholesale object replace.
		for (const { field, input } of Object.values(this.fields)) {
			const value = field.type === 'checkbox'
				? input.checked
				: parseFieldValue(field, input.value, getAtPath(this.workingConfig, field.path));

			setAtPath(this.workingConfig, field.path, value);
		}

		// Channels (global and per-item) are written directly by their row handlers.
		// An empty global list stays empty here; normalizeConfig seeds a default only
		// when the key is missing entirely.
	}

	_applyConfig() {
		this._readConfigFromUI();

		// Apply to sequence player
		if (this.app.sequencePlayer) {
			this.app.sequencePlayer.loadConfig(this.workingConfig);
		}

		// Persist so the next page load restores this config instead of the default
		saveConfig(this.workingConfig);

		console.log('Sequence config applied:', this.workingConfig);
		this.hide();
	}

	/**
	 * Discard the saved config and go back to the built-in default.
	 *
	 * Takes effect immediately rather than waiting for Apply: clearing storage while
	 * the player kept running the old config would recreate exactly the panel/player
	 * split this button exists to undo.
	 */
	_resetToDefaults() {
		const confirmed = window.confirm(
			'Reset the sequence configuration to the built-in default?\n\n'
			+ 'The saved configuration will be discarded and the default sequence loaded '
			+ 'immediately. Use "Export JSON" first if you want to keep the current one.',
		);
		if (!confirmed) return;

		clearSavedConfig();

		// Deep copy: the editor mutates workingConfig in place, and the default is a
		// shared module object that must not be edited.
		this.workingConfig = normalizeConfig(JSON.parse(JSON.stringify(sequenceConfig)));

		if (this.app.sequencePlayer) {
			this.app.sequencePlayer.loadConfig(this.workingConfig);
		}

		this._loadConfigToUI();
		console.log('Sequence config reset to defaults');
	}

	_exportConfig() {
		this._readConfigFromUI();

		const json = JSON.stringify(this.workingConfig, null, 2);
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);

		const a = document.createElement('a');
		a.href = url;
		a.download = `sequence_config_${this.workingConfig.metadata?.subjectId || 'export'}.json`;
		a.click();

		URL.revokeObjectURL(url);
	}

	_importConfig() {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json';
		input.onchange = (e) => {
			const file = e.target.files[0];
			if (!file) return;

			const reader = new FileReader();
			reader.onload = (ev) => {
				try {
					// Normalize so an older or hand-written JSON can't reach the editor
					// (and then the player) missing fields both assume are present.
					const config = normalizeConfig(JSON.parse(ev.target.result));
					this.workingConfig = config;
					this._loadConfigToUI();
					console.log('Config imported:', config);
				} catch (err) {
					console.error('Failed to import config:', err);
					alert('Invalid JSON file');
				}
			};
			reader.readAsText(file);
		};
		input.click();
	}

	show() {
		// Edit the LIVE config — the one the player is actually running. The saved
		// config is only a fallback for the case where the player somehow has none;
		// App restores it at boot, so normally the two are already the same object.
		// (Preferring the saved copy here is what used to make the panel disagree with
		// whatever was really playing until the operator pressed Apply.)
		const liveConfig = this.app.sequencePlayer?.config;

		this.workingConfig = normalizeConfig(
			JSON.parse(JSON.stringify(liveConfig || loadSavedConfig() || {})),
		);

		this._loadConfigToUI();

		this.isVisible = true;
		this.overlay.style.display = 'flex';
	}

	hide() {
		this.isVisible = false;
		this.overlay.style.display = 'none';
	}

	toggle() {
		if (this.isVisible) {
			this.hide();
		} else {
			this.show();
		}
	}
}
