import * as THREE from 'three';

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import { MOVEMENTS } from '../config/movements';

export default class WebGLView {

	// Available models in the public folder. Defined in config/movements.js so the
	// 3D layer and the patient-facing labels can't drift apart; kept as a static
	// here because callers index into WebGLView.AVAILABLE_MODELS.
	static AVAILABLE_MODELS = MOVEMENTS;

	constructor(app) {
		this.app = app;

		// Animation control properties
		this.animationTime = 0;
		this.animationDuration = 0;
		this.isPlaying = false; // Don't auto-start animation
		this.playbackSpeed = 1.0;

		// Current model tracking
		this.currentModelIndex = 0;

		// One-shot callbacks for the next completed model load (see onceModelLoaded)
		this._modelLoadedOnce = [];

		this.initThree();
		this.initLights();
		this.initGrid();
		this.initObject();
		this.initControls();
	}

	/**
	 * Run `callback` once, the next time a model finishes loading.
	 *
	 * Replaces the old single `_onModelLoaded` slot, which SequencePlayer and
	 * DecompositionController both wrote to: the player saved and restored the
	 * previous value, the decomposition controller overwrote it permanently, and
	 * both paths are live at the same time (classification can drive a model swap
	 * mid-sequence). Whoever wrote last silently destroyed the other's callback.
	 *
	 * @param {Function} callback
	 * @returns {Function} cancel - deregisters the callback if it hasn't fired yet
	 */
	onceModelLoaded(callback) {
		this._modelLoadedOnce.push(callback);
		return () => {
			const i = this._modelLoadedOnce.indexOf(callback);
			if (i > -1) this._modelLoadedOnce.splice(i, 1);
		};
	}

	initThree() {
		this.scene = new THREE.Scene();
		this.scene.background = new THREE.Color(0xf0f0f0);

		this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 10000);
		this.camera.position.set(200, 500, 500);

		this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

		this.clock = new THREE.Clock();
	}

	initLights() {
		// Ambient light for overall illumination
		const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
		this.scene.add(ambientLight);

		// Directional light from above
		const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
		directionalLight.position.set(200, 400, 300);
		this.scene.add(directionalLight);

		// Fill light from the side
		const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
		fillLight.position.set(-200, 100, -300);
		this.scene.add(fillLight);
	}

	initControls() {
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enabled = true;
	}

	initGrid() {
		const helper = new THREE.GridHelper(5000, 20);
		helper.position.y = -50;
		helper.material.opacity = 0.5;
		helper.material.transparent = true;
		this.scene.add(helper);
	}

	initObject() {
		// Load the default model
		const defaultModel = WebGLView.AVAILABLE_MODELS[this.currentModelIndex];
		this.loadModel(defaultModel.file);
	}

	/**
	 * Load a model by filename
	 * @param {string} filename - The model filename (e.g., 'wrist.glb')
	 */
	loadModel(filename) {
		const loader = new GLTFLoader();

		loader.load(
			`/${filename}?v=${Date.now()}`, // Path to model file with cache busting
			(gltf) => {
				// Clear old model and mixer if they exist
				if (this.object3D) {
					this.scene.remove(this.object3D);
					this.object3D.traverse((child) => {
						if (child.geometry) child.geometry.dispose();
						if (child.material) {
							if (Array.isArray(child.material)) {
								child.material.forEach(mat => mat.dispose());
							} else {
								child.material.dispose();
							}
						}
					});
				}
				if (this.mixer) {
					this.mixer.stopAllAction();
					this.mixer = null;
				}

				this.object3D = gltf.scene;

				console.log('=== MODEL LOADED ===');
				console.log('File:', `/${filename}`);
				console.log('Animations found:', gltf.animations.length);
				gltf.animations.forEach((clip, index) => {
					console.log(`  [${index}] ${clip.name} - Duration: ${clip.duration.toFixed(2)}s, Tracks: ${clip.tracks.length}`);
				});

				// Center and scale the model if needed
				const box = new THREE.Box3().setFromObject(this.object3D);
				const center = box.getCenter(new THREE.Vector3());
				const size = box.getSize(new THREE.Vector3());

				// Center the model
				this.object3D.position.sub(center);

				// Scale the model to fit in view (increase 200 to make it bigger, decrease to make smaller)
				const maxDim = Math.max(size.x, size.y, size.z);
				const scale = 200 / maxDim;
				this.object3D.scale.multiplyScalar(scale);

				this.scene.add(this.object3D);

				// Store available animations for this model
				this.availableAnimations = gltf.animations;

				// Setup animations if present
				if (gltf.animations && gltf.animations.length > 0) {
					this.mixer = new THREE.AnimationMixer(this.object3D);

					// Load the first animation but don't auto-play
					this.playAnimation(0, false);

					console.log(`Model loaded with ${gltf.animations.length} animation(s)`);
				} else {
					this.availableAnimations = [];
					this.animationDuration = 0;
					this.clipTracks = [];
					this.keyframeTimes = [];
					if (this.app.timeline) {
						this.app.timeline.setClipInfo({
							duration: 0,
							keyframeTimes: [],
							tracks: [],
						});
					}
					console.log('Model loaded successfully (no animations found)');
				}

				// Notify GUI to update model/animation controls
				if (this.app.gui) {
					this.app.gui.updateModelControls();
				}

				// Load saved segments for this model/animation
				if (this.segmentManager) {
					this.segmentManager.setContext(filename, 0);
					this.segmentManager.load();
				}

				// Notify everyone waiting on this load. Drained before dispatch so a
				// callback that starts another load registers for THAT load, not this one.
				const waiting = this._modelLoadedOnce;
				this._modelLoadedOnce = [];
				for (const cb of waiting) {
					try {
						cb();
					} catch (err) {
						console.error('WebGLView: model-loaded callback failed', err);
					}
				}
			},
			(progress) => {
				console.log('Loading model...', (progress.loaded / progress.total * 100) + '%');
			},
			(error) => {
				console.error('Error loading model:', error);
				// Fallback: create a simple cube if model fails to load
				const geometry = new THREE.BoxGeometry(50, 50, 50);
				const material = new THREE.MeshNormalMaterial();
				this.object3D = new THREE.Mesh(geometry, material);
				this.scene.add(this.object3D);
			}
		);
	}

	/**
	 * Play a specific animation by index
	 * @param {number} animationIndex - The index of the animation to play
	 * @param {boolean} autoPlay - Whether to start playing immediately (default: true)
	 * @param {number} weight - Visual blend weight of the action (default: 1). Pass 0 to
	 *   advance the animation (so phase timing + rep counting still fire) WITHOUT moving
	 *   the mesh — the hand holds its neutral pose. Used for stim-rest bouts.
	 */
	playAnimation(animationIndex, autoPlay = true, weight = 1) {
		if (!this.availableAnimations || animationIndex >= this.availableAnimations.length) {
			console.warn(`Animation index ${animationIndex} not found`);
			return;
		}

		// Stop current action if exists
		if (this.action) {
			this.action.stop();
		}

		const clip = this.availableAnimations[animationIndex];
		this.clipTracks = this.extractClipTracks(clip);
		this.keyframeTimes = this.extractKeyframeTimes(clip);
		this.logAnimationKeyInfo(clip, { maxTracksToLog: 12 });
		this.action = this.mixer.clipAction(clip);
		this.action.play();
		// Weight 0 lets the action advance (timing + 'finished' rep events still fire)
		// without influencing the pose, so the hand stays neutral (stim-rest bouts).
		this.action.setEffectiveWeight(weight);

		// Pause immediately if autoPlay is false
		if (!autoPlay) {
			this.action.paused = true;
		}

		// Store animation duration for timeline (fallback to last key time)
		const lastKeyTime = this.keyframeTimes?.length ? this.keyframeTimes[this.keyframeTimes.length - 1] : 0;
		this.animationDuration = (Number.isFinite(clip.duration) && clip.duration > 0) ? clip.duration : lastKeyTime;
		this.animationTime = 0;

		if (this.app.timeline) {
			this.app.timeline.setClipInfo({
				duration: this.animationDuration,
				keyframeTimes: this.keyframeTimes,
				tracks: this.clipTracks,
			});
		}

		console.log(`Playing animation [${animationIndex}]: ${clip.name}`);

		// Update segment manager context and load saved segments
		if (this.segmentManager) {
			const model = WebGLView.AVAILABLE_MODELS[this.currentModelIndex];
			this.segmentManager.setContext(model.file, animationIndex);
			this.segmentManager.load();
		}
	}

	extractKeyframeTimes(clip, { epsilon = 1e-4 } = {}) {
		if (!clip || !clip.tracks) return [];

		let times = [];
		for (const track of clip.tracks) {
			if (!track?.times?.length) continue;
			// track.times is already sorted.
			times.push(...track.times);
		}

		if (!times.length) return [];
		times.sort((a, b) => a - b);

		// Dedupe across tracks with epsilon.
		const unique = [times[0]];
		for (let i = 1; i < times.length; i++) {
			const t = times[i];
			if (Math.abs(t - unique[unique.length - 1]) > epsilon) unique.push(t);
		}
		return unique;
	}

	extractClipTracks(clip) {
		if (!clip || !clip.tracks) return [];
		return clip.tracks.map((track) => ({
			name: track?.name || '(unnamed track)',
			// Keep as typed array (array-like) to avoid large copies.
			times: track?.times || [],
		}));
	}

	logAnimationKeyInfo(clip, { maxTracksToLog = 10 } = {}) {
		if (!clip) return;

		const tracks = clip.tracks || [];
		const trackCount = tracks.length;
		let maxKeys = 0;
		let minKeys = Infinity;
		let uniformDtTrackCount = 0;
		let inferredFpsHistogram = new Map();

		const inferFpsFromTimes = (times) => {
			if (!times || times.length < 3) return null;
			let deltas = [];
			for (let i = 1; i < times.length; i++) {
				const dt = times[i] - times[i - 1];
				if (dt > 0) deltas.push(dt);
			}
			if (deltas.length === 0) return null;
			deltas.sort((a, b) => a - b);
			const medianDt = deltas[Math.floor(deltas.length / 2)];
			if (!isFinite(medianDt) || medianDt <= 0) return null;
			const fps = 1 / medianDt;
			// Snap to common rates if close.
			const common = [24, 25, 30, 50, 60, 120];
			let snapped = fps;
			for (const c of common) {
				if (Math.abs(fps - c) / c < 0.02) {
					snapped = c;
					break;
				}
			}
			return { fps, snapped, medianDt };
		};

		const isMostlyUniformDt = (times, medianDt) => {
			if (!times || times.length < 4 || !medianDt) return false;
			let ok = 0;
			let total = 0;
			for (let i = 1; i < times.length; i++) {
				const dt = times[i] - times[i - 1];
				if (dt <= 0) continue;
				total++;
				if (Math.abs(dt - medianDt) / medianDt < 0.02) ok++;
			}
			return total > 0 && ok / total > 0.9;
		};

		for (const track of tracks) {
			const keyCount = track.times?.length || 0;
			maxKeys = Math.max(maxKeys, keyCount);
			minKeys = Math.min(minKeys, keyCount);

			const info = inferFpsFromTimes(track.times);
			if (info && isMostlyUniformDt(track.times, info.medianDt)) {
				uniformDtTrackCount++;
				const bucket = info.snapped ? String(info.snapped) : info.fps.toFixed(1);
				inferredFpsHistogram.set(bucket, (inferredFpsHistogram.get(bucket) || 0) + 1);
			}
		}

		const uniformRatio = trackCount ? uniformDtTrackCount / trackCount : 0;
		const likelyBaked = maxKeys >= 20 && uniformRatio >= 0.7;

		console.groupCollapsed(`=== ANIMATION KEY INFO: ${clip.name || '(unnamed)'} ===`);
		console.log(`Duration: ${clip.duration.toFixed(3)}s`);
		console.log(`Tracks: ${trackCount}`);
		console.log(`Keys per track: min=${isFinite(minKeys) ? minKeys : 0}, max=${maxKeys}`);
		console.log(`Uniform Δt tracks: ${uniformDtTrackCount}/${trackCount} (${(uniformRatio * 100).toFixed(1)}%)`);
		if (inferredFpsHistogram.size) {
			console.log(`Inferred FPS (uniform tracks):`, Object.fromEntries(inferredFpsHistogram.entries()));
		}
		console.log(`Heuristic: ${likelyBaked ? 'LIKELY BAKED (sampled each frame)' : 'LIKELY SPARSE (original-ish keys)'} (heuristic)`);

		let logged = 0;
		for (const track of tracks) {
			if (logged >= maxTracksToLog) break;
			const times = track.times || [];
			const info = inferFpsFromTimes(times);
			console.log({
				track: track.name,
				keys: times.length,
				interpolation: track.getInterpolation?.(),
				firstTimes: times.slice(0, 5),
				lastTimes: times.slice(Math.max(0, times.length - 5)),
				inferredFps: info ? Number(info.fps.toFixed(2)) : null,
				snappedFps: info ? info.snapped : null,
			});
			logged++;
		}
		console.groupEnd();
	}

	// ---------------------------------------------------------------------------------------------
	// PUBLIC
	// ---------------------------------------------------------------------------------------------

	update() {
		if (this.controls) this.controls.update();

		// Update animations
		if (this.mixer && this.isPlaying) {
			const baseSpeed = this.playbackSpeed;
			// Apply segment-specific speed multiplier if segment manager exists
			const segmentMultiplier = this.segmentManager?.getSpeedAtTime(this.animationTime) ?? 1.0;
			const delta = this.clock.getDelta() * baseSpeed * segmentMultiplier;
			this.mixer.update(delta);

			// Update timeline position
			if (this.action) {
				this.animationTime = this.action.time;
			}
		}
	}

	/**
	 * Set the segment manager reference
	 * @param {SegmentManager} manager
	 */
	setSegmentManager(manager) {
		this.segmentManager = manager;
	}

	// Animation control methods
	play() {
		this.isPlaying = true;
		if (this.action) {
			this.action.paused = false;
		}
		this.clock.getDelta(); // Reset delta to avoid time jump
	}

	pause() {
		this.isPlaying = false;
		if (this.action) {
			this.action.paused = true;
		}
	}

	setAnimationTime(time) {
		if (this.action) {
			this.action.time = time;
			this.animationTime = time;
			if (!this.isPlaying) {
				// Force mixer update when paused to reflect the time change
				this.mixer.update(0);
			}
		}
	}

	draw() {
		this.renderer.render(this.scene, this.camera);
	}

	// ---------------------------------------------------------------------------------------------
	// EVENT HANDLERS
	// ---------------------------------------------------------------------------------------------

	resize(vw, vh) {
		if (!this.renderer) return;
		this.camera.aspect = vw / vh;
		this.camera.updateProjectionMatrix();

		// this.fovHeight = 2 * Math.tan((this.camera.fov * Math.PI) / 180 / 2) * this.camera.position.z;
		// this.fovWidth = this.fovHeight * this.camera.aspect;

		this.renderer.setSize(vw, vh);
	}
}
