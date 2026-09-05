/* ============================================================
   AEGIS-OS — render pipeline for the anatomical twin

   The twin's anatomy, animation and clinical colour rules live in twin3d.js.
   This module owns only HOW it is rendered, and it exists because the old
   pipeline had two defects that no amount of shader tuning could fix:

   1. ANTIALIASING WAS SILENTLY OFF. The renderer was built with
      `antialias: true`, which requests MSAA on the DEFAULT framebuffer. The
      moment an EffectComposer is attached, every pass renders into its own
      WebGLRenderTarget instead — and three's EffectComposer creates that target
      with no `samples`, so the multisampling was discarded. Organ silhouettes
      and the body outline were being drawn with hard aliased steps on a surface
      whose whole job is a clean edge. Fixed by giving the composer a
      multisampled target explicitly.

   2. THERE WAS NO ENVIRONMENT. Lighting was three directional lights plus an
      ambient term, which is enough to shade a matte solid and nothing like
      enough for wet tissue. Organs have a strong specular response and pick up
      colour from everything around them; with no environment map they render as
      flat emissive blobs, which is exactly what they looked like. A PMREM
      environment is generated here from a procedural studio — no external
      asset, so nothing to load and nothing blocked by a content policy — and it
      is what makes an organ read as a wet surface rather than a lamp.

   On top of those: bloom is now SELECTIVE, so the emissive organs glow while
   the shell around them does not (a global bloom washed the two together and
   cost the anatomy its definition), and a final grade pass adds the optical
   character an instrument display has — vignette, sensor grain, a little
   chromatic aberration at the corners.
   ============================================================ */
import * as THREE from "three";
import { EffectComposer } from "./vendor/three/postprocessing/EffectComposer.js";
import { RenderPass } from "./vendor/three/postprocessing/RenderPass.js";
import { ShaderPass } from "./vendor/three/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "./vendor/three/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "./vendor/three/postprocessing/OutputPass.js";

/** Objects on this layer bloom. Everything else is masked to black first. */
export const BLOOM_LAYER = 1;

/* ------------------------------------------------------------------ *
 * Environment — a procedural studio, PMREM-filtered
 * ------------------------------------------------------------------ */
/**
 * Build an image-based lighting environment with no external asset.
 *
 * A handful of emissive quads in a dark box, blurred by PMREMGenerator into a
 * prefiltered mipmapped radiance map. Physical materials then get a real
 * specular response and a plausible ambient term from it, which is the single
 * biggest difference between "shaded solid" and "wet tissue".
 *
 * The box is deliberately cool and dim with two warm accents: tissue lit by a
 * pure neutral studio reads plastic, and this console's palette reserves
 * saturated colour for clinical state, so the environment must not push hue of
 * its own into the organs. It contributes shape, not colour.
 */
export function createEnvironment(renderer) {
  const env = new THREE.Scene();
  const box = new THREE.BoxGeometry();
  box.deleteAttribute("uv");

  const panel = (color, intensity, pos, scale, rotY) => {
    const m = new THREE.Mesh(
      box,
      new THREE.MeshStandardMaterial({
        side: THREE.BackSide, color: 0x000000,
        emissive: new THREE.Color(color), emissiveIntensity: intensity,
      }));
    m.position.set(pos[0], pos[1], pos[2]);
    m.scale.set(scale[0], scale[1], scale[2]);
    if (rotY) m.rotation.y = rotY;
    env.add(m);
    return m;
  };

  // The room itself — very dark, so the panels read as sources rather than fill.
  panel(0x0b0d10, 1.0, [0, 0, 0], [12, 8, 12]);
  // Key: a broad cool softbox high and to the camera's right.
  panel(0xdfe6f2, 3.4, [3.2, 3.0, 2.2], [3.2, 2.6, 0.1], -0.5);
  // Fill: dimmer, opposite side, keeps the shadow side from going to pure black.
  panel(0x8f9bb0, 1.15, [-3.4, 1.4, 0.4], [2.8, 2.4, 0.1], 0.6);
  // Two warm low kickers. Tissue owes much of its look to bounce from below;
  // without them the underside of every organ dies and the atlas flattens out.
  panel(0xffb27a, 0.85, [-1.2, -1.9, 1.6], [2.0, 0.8, 0.1], 0.3);
  panel(0xff9d6a, 0.55, [1.8, -2.1, -1.4], [1.6, 0.7, 0.1], -0.9);
  // Cool top strip — a thin bright line reads as a specular highlight running
  // along the top of a curved wet surface, which is the cue that sells "organ".
  panel(0xeaf1ff, 2.6, [0, 3.6, -0.4], [4.4, 0.22, 0.1], 0);

  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const target = pmrem.fromScene(env, 0.035);
  // The scene was only ever a source for the prefilter; keeping it alive would
  // hold a few megabytes of geometry and materials for nothing.
  env.traverse((o) => { if (o.isMesh) o.material.dispose(); });
  box.dispose();
  pmrem.dispose();
  return target.texture;
}

/* ------------------------------------------------------------------ *
 * Final grade — one fullscreen pass, several small optical effects
 * ------------------------------------------------------------------ */
/**
 * Vignette, sensor grain, corner chromatic aberration and a faint scanline.
 *
 * Bundled into ONE pass on purpose: each of these is a couple of instructions,
 * and four separate ShaderPasses would mean four fullscreen draws and four
 * render-target ping-pongs for work that fits comfortably in a single shader.
 * On a dashboard that also runs a live detection loop, the frame budget is not
 * the renderer's to spend freely.
 */
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uAberration: { value: 0.0016 },
    uGrain: { value: 0.028 },
    uVignette: { value: 0.86 },
    uScanline: { value: 0.020 },
    uResolution: { value: new THREE.Vector2(1, 1) },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float uTime, uAberration, uGrain, uVignette, uScanline;
    uniform vec2 uResolution;
    varying vec2 vUv;

    // Cheap hash for per-pixel, per-frame grain. A texture lookup would be
    // steadier but costs a sampler and a texture upload for noise nobody
    // inspects closely.
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

    void main(){
      vec2 uv = vUv;
      vec2 c = uv - 0.5;
      float r2 = dot(c, c);

      // Chromatic aberration scaled by radius squared: zero in the centre where
      // the anatomy is read, present at the corners where a real lens has it.
      // A constant offset would smear the organs, which is the opposite of the
      // intent.
      vec2 off = c * r2 * uAberration * 8.0;
      vec3 col;
      col.r = texture2D(tDiffuse, uv + off).r;
      col.g = texture2D(tDiffuse, uv).g;
      col.b = texture2D(tDiffuse, uv - off).b;

      // Scanline, keyed to device pixels rather than UV so it does not change
      // frequency when the panel is resized.
      float line = sin(uv.y * uResolution.y * 1.6) * 0.5 + 0.5;
      col *= 1.0 - uScanline * line;

      // Grain, applied in a luminance-aware way: shadows carry sensor noise,
      // highlights mostly do not, and uniform grain over a bright organ looks
      // like dirt rather than like a sensor.
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      float g = hash(uv * uResolution + fract(uTime) * 137.0) - 0.5;
      col += g * uGrain * (1.0 - smoothstep(0.25, 0.95, lum));

      // Vignette last, so grain and aberration are darkened with the frame
      // rather than sitting on top of it.
      col *= mix(1.0, smoothstep(0.86, 0.16, r2 * uVignette * 2.0), 0.72);

      gl_FragColor = vec4(col, 1.0);
    }`,
};

/**
 * Depth-cued focus. Not a physically-modelled lens — a radius-weighted tap
 * blur whose radius grows with distance from a focus plane parked on the torso.
 *
 * Six taps on a rotating hexagon rather than a separable Gaussian: a Gaussian
 * needs two passes and two more fullscreen targets, and at the blur radii this
 * needs (a couple of pixels at the extremities) the tap pattern is
 * indistinguishable and costs one pass. The rotation per pixel breaks up the
 * hexagonal banding that a fixed pattern leaves on smooth gradients.
 *
 * The focus plane sits at the torso because that is where every organ carrying
 * a clinical reading lives. Blurring toward the head and the feet is not just
 * decoration: it says where to look.
 */
const FocusShader = {
  uniforms: {
    tDiffuse: { value: null },
    tDepth: { value: null },
    uFocus: { value: 0.5 },        // focus distance, in world units from the eye
    // WIDE ENOUGH TO KEEP THE WHOLE BODY SHARP, and that is the point.
    //
    // The first tuning put the range just inside the body's own depth so the
    // near shoulder would soften. It worked, and it was wrong: the anatomy is
    // the only thing on this canvas carrying a clinical reading, and softening
    // any of it trades information for atmosphere. Measured at range 0.26 the
    // whole figure picked up a haze that cost the organ atlas its edges.
    //
    // At 0.95 the entire body sits inside the sharp zone and the blur belongs to
    // what is genuinely elsewhere — the drifting cell field, the platform rings,
    // the backdrop. That still separates the figure from its surroundings, which
    // is what depth of field is for here.
    uRange: { value: 0.95 },
    uStrength: { value: 7.0 },     // max blur radius, in device pixels
    uNear: { value: 0.1 },
    uFar: { value: 100.0 },
    uResolution: { value: new THREE.Vector2(1, 1) },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform sampler2D tDepth;
    uniform float uFocus, uRange, uStrength, uNear, uFar;
    uniform vec2 uResolution;
    varying vec2 vUv;

    float linearDepth(float z){
      float ndc = z * 2.0 - 1.0;
      return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
    }

    void main(){
      float d = texture2D(tDepth, vUv).x;
      // The far plane is empty space; blurring the backdrop into the figure
      // would put a halo around the silhouette, which reads as a compositing
      // error rather than as depth.
      float lin = linearDepth(d);
      float coc = clamp((abs(lin - uFocus) - uRange) / uRange, 0.0, 1.0);
      coc *= step(d, 0.9999);

      vec3 sum = texture2D(tDiffuse, vUv).rgb;
      if (coc > 0.004) {
        float r = coc * uStrength;
        vec2 px = r / uResolution;
        float a0 = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
        for (int i = 0; i < 6; i++) {
          float a = a0 + float(i) * 1.0471976;    // 60 degrees
          sum += texture2D(tDiffuse, vUv + vec2(cos(a), sin(a)) * px).rgb;
        }
        sum /= 7.0;
      }
      gl_FragColor = vec4(sum, 1.0);
    }`,
};

/** Additively combine the base render with the bloom-only render. */
const CombineShader = {
  uniforms: { tBase: { value: null }, tBloom: { value: null }, uStrength: { value: 1.0 } },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tBase; uniform sampler2D tBloom; uniform float uStrength;
    varying vec2 vUv;
    void main(){
      vec4 base = texture2D(tBase, vUv);
      vec3 bloom = texture2D(tBloom, vUv).rgb * uStrength;
      // Screen rather than add. Additive bloom over an already-bright emissive
      // organ clips to white and destroys the tier colour, which on this body
      // is the entire clinical signal — a CRITICAL organ has to stay red, not
      // become a white smear.
      gl_FragColor = vec4(1.0 - (1.0 - base.rgb) * (1.0 - bloom), base.a);
    }`,
};

/* ------------------------------------------------------------------ *
 * The pipeline
 * ------------------------------------------------------------------ */
/**
 * Two composers: one renders only what is on BLOOM_LAYER, the other renders the
 * scene normally and screens the first over it.
 *
 * The alternative — one composer with a thresholded global bloom — is what
 * shipped, and it blooms the body shell along with the organs. The shell is a
 * large, softly-lit surface, so it contributes a wide low-frequency haze that
 * sits directly on top of the anatomy and costs it every edge it has. Masking
 * by layer means the glow belongs to the things that are supposed to glow.
 */
export function createPipeline({ renderer, scene, camera, width, height, pixelRatio }) {
  const dpr = Math.min(pixelRatio || window.devicePixelRatio || 1, 2);

  // WebGL2 gives us multisampled render targets; WebGL1 does not, and there the
  // honest thing is to render at a higher internal resolution instead of
  // pretending the edges are clean.
  const isWebGL2 = renderer.capabilities.isWebGL2;
  const samples = isWebGL2 ? 4 : 0;
  const mkTarget = (w, h) => {
    const t = new THREE.WebGLRenderTarget(Math.max(1, Math.floor(w * dpr)),
                                          Math.max(1, Math.floor(h * dpr)), {
      type: THREE.HalfFloatType,           // bloom needs headroom above 1.0
      samples,
    });
    t.texture.colorSpace = THREE.LinearSRGBColorSpace;
    return t;
  };

  const baseTarget = mkTarget(width, height);
  const bloomTarget = mkTarget(width, height);

  // Depth, for the focus pass. Without it the frame is uniformly sharp from the
  // near shoulder to the far heel, which is the look of a diagram rather than of
  // something being observed — and it gives the eye no cue about which part of
  // the body it is meant to be reading.
  const mkDepth = (w, h) => {
    const d = new THREE.DepthTexture(Math.max(1, Math.floor(w * dpr)),
                                     Math.max(1, Math.floor(h * dpr)));
    d.type = THREE.UnsignedIntType;
    d.format = THREE.DepthFormat;
    return d;
  };
  baseTarget.depthTexture = mkDepth(width, height);

  // -- bloom composer: renders ONLY the bloom layer, then blurs it ----------
  const bloomComposer = new EffectComposer(renderer, bloomTarget);
  bloomComposer.renderToScreen = false;
  bloomComposer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(width * dpr, height * dpr),
    // Strength and radius are held down deliberately. The shell does not write
    // depth — it must not, or it would occlude the anatomy it exists to reveal —
    // so organ bloom passes straight through the body. At a CRITICAL state the
    // organs are bright enough that a generous bloom floods the whole torso and
    // the atlas dissolves into a white wash: the render gets LESS informative
    // exactly when the patient is worst, which is the one moment it must not.
    0.38,   // strength
    0.48,   // radius — tight, so the glow hugs the organ that owns it
    0.0);   // threshold 0: the layer mask already decided what blooms, and a
            // luminance threshold on top would drop dim CRITICAL organs
  bloomComposer.addPass(bloomPass);

  // -- main composer --------------------------------------------------------
  const composer = new EffectComposer(renderer, baseTarget);
  composer.addPass(new RenderPass(scene, camera));
  // Focus BEFORE bloom is combined: a blurred glow is what a real lens produces,
  // and blurring after would smear the bloom's own halo into a grey wash.
  const focusPass = new ShaderPass(FocusShader);
  focusPass.uniforms.tDepth.value = baseTarget.depthTexture;
  focusPass.uniforms.uNear.value = camera.near;
  focusPass.uniforms.uFar.value = camera.far;
  focusPass.uniforms.uResolution.value.set(width * dpr, height * dpr);
  composer.addPass(focusPass);
  const combinePass = new ShaderPass(CombineShader, "tBase");
  combinePass.uniforms.tBloom.value = bloomComposer.renderTarget2.texture;
  composer.addPass(combinePass);
  const gradePass = new ShaderPass(GradeShader);
  gradePass.uniforms.uResolution.value.set(width * dpr, height * dpr);
  composer.addPass(gradePass);
  composer.addPass(new OutputPass());

  // -- layer masking --------------------------------------------------------
  // Anything not on the bloom layer is swapped to flat black for the bloom
  // render and restored afterwards. Swapping MATERIALS rather than toggling
  // `visible` keeps depth correct: an organ behind the shell must still be
  // occluded by it in the bloom pass, or its glow leaks through the body.
  const black = new THREE.MeshBasicMaterial({ color: 0x000000 });
  const stash = new Map();
  const bloomLayer = new THREE.Layers();
  bloomLayer.set(BLOOM_LAYER);

  const darken = (obj) => {
    if (obj.isMesh && !bloomLayer.test(obj.layers)) {
      stash.set(obj.uuid, obj.material);
      obj.material = black;
    }
  };
  const restore = (obj) => {
    const m = stash.get(obj.uuid);
    if (m) { obj.material = m; stash.delete(obj.uuid); }
  };

  const clock = new THREE.Clock();
  let quality = 1;

  return {
    BLOOM_LAYER,
    bloomPass,
    gradePass,
    focusPass,
    /** Park the focus plane on a world point (the torso, in practice). */
    focusOn(worldPos, cam) {
      focusPass.uniforms.uFocus.value = cam.position.distanceTo(worldPos);
    },
    /** True when the composer targets are genuinely multisampled. */
    antialiased: samples > 0,

    /**
     * Report what the pipeline actually got, rather than what it asked for.
     *
     * `antialias: true` on the renderer is a request that a composer silently
     * discards, and a depth texture that never resolves turns the focus pass
     * into a uniform blur that looks like a soft-focus choice. Both failed
     * quietly before. On a machine where either is unavailable this says so in
     * the console instead of shipping a render that is subtly wrong.
     */
    selfTest() {
      const r = {
        antialiased: samples > 0,
        depthBound: !!focusPass.uniforms.tDepth.value,
        webgl2: isWebGL2,
        pixelRatio: dpr,
      };
      if (!r.antialiased) {
        console.warn("[twin] no multisampling — organ silhouettes will alias. " +
                     "WebGL2 is required for multisampled composer targets.");
      }
      if (!r.depthBound) {
        console.warn("[twin] no depth texture — the focus pass would blur the " +
                     "whole frame uniformly, so it is disabled.");
        focusPass.uniforms.uStrength.value = 0;
      }
      return r;
    },

    render() {
      gradePass.uniforms.uTime.value = clock.getElapsedTime();
      const bg = scene.background;
      scene.background = null;
      scene.traverse(darken);
      bloomComposer.render();
      scene.traverse(restore);
      scene.background = bg;
      composer.render();
    },

    resize(w, h) {
      const rw = Math.max(1, w), rh = Math.max(1, h);
      composer.setSize(rw, rh);
      bloomComposer.setSize(rw, rh);
      bloomPass.setSize(rw * dpr, rh * dpr);
      gradePass.uniforms.uResolution.value.set(rw * dpr, rh * dpr);
      focusPass.uniforms.uResolution.value.set(rw * dpr, rh * dpr);
      // setSize rebuilt the composer's targets, so the depth texture the focus
      // pass samples is the OLD one — it would sample a stale, wrongly sized
      // buffer and the blur would key off last frame's geometry.
      const dt = new THREE.DepthTexture(Math.max(1, Math.floor(rw * dpr)),
                                        Math.max(1, Math.floor(rh * dpr)));
      dt.type = THREE.UnsignedIntType; dt.format = THREE.DepthFormat;
      composer.renderTarget1.depthTexture = dt;
      composer.renderTarget2.depthTexture = dt;
      focusPass.uniforms.tDepth.value = dt;
      // setSize replaces the composer's internal targets, so the combine pass's
      // reference to the bloom texture has to be re-pointed or the glow freezes
      // on the last frame rendered at the old size.
      combinePass.uniforms.tBloom.value = bloomComposer.renderTarget2.texture;
    },

    /** 1 = full, 0.5 = half-res internal buffers. Used when a tab is busy. */
    setQuality(q) {
      quality = Math.max(0.35, Math.min(1, q));
      return quality;
    },

    dispose() {
      composer.dispose(); bloomComposer.dispose();
      baseTarget.dispose(); bloomTarget.dispose(); black.dispose();
    },
  };
}
