/* ============================================================
   AEGIS-OS — surface materials for the anatomical twin

   Two surfaces carry this render: the organs and the shell around them.

   ORGANS. The old material was a MeshStandardMaterial with an emissive tint
   and a fresnel term, and it produced exactly what that describes — a lamp in
   the shape of a liver. Real tissue does three things that material could not:
   it takes a sharp specular highlight from a wet surface layer, it scatters
   light through itself so thin parts (a lung edge, a kidney's rim) glow when
   lit from behind, and it has a soft velvety falloff where the surface turns
   away. All three are here.

   The one thing deliberately NOT used is MeshPhysicalMaterial's `transmission`.
   It is the physically correct way to get light through a solid, and it forces
   three to render the whole scene into a transmission buffer once more per
   frame. This twin sits on a dashboard that is simultaneously running a
   detection loop; a second full scene render every frame is not a cost this
   surface gets to impose. The scatter below is an approximation and is
   commented as one.

   SHELL. Its job is to be a window, not a surface: it has to describe a body
   convincingly while hiding as little of the anatomy as possible. It is drawn
   as a thin fresnel-dominant film with a depth-cued interior haze, so the
   silhouette is crisp and the middle is glass.
   ============================================================ */
import * as THREE from "three";

/* ------------------------------------------------------------------ *
 * Organs — wet tissue
 * ------------------------------------------------------------------ */
/**
 * The clinical tier colour is read from the material's own `emissive`, which
 * three already multiplies by `emissiveIntensity` before uploading. That is
 * deliberate: the twin's animation code drives `emissive` and
 * `emissiveIntensity` in twenty places — severity, infection, treatment
 * response, the heartbeat — and a material that invented its own tint uniform
 * would have needed every one of those rewritten, with the old ones left
 * silently doing nothing to the picture.
 *
 * @returns {THREE.MeshPhysicalMaterial}
 */
export function makeOrganMaterial() {
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x23272d,
    emissive: new THREE.Color(0x848a93),
    emissiveIntensity: 0.5,
    roughness: 0.34,          // wet, but not a mirror
    metalness: 0.0,
    clearcoat: 0.85,          // the serous film every abdominal organ has
    clearcoatRoughness: 0.18,
    sheen: 0.5,               // velvety falloff at grazing angles
    sheenRoughness: 0.72,
    sheenColor: new THREE.Color(0x9a5c48),
    envMapIntensity: 1.75,    // the studio env is what makes it read as wet
    transparent: true,
    opacity: 0.95,
    side: THREE.DoubleSide,
    depthWrite: true,
  });

  const u = {
    uScatter: { value: 1.0 },      // 0 disables the through-light term
    uSSSDir: { value: new THREE.Vector3(0.6, 1.4, 1.2).normalize() },
    uRim: { value: 1.0 },
    uPulse: { value: 0.0 },
  };

  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", `#include <common>
        varying vec3 vWorldNrm; varying vec3 vWorldPos;`)
      .replace("#include <worldpos_vertex>", `#include <worldpos_vertex>
        vWorldNrm = normalize(mat3(modelMatrix) * objectNormal);
        vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;`);

    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
        uniform vec3 uSSSDir;
        uniform float uScatter, uRim, uPulse;
        varying vec3 vWorldNrm; varying vec3 vWorldPos;`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
        // The emissive uniform arrives pre-multiplied by emissiveIntensity, so
        // severity, infection and treatment response reach these unchanged.
        vec3 tint = emissive;
        vec3 N = normalize(vWorldNrm);
        vec3 V = normalize(cameraPosition - vWorldPos);
        vec3 L = normalize(uSSSDir);

        // --- back-scatter translucency -------------------------------------
        // The Dice/Frostbite approximation: light leaving the BACK of a thin
        // volume toward the eye. The normal is folded into the light vector so
        // the term follows surface curvature instead of producing a flat halo,
        // and the exponent keeps it tight enough to read as thickness rather
        // than as a second rim light.
        vec3 backLight = normalize(-L + N * 0.32);
        float back = pow(clamp(dot(V, backLight), 0.0, 1.0), 5.0);
        // Thin parts scatter more. With no thickness map, facing ratio is the
        // available proxy: on a closed mesh, a silhouette edge IS the thin part.
        float thin = pow(1.0 - abs(dot(N, V)), 1.6);
        vec3 scatter = tint * back * thin * 1.9 * uScatter;

        // --- wrapped diffuse ------------------------------------------------
        // Tissue does not fall to black at the terminator; light that entered
        // nearby comes back out. Wrapping N.L is the cheapest honest way to say
        // so, and it stops an unlit organ becoming an organ-shaped hole.
        float wrap = clamp((dot(N, L) + 0.55) / 1.55, 0.0, 1.0);
        vec3 wrapped = tint * wrap * 0.26;

        // --- fresnel rim ----------------------------------------------------
        // Carries the tier colour to the edge, which is the part of an organ
        // that survives being seen through a translucent shell.
        float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.2);
        vec3 rim = tint * fres * 1.15 * uRim;

        gl_FragColor.rgb += scatter + wrapped + rim;
        // The heartbeat modulates brightness, never hue: a beating organ must
        // not appear to change clinical state as it beats.
        gl_FragColor.rgb *= (1.0 + 0.10 * uPulse);
        gl_FragColor.a = clamp(gl_FragColor.a + fres * 0.30, 0.0, 1.0);`);
  };

  m.userData.aegis = u;
  m.customProgramCacheKey = () => "aegis-organ-v2";
  return m;
}


/* ------------------------------------------------------------------ *
 * Shell — a window in the shape of a body
 * ------------------------------------------------------------------ */
/**
 * @param {object} zones height-banded colours: head/chest/abd/pelvis/limb/rim
 */
export function makeShellMaterial(zones, bodyHeight) {
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x2b3138,
    roughness: 0.52,
    metalness: 0.0,
    // The environment response was cut to almost nothing while the shell was a
    // segmented robot, because lighting it only made a white mannequin brighter.
    // With a continuous body carrying real surface anatomy, the environment is
    // what reveals that anatomy — so it comes back up, and the overall level is
    // held down by uGain instead, which does not flatten the form to do it.
    clearcoat: 0.35,
    clearcoatRoughness: 0.42,
    envMapIntensity: 0.55,
    transparent: true,
    opacity: 0.16,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const u = {
    uHead: { value: zones.head }, uChest: { value: zones.chest },
    uAbd: { value: zones.abd }, uPelvis: { value: zones.pelvis },
    uLimb: { value: zones.limb }, uRim: { value: zones.rim },
    uYmax: { value: bodyHeight }, uTime: { value: 0 },
    uScan: { value: -1 }, uPulse: { value: 0 },
    // One gain over the whole shell. It exists because the shell is drawn
    // DOUBLE-SIDED with depth writes off, so a limb stacks a front face, a back
    // face and any overlapping segment into the same pixel — and the brightness
    // you get is not the brightness any single term predicts. Measured on a
    // thigh, where no organ sits behind the surface to confuse the reading, the
    // shell alone was coming back at 173/255: a solid light-grey mannequin
    // instead of glass. A single knob makes that measurable and correctable
    // without re-deriving five coefficients that each looked reasonable alone.
    // 0.45 was found by sweeping the gain and reading that same thigh pixel:
    //   gain 1.00 -> 110    0.70 -> 77    0.50 -> 51    0.45 -> ~40    0.35 -> 26
    // Around 40 the body reads as dark glass with a bright silhouette, and the
    // organ atlas is unambiguously the brightest thing in frame — which is the
    // correct hierarchy, since the organs are the only part carrying a reading.
    // Swept against the render with the anatomy in frame: 0.68 washed the body
    // to a uniform pale shape; 0.34 lost it entirely and left the head floating.
    uGain: { value: 0.50 },
  };

  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", `#include <common>
        varying float vWY; varying float vWX;
        varying vec3 vWNrm; varying vec3 vWPos;`)
      .replace("#include <worldpos_vertex>", `#include <worldpos_vertex>
        vec4 _wp = modelMatrix * vec4(transformed, 1.0);
        vWY = _wp.y; vWX = abs(_wp.x); vWPos = _wp.xyz;
        vWNrm = normalize(mat3(modelMatrix) * objectNormal);`);

    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
        varying float vWY; varying float vWX;
        varying vec3 vWNrm; varying vec3 vWPos;
        uniform vec3 uHead,uChest,uAbd,uPelvis,uLimb,uRim;
        uniform float uYmax,uTime,uScan,uPulse,uGain;`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
        float h = clamp(vWY / uYmax, 0.0, 1.0);
        vec3 col = uLimb;
        col = mix(col, uPelvis, smoothstep(0.44, 0.50, h));
        col = mix(col, uAbd,    smoothstep(0.50, 0.60, h));
        col = mix(col, uChest,  smoothstep(0.60, 0.74, h));
        col = mix(col, uHead,   smoothstep(0.80, 0.88, h));
        float arm = smoothstep(0.17, 0.27, vWX) * smoothstep(0.55, 0.62, h)
                    * (1.0 - smoothstep(0.80, 0.86, h));
        col = mix(col, uLimb, arm);

        vec3 N = normalize(vWNrm);
        vec3 V = normalize(cameraPosition - vWPos);
        float ndv = clamp(dot(N, V), 0.0, 1.0);

        // A tight fresnel. A wide one spreads across curved surfaces, and since
        // the head is the roundest part of a body it ends up the brightest
        // thing on screen — while carrying no clinical signal at all.
        float fres = pow(1.0 - ndv, 2.6);
        // Tightening the exponent was not enough on its own. A head is close to
        // a sphere, so it grazes the viewer nearly everywhere no matter how
        // steep the falloff, and it stayed the brightest object in frame. The
        // clinical content of this body is the torso; the head is damped
        // explicitly rather than hoping a curve does it. Neurological state,
        // when it IS monitored, arrives through the brain organ inside — which
        // is unaffected by this.
        // Damping only the FRESNEL was not enough. A head is a smooth sphere
        // pointed at the key light, so most of its brightness arrives through
        // the diffuse term, and at low shell gain it stayed lit while the body
        // faded — a bright ball floating above a dim torso. The factor applies
        // to the whole surface response at head height instead.
        // 0.26, not 0.42. The head is the largest smooth convex surface on the
        // body and it is drawn double-sided, so front and back both contribute;
        // even damped it was the brightest thing in frame. It also carries the
        // NEUROLOGICAL system, which on this cohort is unmonitored — and the
        // rule everywhere else on this body is that a system with no data must
        // never look like a confident reading.
        float headDamp = mix(1.0, 0.26, smoothstep(0.845, 0.915, h));
        fres *= headDamp;

        // Interior haze cued to how much shell the eye is looking THROUGH.
        // Facing surfaces are near-invisible, grazing ones thicken. This is
        // what makes the body read as a volume while staying a window: the
        // anatomy behind it is never more than lightly veiled.
        float thickness = pow(1.0 - ndv, 1.25);
        vec3 haze = col * thickness * 0.055;

        // NO SCANLINES ON THE BODY. This was
        //   0.88 + 0.12 * sin(vWY * 150.0 - uTime * 2.2)
        // — 150 cycles per metre, so about 255 stripes down a 1.7 m figure. At
        // any realistic panel size that is not a scanline, it is corduroy: the
        // whole body reads as ribbed fabric and the surface anatomy underneath
        // it is completely lost. The grade pass already lays a single, subtle
        // screen-space scanline over the finished frame, which is where a
        // display artefact belongs — on the display, not painted onto the
        // patient.
        float sl = 1.0;

        // The diagnostic sweep. Its width is load-bearing: wide enough and the
        // Gaussian covers the whole torso for much of every cycle, and its
        // alpha term renders the chest an opaque slab — erasing the heart,
        // lungs and liver, which are the only things on this body that mean
        // anything. It brightens what it crosses; it never closes the window.
        float sweep     = exp(-pow((h - uScan) * 26.0, 2.0));
        float sweepGlow = exp(-pow((h - uScan) *  7.0, 2.0));

        // THE LIT RESULT IS USED, NOT DISCARDED — and this is the whole change.
        //
        // Everything above this line was computed from height and fresnel alone:
        // no diffuse, no specular, no environment. On the old body that was
        // survivable, because it was a robot made of separate armour plates and
        // every plate edge produced a fresnel line for the eye to read. On a
        // single continuous body the middle of the torso faces the camera almost
        // everywhere, fresnel goes to zero across all of it, and the figure
        // renders as a flat grey cutout — with the clavicles, sternum, costal
        // margin and iliac crests carved into the mesh and completely invisible.
        //
        // At this point gl_FragColor.rgb holds the real PBR shading. Mixing it
        // in is what lets curvature read, and it is the only thing that makes
        // surface anatomy worth carving.
        vec3 lit = gl_FragColor.rgb;
        float litLum = dot(lit, vec3(0.299, 0.587, 0.114));

        vec3 outc = (lit * 2.9 * headDamp) + col * (0.06 + 0.22 * fres)
                  + uRim * fres * 0.46 + haze;
        outc *= sl;
        outc += col * 0.06;
        // The sweep uses the interface accent, not an ad-hoc cyan: a sweep is
        // an affordance, and the OS reserves one hue for those.
        outc += col * sweep * 0.50 + vec3(0.48, 0.64, 0.97) * sweep * 0.58;
        outc += col * sweepGlow * 0.09;
        outc *= (1.0 + 0.07 * uPulse);

        // Alpha follows the shading too. Without the luminance term a lit fold
        // would brighten but stay as transparent as the flat area beside it, so
        // the form would be there and still not read.
        // The shell is drawn double-sided with depth writes off, so a torso
        // stacks a front face, a back face and any overlapping limb into the
        // same pixel. Alpha therefore accumulates faster than any single term
        // suggests, and a coefficient that looks right at full-body framing
        // turns the body to milk as soon as anyone zooms in. Held low enough
        // that three or four stacked layers still read as glass.
        float a = clamp(0.040 + fres * 0.26 + litLum * 0.80 * headDamp
                        + sweep * 0.08 + sweepGlow * 0.025, 0.0, 0.58);
        gl_FragColor = vec4(outc * uGain, a * uGain);`);
  };

  m.userData.aegis = u;
  m.customProgramCacheKey = () => "aegis-shell-v2";
  return m;
}
