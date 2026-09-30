# 3D Anatomical Organ Models — Attribution

The organ meshes in this directory (`VH_M_Heart.glb`, `VH_M_Lung.glb`,
`VH_M_Liver.glb`, `VH_M_Kidney_L.glb`, `VH_M_Kidney_R.glb`, `VH_M_Spleen.glb`,
`VH_M_Pancreas.glb`, `VH_M_Small_Intestine.glb`, `VH_M_Urinary_Bladder.glb`,
`VH_M_Spinal_Cord.glb`, `VH_M_Thymus.glb`, `VH_M_Blood_Vasculature.glb`)
are from the **HuBMAP Common Coordinate Framework (CCF) 3D Reference Object
Library**, v1.2.

- Source: https://github.com/hubmapconsortium/ccf-releases (v1.2/models)
- Project: Human Reference Atlas — https://humanatlas.io
- Derived from the Visible Human Project (U.S. National Library of Medicine).

**License: Creative Commons Attribution 4.0 International (CC BY 4.0)**
https://creativecommons.org/licenses/by/4.0/

**Changes made (2026-09-27).** The files are re-encoded, not re-shaped: vertex
positions and triangle indices are byte-identical to the originals. Three vertex
attributes the renderer never reads are removed (`COLOR_0`, `TANGENT`,
`TEXCOORD_0`; the files contain no textures, and AegisOS applies its own
holographic rendering materials at runtime), and normals are stored as 8-bit
normalised unit vectors under the `KHR_mesh_quantization` extension (at most
0.4 degrees from the original direction; normals that are degenerate in the
source stay degenerate). 24.3 MB -> 16.4 MB. The re-encoding is reproducible
and self-checking: `scripts/slim_glb.py`. Reuse requires attribution to the
HuBMAP CCF / Human Reference Atlas as above.
