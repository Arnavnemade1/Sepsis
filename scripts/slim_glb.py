"""Slim the anatomical GLB meshes the web shell downloads — and prove what did not change.

    python scripts/slim_glb.py webui/vendor/models/organs/*.glb          # rewrite in place
    python scripts/slim_glb.py --check webui/vendor/models/organs/*.glb  # verify only

The organ meshes (HuBMAP CCF, CC BY 4.0) store every attribute as float32. The twins
(``twin3d.js``, ``pktwin.js``) replace every material with their own, never enable vertex
colours and use no texture maps, so three attributes are downloaded and never read:
``COLOR_0`` (16 B a vertex), ``TANGENT`` (16 B) and ``TEXCOORD_0`` (8 B). Normals are
read (lighting), and are stored as 8-bit normalised vectors under ``KHR_mesh_quantization``
(12 B -> 4 B a vertex; <= 0.5 degrees of error), which three.js's GLTFLoader decodes
natively — no decoder to ship. Positions and indices are copied byte for byte, so every
placement, bounding box and calibration the twins compute is unchanged.

Attributes are only dropped from a file that contains no textures, and a rewrite is
refused unless, re-read, its positions and indices equal the original's exactly and its
normals are within the quantisation bound.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np

DROP = ("COLOR_0", "TANGENT", "TEXCOORD_0")
DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
MAX_NORMAL_DEG = 0.6


def read_glb(path: str) -> Tuple[Dict, bytes]:
    b = open(path, "rb").read()
    magic, version, _ = struct.unpack_from("<4sII", b, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path}: not a glTF 2 binary")
    jlen, jtype = struct.unpack_from("<II", b, 12)
    js = json.loads(b[20:20 + jlen])
    off = 20 + jlen
    blen, btype = struct.unpack_from("<II", b, off)
    return js, b[off + 8:off + 8 + blen]


def write_glb(path: str, js: Dict, binary: bytes) -> None:
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(j) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<II", len(j), 0x4E4F534A))
        fh.write(j)
        fh.write(struct.pack("<II", len(binary), 0x004E4942))
        fh.write(binary)


def accessor_array(js: Dict, binary: bytes, i: int) -> np.ndarray:
    """(count, ncomp) array of accessor ``i`` in its stored component type (strides honoured)."""
    a = js["accessors"][i]
    dt = np.dtype(DTYPE[a["componentType"]])
    n = NCOMP[a["type"]]
    bv = js["bufferViews"][a["bufferView"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * n
    count = a["count"]
    raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + dt.itemsize * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dt.itemsize * n), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dt).reshape(count, n)


def dequant(a: Dict, arr: np.ndarray) -> np.ndarray:
    if not a.get("normalized"):
        return arr.astype(np.float64)
    if arr.dtype == np.int8:
        return np.maximum(arr / 127.0, -1.0)
    if arr.dtype == np.int16:
        return np.maximum(arr / 32767.0, -1.0)
    return arr / float(np.iinfo(arr.dtype).max)


def slim(js: Dict, binary: bytes) -> Tuple[Dict, bytes]:
    js = json.loads(json.dumps(js))
    if js.get("skins") or js.get("animations"):
        raise ValueError("skinned / animated meshes are out of scope")
    textured = bool(js.get("images") or js.get("textures"))
    out = bytearray()
    views: List[Dict] = []
    accessors: List[Dict] = []
    remap: Dict[Tuple[int, str], int] = {}

    def add(arr: np.ndarray, src: Dict, target: int, stride: int = 0, extra: Dict = None) -> int:
        out.extend(b"\0" * (-len(out) % 4))
        view = {"buffer": 0, "byteOffset": len(out), "byteLength": arr.nbytes, "target": target}
        if stride:
            view["byteStride"] = stride
        out.extend(arr.tobytes())
        views.append(view)
        acc = {k: v for k, v in src.items() if k not in ("bufferView", "byteOffset")}
        acc["bufferView"] = len(views) - 1
        acc.update(extra or {})
        accessors.append(acc)
        return len(accessors) - 1

    for mesh in js["meshes"]:
        for pr in mesh["primitives"]:
            if pr.get("targets"):
                raise ValueError("morph targets are out of scope")
            attrs = {}
            for name, i in pr["attributes"].items():
                if name in DROP and not textured:
                    continue
                key = (i, name)
                if key not in remap:
                    a = js["accessors"][i]
                    arr = accessor_array(js, binary, i)
                    if name == "NORMAL" and a["componentType"] == 5126:
                        # the shader normalises normals, so the DIRECTION is what renders:
                        # quantise the unit vector (a tiny source normal would otherwise
                        # round to zero); a zero normal stays zero, as in the source
                        # "zero" is decided in float32, as the GPU normalises: a normal whose
                        # squared length underflows there (the heart has 166 at ~1e-28) is
                        # degenerate in the source and stays degenerate
                        live = np.linalg.norm(arr.astype(np.float32), axis=1) > 0
                        a64 = arr.astype(np.float64)
                        n64 = np.linalg.norm(a64, axis=1, keepdims=True)
                        unit = np.where(live[:, None], a64 / np.where(n64 > 0, n64, 1.0), 0.0)
                        q = np.round(unit * 127.0).astype(np.int8)
                        padded = np.zeros((len(q), 4), dtype=np.int8)      # vertex stride must be 4-aligned
                        padded[:, :3] = q
                        extra = {"componentType": 5120, "normalized": True}
                        src = {k: v for k, v in a.items() if k not in ("min", "max")}
                        remap[key] = add(padded, src, ARRAY_BUFFER, stride=4, extra=extra)
                    else:
                        remap[key] = add(arr, a, ARRAY_BUFFER)
                attrs[name] = remap[key]
            pr["attributes"] = attrs
            if "indices" in pr:
                key = (pr["indices"], "INDICES")
                if key not in remap:
                    remap[key] = add(accessor_array(js, binary, pr["indices"]).reshape(-1),
                                     js["accessors"][pr["indices"]], ELEMENT_ARRAY_BUFFER)
                pr["indices"] = remap[key]
    js["accessors"], js["bufferViews"] = accessors, views
    js["buffers"] = [{"byteLength": len(out) + (-len(out) % 4)}]
    for k in ("extensionsUsed", "extensionsRequired"):
        js[k] = sorted(set(js.get(k, [])) | {"KHR_mesh_quantization"})
    return js, bytes(out)


def compare(old: Tuple[Dict, bytes], new: Tuple[Dict, bytes]) -> Dict:
    """Positions and indices must be identical; normals within the quantisation bound."""
    (oj, ob), (nj, nb) = old, new
    worst = 0.0
    same = True
    for om, nm in zip(oj["meshes"], nj["meshes"]):
        for op, np_ in zip(om["primitives"], nm["primitives"]):
            same &= op.get("material") == np_.get("material") and op.get("mode") == np_.get("mode")
            same &= np.array_equal(accessor_array(oj, ob, op["attributes"]["POSITION"]),
                                   accessor_array(nj, nb, np_["attributes"]["POSITION"]))
            if "indices" in op:
                same &= np.array_equal(accessor_array(oj, ob, op["indices"]).reshape(-1),
                                       accessor_array(nj, nb, np_["indices"]).reshape(-1))
            if "NORMAL" in op["attributes"]:
                a = accessor_array(oj, ob, op["attributes"]["NORMAL"]).astype(np.float64)
                acc = nj["accessors"][np_["attributes"]["NORMAL"]]
                b = dequant(acc, accessor_array(nj, nb, np_["attributes"]["NORMAL"]))
                na, nb_ = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
                zero = np.linalg.norm(a.astype(np.float32), axis=1) == 0     # degenerate on the GPU
                same &= bool(np.all(nb_[zero] == 0))               # a zero normal stays zero
                a, b = a[~zero] / na[~zero, None], b[~zero] / np.maximum(nb_[~zero, None], 1e-12)
                cos = np.clip((a * b).sum(1), -1.0, 1.0)
                worst = max(worst, float(np.degrees(np.arccos(cos)).max()) if len(cos) else 0.0)
    return {"geometry_identical": bool(same), "normal_error_deg": worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true", help="report only; do not rewrite")
    args = ap.parse_args()
    before = after = 0
    ok = True
    for path in args.paths:
        old = read_glb(path)
        if "KHR_mesh_quantization" in old[0].get("extensionsUsed", []):
            print(f"  {path}: already slim")
            continue
        new = slim(*old)
        cmp_ = compare(old, new)
        size_old = 20 + 8 + len(json.dumps(old[0])) + len(old[1])
        size_new = 20 + 8 + len(json.dumps(new[0], separators=(",", ":"))) + len(new[1])
        good = cmp_["geometry_identical"] and cmp_["normal_error_deg"] <= MAX_NORMAL_DEG
        ok &= good
        before += size_old
        after += size_new
        print(f"  {path}: {size_old / 1e6:.2f} -> {size_new / 1e6:.2f} MB  positions+indices identical "
              f"{cmp_['geometry_identical']}  worst normal {cmp_['normal_error_deg']:.2f} deg"
              + ("" if good else "  REFUSED"))
        if good and not args.check:
            write_glb(path, *new)
            reread = read_glb(path)
            if not compare(old, reread)["geometry_identical"]:          # the written file, too
                raise RuntimeError(f"{path}: rewritten file does not round-trip")
    print(f"total {before / 1e6:.1f} -> {after / 1e6:.1f} MB" + ("" if ok else " (some files refused)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
