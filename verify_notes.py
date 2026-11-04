"""verify_notes: the clinical-text infrastructure (backend/ml/notes), on invented notes only.

    PYTHONPATH=. python verify_notes.py            # or: make verify-notes

  1. READERS     assertion, negation (before and after the term, never across sentences),
                 hedging, the indication kept apart from the findings; both readers share names
  2. ENCODER     the prototype scoring on a deterministic stand-in embedder; local files only
  3. TIMING      every planted trap in ``notes.synthetic``: storetime not charttime, missing
                 storetime dropped, the ED lookback, the stay's end, prior summaries only
  4. FEATURES    the grid rows those traps produce, and PREFIX INVARIANCE: an hour's rows are
                 identical whether or not any later note exists
  5. PRIVACY     no text in any output; caches keyed by hash, refused inside the repository
  6. SOURCES     the release layout round-trips; the lake ingests the note tables, text intact
  7. MUTATIONS   each guard is broken on purpose and a check above is seen to go red
"""
from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.mimic import extract as E
from backend.ml.notes import features as F
from backend.ml.notes import readers as R
from backend.ml.notes import sources as S
from backend.ml.notes import synthetic as SY
from backend.ml.notes import timing as T

_results = []
_quiet = False


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok)))
    if not _quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def say(msg: str) -> None:
    if not _quiet:
        print(msg)


def _kw(text: str) -> dict:
    r = R.KeywordReader()
    return dict(zip(r.names, r.read([text])[0]))


def c1_readers() -> None:
    say("\n1. readers")
    check("an asserted finding scores 1", _kw("IMPRESSION: Right lower lobe consolidation.")["Note_pneumonia"] == 1.0)
    check("a negation before the term scores 0", _kw("IMPRESSION: No pneumonia.")["Note_pneumonia"] == 0.0)
    check("a negation after the term scores 0", _kw("IMPRESSION: The effusion has resolved.")["Note_effusion"] == 0.0)
    check("a hedge scores 0.5", _kw("IMPRESSION: Possible consolidation.")["Note_pneumonia"] == 0.5)
    s = _kw("IMPRESSION: No effusion. Consolidation in the left base.")
    check("a negation stays in its sentence", s["Note_pneumonia"] == 1.0 and s["Note_effusion"] == 0.0)
    s = _kw("INDICATION: Fever, evaluate for pneumonia.\nIMPRESSION: No acute process.")
    check("the clinician's question is suspicion, not a finding",
          s["Note_ind_infection"] == 1.0 and s["Note_pneumonia"] == 0.0, f"ind {s['Note_ind_infection']}, pneumonia {s['Note_pneumonia']}")
    s = _kw("Consolidation. INDICATION: none")
    check("text before any header is the answer", s["Note_pneumonia"] == 1.0)
    check("an empty or missing report is all zeros", not R.KeywordReader().read(["", None]).any())
    enc = R.EncoderReader(_stub_embed)
    check("both readers emit the same names, in order", enc.names == R.KeywordReader().names)
    check("readers differ in fingerprint", enc.fingerprint != R.KeywordReader().fingerprint)


_VOCAB = {}


def _stub_embed(texts):
    """Bag of words over a growing vocabulary, plus a strong 'negated' axis: deterministic, local,
    and enough to tell 'there is X' from 'there is no X'. It tests plumbing, not language."""
    out = np.zeros((len(texts), 512))
    for i, t in enumerate(texts):
        w = R._words(t)
        for x in w:
            if x in ("there", "is", "evaluate", "for"):
                continue
            out[i, _VOCAB.setdefault(x, 1 + len(_VOCAB) % 500)] += 1.0
        if any(c in w for c in ("no", "not", "without")):
            out[i, 0] += 3.0
    return out


def c2_encoder() -> None:
    say("\n2. encoder")
    enc = R.EncoderReader(_stub_embed, model_id="stub")
    s = dict(zip(enc.names, enc.read(["IMPRESSION: there is pneumonia", "IMPRESSION: there is no pneumonia"])[0]))
    n = dict(zip(enc.names, enc.read(["IMPRESSION: there is no pneumonia"])[0]))
    check("a sentence equal to the positive prototype scores 1", s["Note_pneumonia"] == 1.0)
    check("its negation scores 0", n["Note_pneumonia"] == 0.0)
    # with no floor, only the negated prototype's margin can keep a negation at 0
    open_ = R.EncoderReader(_stub_embed, model_id="stub", floor=0.0)
    m = dict(zip(open_.names, open_.read(["IMPRESSION: there is no pneumonia"])[0]))
    check("...because of the negated prototype, not only the similarity floor", m["Note_pneumonia"] == 0.0,
          f"{m['Note_pneumonia']:.3f} with floor 0")
    out = enc.read(["IMPRESSION: Pulmonary edema. No effusion.", "", "INDICATION: evaluate for infection"])
    check("scores stay in [0, 1], one row per text", out.shape == (3, len(enc.names)) and out.min() >= 0 and out.max() <= 1)
    check("the indication is scored apart", out[2][enc.names.index("Note_ind_infection")] == 1.0
          and not out[2][:len(R.FINDINGS)].any())
    check("floor and scale are part of the fingerprint",
          R.EncoderReader(_stub_embed, "stub", floor=0.4).fingerprint != enc.fingerprint)
    try:
        R.local_encoder(os.path.join(tempfile.gettempdir(), "no-such-model-dir"))
        refused = False
    except FileNotFoundError:
        refused = True
    check("a model not on disk is an error, never a download", refused)


def _rows():
    w = SY.world()
    return w, F.note_rows(w["radiology"], w["stays"], R.KeywordReader())


def _at(rows, stay, hr, name):
    v = rows[(rows.stay_id == stay) & (rows.hr == hr) & (rows.name == name)].value
    return float(v.iloc[0]) if len(v) else None


def c3_timing() -> None:
    say("\n3. timing")
    w = SY.world()
    av = T.radiology_available(w["radiology"], w["stays"])
    got = {n: int(h) for n, h in zip(av.note_id, av.hr) if True}
    s1 = av[av.stay_id == 1001]
    check("a read lands at its storetime hour, not when the film was taken", got.get("r_late") == 9, f"hour {got.get('r_late')}")
    check("a report without a storetime is dropped", "r_nostore" not in got)
    check("a read stored after the stay ended is dropped", "r_after" not in got)
    check("the ED film, stored 10 h before admission, is known at hour 0", got.get("r_ed") == 0)
    check(f"a read older than the {T.LOOKBACK_HOURS:.0f} h lookback is dropped", "r_old" not in got)
    check("a patient's reports inform only their own stays",
          set(av[av.note_id == "r_other"].stay_id) == {2001} and "r_other" not in set(s1.note_id))
    pr = T.prior_discharge(w["discharge"], w["stays"], w["admissions"])
    check("a prior admission's summary is history", set(pr[pr.stay_id == 1001].note_id) == {"d_prior"})
    check("the current admission's summary, written after the outcome, is never read", "d_current" not in set(pr.note_id))


def c4_features() -> None:
    say("\n4. features")
    w, rows = _rows()
    check("the consolidation read appears at hour 9 and nowhere earlier",
          _at(rows, 1001, 9, "Note_pneumonia") == 1.0 and rows[(rows.stay_id == 1001) & (rows.hr < 9)
                                                               & (rows.name == "Note_pneumonia")].value.max() == 0.0)
    check("the ED cholecystitis is known at hour 0", _at(rows, 1001, 0, "Note_cholecystitis") == 1.0)
    check("the dropped effusion read never appears", rows[(rows.stay_id == 1001) & (rows.name == "Note_effusion")].value.max() == 0.0)
    check("negated and hedged reads score 0 and 0.5", _at(rows, 1001, 14, "Note_pneumonia") == 0.0
          and _at(rows, 1001, 21, "Note_pneumonia") == 0.5)
    check("two reads in one hour: the highest per finding, and a count of 2",
          _at(rows, 1001, 35, "Note_edema") == 1.0 and _at(rows, 1001, 35, "Note_atelectasis") == 1.0
          and _at(rows, 1001, 35, "Note_abscess") == 0.5 and _at(rows, 1001, 35, F.COUNT) == 2.0)
    check("nothing at or after the stay's end", rows[rows.stay_id == 1001].hr.max() < 48
          and rows[rows.stay_id == 2001].hr.max() < 30)
    hist = F.history_rows(w["discharge"], w["stays"], w["admissions"], R.KeywordReader())
    hv = {n: v for s, n, v in hist.itertuples(index=False) if s == 1001}
    check("history carries the prior summary's findings", hv.get("NoteHist_urinary") == 1.0 and hv.get("NoteHist_abscess") == 1.0
          and hv.get(F.HIST_COUNT) == 1.0)
    check("...and none of the current admission's", hv.get("NoteHist_pneumonia") == 0.0 and hv.get("NoteHist_empyema") == 0.0)
    check("a patient with no prior admission has no history rows", not (hist.stay_id == 2001).any())

    # prefix invariance: drop every note stored after hour k; rows for hours < k must not move
    full = rows
    moved = []
    for k in range(0, 48, 3):
        cut = w["radiology"][w["radiology"].storetime.isna() | (w["radiology"].storetime < SY.h(k))]
        part = F.note_rows(cut, w["stays"], R.KeywordReader())
        a = full[full.hr < k].reset_index(drop=True)
        b = part[part.hr < k].reset_index(drop=True)
        if not a.equals(b):
            moved.append(k)
    check("prefix invariance: no hour's rows depend on a later note", not moved, f"moved at cut hours {moved}")
    names = set(F.names_all())
    check("every note feature is tier S", set(F.tiers().values()) == {"S"} and set(F.tiers()) == names)
    check("no note feature name collides with the grid's", not names & set(E.DYNAMIC + E.STATIC))


def c5_privacy() -> None:
    say("\n5. privacy")
    w, rows = _rows()
    text = set(w["radiology"].text) | set(w["discharge"].text)
    flat = {str(x) for x in rows.to_numpy().ravel()}
    check("no report text appears in the feature rows", not (flat & text) and rows.dtypes.astype(str).tolist().count("object") <= 1)
    bad = rows.assign(note=w["radiology"].text.iloc[0])
    try:
        F._numeric(bad, ["stay_id", "hr"])
        refused = False
    except TypeError:
        refused = True
    check("a text column is refused, not passed on", refused)
    try:
        F._numeric(rows.assign(name=rows.name.where(rows.index != 0, "IMPRESSION: Pneumonia.")), ["stay_id", "hr"])
        refused = False
    except TypeError:
        refused = True
    check("so is text smuggled in as a feature name", refused)
    try:
        R.CachedReader(R.KeywordReader(), os.path.join(config.REPO_ROOT, "artifacts", "note-cache"))
        refused = False
    except ValueError:
        refused = True
    check("a score cache inside the repository is refused", refused)
    tmp = tempfile.mkdtemp()
    try:
        c = R.CachedReader(R.KeywordReader(), tmp)
        texts = list(w["radiology"].text)
        a = c.read(texts)
        first = c.misses
        b = c.read(texts)
        check("cached scores are identical, and the second read is all hits",
              np.array_equal(a, b) and first == len(texts) and c.misses == first, f"misses {first} then {c.misses - first}")
        stored = [f for _, _, fs in os.walk(tmp) for f in fs]
        check("the cache is keyed by hash: no file name or file holds the text",
              all(len(f) == 68 for f in stored) and not any(t.encode() in open(os.path.join(dp, f), "rb").read()
                                                             for dp, _, fs in os.walk(tmp) for f in fs for t in texts))
        check("a different reader never shares the cache", R.CachedReader(R.EncoderReader(_stub_embed), tmp).dir != c.dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def c6_sources() -> None:
    say("\n6. sources")
    from backend.ml.physionet import fetch
    from backend.ml.physionet import lake as PL
    tmp = tempfile.mkdtemp()
    try:
        SY.write_release(tmp)
        src = S.NoteSource(tmp)
        w = SY.world()
        r = src.read("radiology")
        check("the release layout round-trips", len(r) == len(w["radiology"]) and set(r.note_id) == set(w["radiology"].note_id)
              and r.storetime.isna().sum() == 1)
        check("reading by subject keeps only those subjects", set(src.read("radiology", {2}).subject_id) == {2})
        root = os.path.join(tmp, "lake")
        for spec in S.table_specs():
            st = fetch.LocalStream(os.path.join(tmp, spec.source))
            PL.ingest(spec, st, root, None)
            st.close()
        spec = {s.table: s for s in S.table_specs()}["radiology"]
        import pyarrow.parquet as pq
        d = PL.table_dir(root, spec)
        back = pd.concat([pq.read_table(os.path.join(d, f)).to_pandas() for f in PL.read_manifest(root, spec)["files"]])
        h = lambda xs: hashlib.sha256("\n".join(sorted(x for x in xs if isinstance(x, str))).encode()).hexdigest()
        check("the lake ingests the note tables with every report's text intact",
              len(back) == len(w["radiology"]) and h(back.text) == h(w["radiology"].text))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


SECTIONS = {"1": c1_readers, "2": c2_encoder, "3": c3_timing, "4": c4_features, "5": c5_privacy, "6": c6_sources}


@contextmanager
def _mutated(module, old: str, new: str):
    src = inspect.getsource(module)
    if old not in src:
        raise AssertionError(f"mutation target not found in {module.__name__}: {old!r}")
    saved = {k: v for k, v in vars(module).items() if inspect.isfunction(v) or inspect.isclass(v)}
    try:
        exec(compile(src.replace(old, new), module.__file__, "exec"), module.__dict__)
        yield
    finally:
        module.__dict__.update(saved)


MUTATIONS = [
    ("the film's time, not the read's", T, "t = _hours(j.storetime, j.intime)", "t = _hours(j.charttime, j.intime)", "3"),
    ("a missing storetime guessed from the film", T, 'n = notes.dropna(subset=["storetime"])',
     'n = notes.assign(storetime=notes.storetime.fillna(notes.charttime))', "3"),
    ("no lookback limit", T, "keep = (t >= -LOOKBACK_HOURS) & (t < end)", "keep = t < end", "3"),
    ("reads past the stay's end", T, "keep = (t >= -LOOKBACK_HOURS) & (t < end)", "keep = t >= -LOOKBACK_HOURS", "3"),
    ("the current admission's summary read", T, "keep = (j.dischtime < j.intime) & (j.written < j.intime) & (j.hadm_id != j.hadm_id_stay)",
     "keep = j.written < j.intime + pd.Timedelta(days=30)", "3"),
    ("no negation", R, "if _cue_before(w, i, NEGATE_BEFORE) or any(_find(after, c) for c in NEGATE_AFTER):", "if False:", "1"),
    ("negation only before the term", R, " or any(_find(after, c) for c in NEGATE_AFTER):", ":", "1"),
    ("no hedging", R, "elif _cue_before(w, i, HEDGE) or any(_find(after[:CUE_WINDOW], c) for c in HEDGE):", "elif False:", "1"),
    ("the indication read as findings", R, '(q if kind == "q" else a).append', "a.append", "1"),
    ("the encoder ignores its negated prototype", R, "np.clip((cp - cn) / self.scale, 0.0, 1.0)", "np.clip(cp / self.scale, 0.0, 1.0)", "2"),
    ("text passed through", F, "if c not in keys and c != \"name\" and not pd.api.types.is_numeric_dtype(df[c]):", "if False:", "5"),
    ("any feature name accepted", F, 'if "name" in df and not df.name.isin(set(names_all())).all():', "if False:", "5"),
    ("a cache allowed in the repository", R, "if d == root or d.startswith(root + os.sep):", "if False:", "5"),
    ("a cache shared across readers", R, 'self.dir = os.path.join(directory, f"{inner.name}-{inner.fingerprint}")',
     "self.dir = directory", "5"),
]


def run_mutations() -> None:
    global _quiet
    print("\n7. mutations")
    for label, module, old, new, section in MUTATIONS:
        before = len(_results)
        _quiet = True
        try:
            with _mutated(module, old, new):
                try:
                    SECTIONS[section]()
                    red = not all(ok for _, ok in _results[before:])
                except Exception:
                    red = True
        finally:
            del _results[before:]
            _quiet = False
        check(f"caught: {label}", red, f"section {section}")


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = sys.argv[1:] if argv is None else argv
    for fn in SECTIONS.values():
        fn()
    if "--no-mutations" not in args:
        run_mutations()
    ok = sum(r[1] for r in _results)
    failed = [r[0] for r in _results if not r[1]]
    print(f"\nverify_notes: {ok}/{len(_results)} passed")
    for name in failed:
        print("  FAILED:", name)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
