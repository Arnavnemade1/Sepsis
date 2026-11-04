"""Text -> a fixed set of finding scores in [0, 1]. Two readers, one interface, compared head to head.

A radiology report has two voices. The INDICATION / HISTORY section is the clinician's
question ("fever, eval for pneumonia"): suspicion, scored as ``ind_*``. The FINDINGS /
IMPRESSION sections are the radiologist's answer, scored per finding. A finding is 1 when
asserted, 0.5 when hedged ("possible", "cannot exclude"), 0 when negated or absent.

``KeywordReader`` is the baseline: a lexicon, sentence-scoped negation and hedging cues (in the
spirit of NegEx). ``EncoderReader`` scores the same names with a local text encoder: each
sentence against a positive and a negated prototype of each finding. The encoder is kept only
if it beats the keywords on the dev folds; both carry a ``fingerprint`` so cached scores and
checkpoints can never be mixed between readers or versions.

Neither reader logs, prints or returns text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: finding -> terms (lower case, matched on word boundaries). Infection sources first, then the
#: findings that offer another explanation for the same picture (edema, atelectasis).
FINDINGS: Dict[str, Tuple[str, ...]] = {
    "pneumonia": ("pneumonia", "consolidation", "consolidative", "infiltrate", "airspace disease",
                  "airspace opacity", "airspace opacities"),
    "effusion": ("pleural effusion", "pleural effusions", "effusion", "effusions"),
    "empyema": ("empyema", "loculated effusion", "loculated pleural"),
    "abscess": ("abscess", "abscesses", "rim-enhancing collection", "fluid collection", "collection"),
    "free_air": ("free air", "pneumoperitoneum", "free intraperitoneal air", "perforation"),
    "colitis": ("colitis", "bowel wall thickening", "pneumatosis"),
    "cholecystitis": ("cholecystitis", "cholangitis", "gallbladder wall thickening"),
    "urinary": ("pyelonephritis", "hydronephrosis", "perinephric stranding"),
    "soft_tissue": ("cellulitis", "fasciitis", "soft tissue gas", "subcutaneous gas"),
    "edema": ("pulmonary edema", "interstitial edema", "vascular congestion", "fluid overload"),
    "atelectasis": ("atelectasis", "atelectatic"),
}
#: the clinician's question, read from the indication only
INDICATIONS: Dict[str, Tuple[str, ...]] = {
    "infection": ("infection", "infectious", "sepsis", "septic", "fever", "febrile", "pneumonia",
                  "abscess", "leukocytosis", "elevated wbc", "source"),
    "respiratory": ("hypoxia", "hypoxemia", "shortness of breath", "sob", "dyspnea", "desaturation",
                    "respiratory distress", "increased oxygen"),
}
NEGATE_BEFORE = ("no", "not", "without", "negative for", "free of", "no evidence of", "absence of",
                 "resolution of", "resolved", "neither", "nor", "no new", "no definite")
NEGATE_AFTER = ("is not seen", "are not seen", "has resolved", "have resolved", "is absent", "are absent",
                "not identified", "is not identified", "are not identified", "has cleared", "ruled out")
HEDGE = ("possible", "possibly", "may represent", "may reflect", "cannot be excluded", "cannot exclude",
         "can not be excluded", "questionable", "equivocal", "concern for", "suspicious for", "versus",
         "differential", "rule out", "could represent", "not excluded", "probable")
CUE_WINDOW = 6          # words before a term that a negation or hedge cue may stand in

_INDICATION_HEAD = re.compile(r"^\s*(indication|history|clinical history|reason for exam|clinical information|"
                              r"reason for examination|clinical indication)\s*:", re.I | re.M)
_ANSWER_HEAD = re.compile(r"^\s*(findings|impression|conclusion|comparison|technique|interpretation)\s*:", re.I | re.M)
_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")


def split_sections(text: str) -> Tuple[str, str]:
    """(indication, answer). Text before any header, and every non-indication section, is answer."""
    heads = sorted([(m.start(), "q") for m in _INDICATION_HEAD.finditer(text)] +
                   [(m.start(), "a") for m in _ANSWER_HEAD.finditer(text)])
    if not heads:
        return "", text
    q, a = [], [text[:heads[0][0]]]
    for (start, kind), nxt in zip(heads, heads[1:] + [(len(text), "")]):
        (q if kind == "q" else a).append(text[start:nxt[0]])
    return "\n".join(q), "\n".join(a)


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def _words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s.lower())


def _find(words: List[str], phrase: str) -> List[int]:
    p = _words(phrase)
    return [i for i in range(len(words) - len(p) + 1) if words[i:i + len(p)] == p]


def _cue_before(words: List[str], i: int, cues: Sequence[str]) -> bool:
    lo = max(0, i - CUE_WINDOW)
    return any(lo <= j < i for c in cues for j in _find(words, c))


def score_sentence(sentence: str, terms: Sequence[str]) -> Optional[float]:
    """1 asserted, 0.5 hedged, 0 negated; None when no term occurs."""
    w = _words(sentence)
    best = None
    for t in terms:
        for i in _find(w, t):
            after = w[i + len(_words(t)):]
            if _cue_before(w, i, NEGATE_BEFORE) or any(_find(after, c) for c in NEGATE_AFTER):
                s = 0.0
            elif _cue_before(w, i, HEDGE) or any(_find(after[:CUE_WINDOW], c) for c in HEDGE):
                s = 0.5
            else:
                s = 1.0
            best = s if best is None else max(best, s)
    return best


class Reader:
    """``read(texts) -> (n, len(names))`` scores in [0, 1]. Subclasses set ``name`` and ``_params``."""
    name = "reader"

    @property
    def names(self) -> List[str]:
        return [f"Note_{f}" for f in FINDINGS] + [f"Note_ind_{k}" for k in INDICATIONS]

    def _params(self) -> Dict:
        return {}

    @property
    def fingerprint(self) -> str:
        body = {"reader": self.name, "findings": FINDINGS, "indications": INDICATIONS, **self._params()}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def read(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class KeywordReader(Reader):
    name = "keyword"

    def _params(self) -> Dict:
        return {"negate": [NEGATE_BEFORE, NEGATE_AFTER], "hedge": HEDGE, "window": CUE_WINDOW}

    def read(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.names)), dtype=np.float32)
        for r, text in enumerate(texts):
            q, a = split_sections(text or "")
            sa, sq = sentences(a), sentences(q)
            for c, terms in enumerate(FINDINGS.values()):
                hits = [x for x in (score_sentence(s, terms) for s in sa) if x is not None]
                out[r, c] = max(hits) if hits else 0.0
            for c, terms in enumerate(INDICATIONS.values(), start=len(FINDINGS)):
                hits = [x for x in (score_sentence(s, terms) for s in sq) if x is not None]
                out[r, c] = 1.0 if hits and max(hits) > 0 else 0.0     # asked about, even as "r/o"
        return out


Embedder = Callable[[List[str]], np.ndarray]


class EncoderReader(Reader):
    """The same findings, scored by similarity. Each answer sentence is compared with a positive
    prototype ("there is pneumonia") and a negated one ("there is no pneumonia") of each finding;
    the score is the margin ``cos(pos) - cos(neg)`` scaled to [0, 1] by ``scale``, counted only when
    the sentence is about the finding at all (``cos(pos) >= floor``), and maxed over sentences.
    ``floor`` and ``scale`` are untuned defaults: fit them on dev folds, never on the lock.

    ``embed`` maps strings to vectors. ``local_encoder`` builds one from a model on disk."""
    name = "encoder"

    def __init__(self, embed: Embedder, model_id: str = "custom", floor: float = 0.5, scale: float = 0.2):
        self.embed, self.model_id, self.floor, self.scale = embed, model_id, floor, scale
        self._proto: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def _params(self) -> Dict:
        return {"model": self.model_id, "floor": self.floor, "scale": self.scale}

    def _unit(self, texts: List[str]) -> np.ndarray:
        v = np.asarray(self.embed(texts), dtype=np.float64)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def prototypes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._proto is None:
            pos = [f"there is {t[0]}" for t in FINDINGS.values()]
            neg = [f"there is no {t[0]}" for t in FINDINGS.values()]
            ind = [f"evaluate for {t[0]}" for t in INDICATIONS.values()]
            self._proto = (self._unit(pos), self._unit(neg), self._unit(ind))
        return self._proto

    def read(self, texts: Sequence[str]) -> np.ndarray:
        pos, neg, ind = self.prototypes()
        out = np.zeros((len(texts), len(self.names)), dtype=np.float32)
        for r, text in enumerate(texts):
            q, a = split_sections(text or "")
            sa, sq = sentences(a), sentences(q)
            if sa:
                e = self._unit(sa)
                cp, cn = e @ pos.T, e @ neg.T                     # (sentences, findings)
                m = np.where(cp >= self.floor, np.clip((cp - cn) / self.scale, 0.0, 1.0), 0.0)
                out[r, :len(FINDINGS)] = m.max(axis=0)
            if sq:
                ci = self._unit(sq) @ ind.T
                out[r, len(FINDINGS):] = (ci.max(axis=0) >= self.floor).astype(np.float32)
        return out


def local_encoder(path: str, max_tokens: int = 128, batch: int = 32) -> Embedder:  # pragma: no cover - needs a model
    """Mean-pooled embeddings from a ``transformers`` model ON DISK. ``local_files_only`` means a
    missing file is an error, never a download, and nothing is sent anywhere."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path}: download the model weights first; this never fetches them")
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def embed(texts: List[str]) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
                h = model(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
        return np.concatenate(outs) if outs else np.zeros((0, model.config.hidden_size))
    return embed


class CachedReader(Reader):
    """Scores cached by ``sha256(text)`` under ``directory`` (inside the lake, never the repo),
    keyed by the inner reader's fingerprint: a lexicon or model change is a new cache."""

    def __init__(self, inner: Reader, directory: str, repo_root: Optional[str] = None):
        from backend.ml import config
        root = os.path.realpath(repo_root or config.REPO_ROOT)
        d = os.path.realpath(directory)
        if d == root or d.startswith(root + os.sep):
            raise ValueError(f"{directory} is inside the repository; note-derived caches live in the lake")
        self.inner, self.name = inner, f"cached-{inner.name}"
        self.dir = os.path.join(directory, f"{inner.name}-{inner.fingerprint}")
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        self.misses = 0

    @property
    def names(self) -> List[str]:
        return self.inner.names

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def read(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha256((t or "").encode()).hexdigest() for t in texts]
        out = np.zeros((len(texts), len(self.names)), dtype=np.float32)
        todo = []
        for i, k in enumerate(keys):
            p = os.path.join(self.dir, k[:2], k + ".npy")
            if os.path.exists(p):
                out[i] = np.load(p)
            else:
                todo.append(i)
        if todo:
            self.misses += len(todo)
            fresh = self.inner.read([texts[i] for i in todo])
            for j, i in enumerate(todo):
                p = os.path.join(self.dir, keys[i][:2], keys[i] + ".npy")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, fresh[j])
                out[i] = fresh[j]
        return out
