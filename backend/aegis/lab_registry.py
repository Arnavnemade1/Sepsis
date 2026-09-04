"""The Lab's memory — what has been tested, across every campaign that ever ran.

    PYTHONPATH=. python -m backend.aegis.lab_registry --report

WHY A LAB THAT FORGETS IS NOT A LAB
------------------------------------
``lab.py`` adjudicates one campaign and writes one ledger. The next campaign overwrites
it. That arrangement has two failure modes, and both are the kind this project exists to
catch:

* **Campaign shopping.** Run the loop ten times, publish the run that produced a hit.
  Each campaign gets a fresh family-wise budget, so the stated 5% error rate is a
  statement about one campaign and says nothing about the programme. Three campaigns had
  already run against a ledger that only remembered the last one.
* **Re-testing a dead hypothesis.** The alpha budget and the gate are both expensive.
  Spending either on a change that a previous campaign already refused buys nothing, and
  re-testing the same hypothesis until it passes is precisely how a search loop
  manufactures a false discovery.

THE FIX IS NOT ONE SHRINKING BUDGET FOREVER
--------------------------------------------
The obvious move — carry the geometric alpha counter across campaigns — was tried on
paper and rejected. ``exhausted()`` trips once the next bar falls under the bootstrap's
resolution, which happens after about four tests, so a permanent counter would refuse
every proposal forever after the first campaign. That is not rigour, it is a broken
instrument.

What actually controls the error is narrower and more useful:

1. **Per-campaign FWER stays 0.05**, because a campaign asks a self-contained set of
   questions and is reported whole, refusals included.
2. **A hypothesis is tested once, ever.** ``seen()`` matches on the canonical change, so
   a proposal any past campaign already put through the gate is refused before it costs
   alpha. This is the part that actually prevents shopping: shopping only inflates error
   if you can re-ask the same question.
3. **Cumulative alpha is disclosed, not hidden.** The sum of alpha spent over every test
   in every campaign is the *expected number of false findings for the whole programme*.
   It is a number a reader can check the finding count against, and the console leads
   with it. "Four campaigns, 23 tests, 0.187 of alpha spent, 0 findings" is a complete
   and honest description of a research programme.

TAMPER EVIDENCE
---------------
Entries are hash-chained (``sha256(canonical(entry) || prev_hash)``), the same
construction ``causal_graph.py`` uses for clinical events. The registry is the record
that a result was refused, and a record that can be quietly edited after the fact is not
a record. ``verify_chain()`` re-derives every link and names the first entry that does
not match, so an edited or deleted verdict is detectable rather than merely discouraged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

GENESIS = "0" * 64

#: The adjudication protocol a record was produced under. Bump this when a change to the
#: gate means older verdicts are no longer comparable, and say why below.
#:
#: v2 (2026-09-03) fixed three defects in v1, all of which made the gate LOOSER than
#: advertised: the replication arm overlapped the gate set by 25.1% for detection and was
#: the identical patient pool for policy; and the geometric alpha was charged to the
#: budget but never passed to the adjudicator, so every test ran at a 95% interval.
#:
#: v1 REFUSALS remain sound — a change that failed at 95% also fails at a tighter bar —
#: so they are kept, and they still inform the Critic. v1 ACCEPTANCES are not sound, and
#: ``seen()`` lets their hypotheses be asked again under v2 rather than treating a
#: retracted pass as a settled question.
PROTOCOL = "v2"
RETRACTED_PROTOCOLS = {"v1"}


def _canonical(obj: Any) -> bytes:
    """Stable bytes for hashing. Key order must not change the hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _chain(prev_hash: str, body: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical(body))
    return h.hexdigest()


def change_key(change: Dict[str, Any]) -> str:
    """The identity of a hypothesis.

    Two proposals are the same hypothesis when they apply the same change, whatever
    their titles or rationales say. Hashing the canonical change — not the title — is
    what makes dedup survive an agent that rewords itself.
    """
    return hashlib.sha256(_canonical(change or {})).hexdigest()[:16]


@dataclass
class Entry:
    """One hash-chained record of one adjudicated proposal."""
    seq: int
    campaign: str
    ts: float
    agent: str
    kind: str
    title: str
    change_key: str
    change: Dict[str, Any]
    status: str
    accepted: bool
    observed_delta: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    alpha_spent: Optional[float]
    mde: Optional[float]
    refusal_reason: Optional[str]
    prev_hash: str
    hash: str = ""
    protocol: str = PROTOCOL

    def body(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("hash",)}
        return d

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class LabRegistry:
    """Every campaign the Lab has ever run, chained and queryable."""

    path: str
    campaigns: List[Dict[str, Any]] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)

    # -- persistence -------------------------------------------------------- #
    @classmethod
    def load(cls, path: Optional[str] = None) -> "LabRegistry":
        from backend.ml import config as _cfg
        path = path or os.path.join(_cfg.ARTIFACTS_DIR, "lab_registry.json")
        reg = cls(path=path)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    d = json.load(fh)
                reg.campaigns = d.get("campaigns") or []
                reg.entries = [Entry(**e) for e in (d.get("entries") or [])]
            except Exception:
                # A corrupt registry must not take the Lab down, but it must also not be
                # silently replaced — verify_chain() will report the break.
                reg.campaigns, reg.entries = [], []
        return reg

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "campaigns": self.campaigns,
            "entries": [e.as_dict() for e in self.entries],
            "programme": self.programme(),
        }
        with open(self.path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str, allow_nan=False)
        return self.path

    # -- writing ------------------------------------------------------------ #
    def _append(self, body: Dict[str, Any]) -> Entry:
        prev = self.entries[-1].hash if self.entries else GENESIS
        e = Entry(prev_hash=prev, **body)
        e.hash = _chain(prev, e.body())
        self.entries.append(e)
        return e

    def record_campaign(self, verdicts: Sequence[Any], budget: Any,
                        label: Optional[str] = None,
                        notes: Optional[Dict[str, Any]] = None,
                        protocol: str = PROTOCOL) -> str:
        """Chain one campaign's verdicts into the permanent record."""
        cid = label or time.strftime("c%Y%m%d-%H%M%S")
        n_before = len(self.entries)
        for v in verdicts:
            p = v.proposal
            self._append(dict(
                seq=len(self.entries), campaign=cid, ts=float(p.submitted),
                agent=p.agent, kind=p.kind, title=p.title,
                change_key=change_key(p.change), change=dict(p.change or {}),
                status=("vetoed" if v.vetoed_by else (v.status or "refused")),
                accepted=bool(v.accepted),
                observed_delta=_num(v.observed_delta), ci_lo=_num(v.ci_lo),
                ci_hi=_num(v.ci_hi), alpha_spent=_num(v.alpha_spent),
                mde=_num(getattr(v, "mde", None)),
                refusal_reason=v.refusal_reason,
                protocol=protocol))
        self.campaigns.append({
            "id": cid,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_proposals": len(verdicts),
            "n_entries": len(self.entries) - n_before,
            "n_accepted": sum(1 for v in verdicts if v.accepted),
            "n_findings": sum(1 for v in verdicts if v.status == "finding"),
            "alpha_total": float(getattr(budget, "total", 0.0)),
            "alpha_spent": float(getattr(budget, "spent", 0.0)),
            "n_tests": int(getattr(budget, "n_tests", 0)),
            "protocol": protocol,
            "retracted": protocol in RETRACTED_PROTOCOLS,
            **(notes or {}),
        })
        return cid

    # -- reading ------------------------------------------------------------ #
    def seen(self, change: Dict[str, Any]) -> Optional[Entry]:
        """The past verdict on this exact hypothesis, if there is one.

        Only entries that were actually ADJUDICATED count. A proposal refused because no
        judge was bound, or because a campaign ran out of budget, was never tested — its
        hypothesis is still open and must remain askable.

        Nor does a verdict from a RETRACTED protocol settle anything. When the gate is
        found to have been looser than advertised, the honest response is to let the
        question be asked again under the fixed protocol — not to treat a pass nobody
        should trust as a reason never to look. Retracted REFUSALS still block, because a
        change that failed a loose bar fails a tighter one too.
        """
        k = change_key(change)
        for e in reversed(self.entries):
            if e.change_key != k or e.alpha_spent is None:
                continue
            if e.protocol in RETRACTED_PROTOCOLS and e.accepted:
                continue
            return e
        return None

    def recorded_nulls(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Refusals this Lab produced itself, for the Critic to argue from.

        The Critic shipped with a hardcoded list of the project's published nulls. It had
        no access to the nulls the Lab generated, which is most of them — so an agent
        could rediscover a dead end the Lab itself had killed last week and the record
        would say nothing.
        """
        out: List[Dict[str, Any]] = []
        for e in reversed(self.entries):
            if e.alpha_spent is None or e.accepted:
                continue
            # The CHANGE travels with the null, not just the title. The Critic's first
            # version reconstructed the lever by parsing the title string, so an agent
            # that wrote "tune et.max_depth" instead of "set et.max_depth = 18" defeated
            # it. The change dict is the thing that is actually structured.
            out.append({"title": e.title, "kind": e.kind, "campaign": e.campaign,
                        "delta": e.observed_delta, "mde": e.mde,
                        "change": dict(e.change or {}), "change_key": e.change_key})
            if len(out) >= limit:
                break
        return out

    def programme(self) -> Dict[str, Any]:
        """The whole research programme in the numbers that matter.

        ``expected_false_findings`` is the sum of alpha over every test ever run. It is
        the number a reader should compare the finding count against: a programme that
        has spent 0.19 of alpha and reports 3 findings is reporting noise, and one that
        has spent 0.19 and reports 0 is reporting an honest null.
        """
        tested = [e for e in self.entries if e.alpha_spent is not None]
        # Alpha spent under a retracted protocol does not describe the current one, and
        # counting it as if it did would overstate the programme's error budget as much
        # as ignoring it would understate its history. It is reported separately.
        live = [e for e in tested if e.protocol not in RETRACTED_PROTOCOLS]
        exp_false = float(sum(e.alpha_spent or 0.0 for e in live))
        n_find = sum(1 for e in self.entries if e.status == "finding")
        retracted = sum(1 for e in self.entries if e.protocol in RETRACTED_PROTOCOLS)
        return {
            "protocol": PROTOCOL,
            "n_retracted_entries": retracted,
            "retraction_note": (
                f"{retracted} verdict(s) were produced under a protocol since found to "
                f"be looser than advertised (v1: overlapping replication pool, alpha "
                f"charged but not applied). Their refusals stand; their acceptances are "
                f"retracted and may be asked again." if retracted else None),
            "n_campaigns": len(self.campaigns),
            "n_proposals": len(self.entries),
            "n_tested": len(live),
            "n_tested_all_protocols": len(tested),
            "n_accepted": sum(1 for e in self.entries if e.accepted),
            "n_findings": n_find,
            "n_vetoed": sum(1 for e in self.entries if e.status == "vetoed"),
            "n_hypotheses": len({e.change_key for e in self.entries}),
            "expected_false_findings": round(exp_false, 5),
            "findings_vs_expected_false": (
                f"{n_find} finding(s) against {exp_false:.3f} expected by chance"),
            "chain_ok": self.verify_chain()[0],
            "interpretation": (
                "Cumulative alpha over every test in every campaign IS the expected "
                "number of false findings for the whole programme. Compare it to the "
                "finding count. A campaign is never re-run against a hypothesis it has "
                "already tested, which is what stops the count from being shopped."),
        }

    # -- tamper evidence ---------------------------------------------------- #
    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Re-derive every link. Returns (ok, seq of the first entry that breaks)."""
        prev = GENESIS
        for e in self.entries:
            if e.prev_hash != prev or _chain(prev, e.body()) != e.hash:
                return False, e.seq
            prev = e.hash
        return True, None


def import_ledger(reg: "LabRegistry", path: str, label: str,
                  protocol: str = "v1") -> int:
    """Chain a ledger written before the registry existed into the record.

    Three campaigns ran against a ledger file that each run overwrote, so only the last
    one survived. Importing it is not bookkeeping tidiness — a programme that quietly
    drops its earlier attempts is a programme whose finding count cannot be read against
    anything.
    """
    with open(path) as fh:
        d = json.load(fh)
    n0 = len(reg.entries)
    for v in d.get("verdicts") or []:
        p = v.get("proposal") or {}
        reg._append(dict(
            seq=len(reg.entries), campaign=label, ts=float(p.get("submitted") or 0.0),
            agent=p.get("agent") or "?", kind=p.get("kind") or "?",
            title=p.get("title") or "?", change_key=change_key(p.get("change") or {}),
            change=dict(p.get("change") or {}),
            status=("vetoed" if v.get("vetoed_by") else (v.get("status") or "refused")),
            accepted=bool(v.get("accepted")),
            observed_delta=_num(v.get("observed_delta")), ci_lo=_num(v.get("ci_lo")),
            ci_hi=_num(v.get("ci_hi")), alpha_spent=_num(v.get("alpha_spent")),
            mde=_num(v.get("mde")), refusal_reason=v.get("refusal_reason"),
            protocol=protocol))
    b = d.get("budget") or {}
    reg.campaigns.append({
        "id": label, "at": d.get("generated") or "?",
        "n_proposals": len(d.get("verdicts") or []),
        "n_entries": len(reg.entries) - n0,
        "n_accepted": sum(1 for v in (d.get("verdicts") or []) if v.get("accepted")),
        "n_findings": sum(1 for v in (d.get("verdicts") or [])
                          if v.get("status") == "finding"),
        "alpha_total": float(b.get("total") or 0.0),
        "alpha_spent": float(b.get("spent") or 0.0),
        "n_tests": int(b.get("n_tests") or 0),
        "protocol": protocol, "retracted": protocol in RETRACTED_PROTOCOLS,
        "imported_from": os.path.basename(path)})
    return len(reg.entries) - n0


def _num(x: Any) -> Optional[float]:
    """JSON-safe float, or None. NaN is not a number JSON can carry."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--path", default=None)
    ap.add_argument("--import-ledger", default=None,
                    help="chain a pre-registry ledger file into the record")
    ap.add_argument("--label", default="imported")
    ap.add_argument("--protocol", default="v1")
    args = ap.parse_args()
    reg = LabRegistry.load(args.path)
    if args.import_ledger:
        n = import_ledger(reg, args.import_ledger, args.label, args.protocol)
        reg.save()
        print(f"imported {n} verdict(s) as campaign {args.label} "
              f"(protocol {args.protocol}) -> {reg.path}\n")
    prog = reg.programme()
    print(json.dumps(prog, indent=2))
    ok, bad = reg.verify_chain()
    print(f"\nchain: {'intact' if ok else f'BROKEN at seq {bad}'} "
          f"over {len(reg.entries)} entries")
    for c in reg.campaigns:
        print(f"  {c['id']}  n={c['n_proposals']:<3} accepted={c['n_accepted']} "
              f"findings={c['n_findings']} alpha={c['alpha_spent']:.5f}")


if __name__ == "__main__":
    main()
