"""A Clinical Safety-Logic DSL that compiles to Z3 (MEGA-UPGRADE Layer 2).

Today the neurosymbolic safety rules are hand-written Python. This module turns them
into a small **domain-specific language** a hospital can author in near-clinical
prose, which the OS **compiles to Z3 assertions** and **proves** a proposed protocol
satisfies — emitting a machine-readable, auditable proof certificate per rule.

    RULE sepsis_antibiotic:
        ALWAYS (risk = HIGH -> WITHIN 1hr antibiotic != NONE)
    RULE allergy_guard:
        ALWAYS (patient.penicillin_allergy -> antibiotic != PipTazo)
    RULE fluid_cap:
        ALWAYS (patient.CHF -> fluid_rate <= 5.0)

What this is — and is NOT (the integrity rules; this is the crux of the claim):

  * It is a **bounded safety-constraint language**, not a full LTL/CTL temporal
    model checker. ``ALWAYS(φ)`` is the invariant that φ must hold for the decision
    under evaluation; ``WITHIN <t>hr ψ`` requires ψ AND that the action's *planned*
    administration time (``antibiotic_time_hr`` in the protocol — the Hour-1-bundle
    timing) is ≤ t. We check a *decision point against its planned timeline*, not an
    infinite execution trace. Stated, not hidden.

  * The compilation to Z3 is **real**: each rule becomes a Z3 ``BoolRef``. "Proven"
    means the solver shows the protocol's bindings make the rule's *negation*
    ``unsat`` — a genuine validity proof, the same discipline as the existing cage.

  * It composes the existing Z3 stack rather than replacing it; the unique
    contribution is that *hospitals write the rules*, and verification is automatic.

Bonus that falls out for free: :func:`check_consistency` proves whether a rule set
is *jointly satisfiable* for a patient — i.e., it can detect a self-contradictory
hospital policy (two rules no protocol can satisfy at once) before deployment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from z3 import (And, Bool, BoolRef, If, Implies, Int, Not, Or, Real, RealVal,
                IntVal, BoolVal, Solver, sat, unsat)
from backend.aegis.z3lock import z3_guarded as _z3_guarded


# --------------------------------------------------------------------------- #
# Variable schema — the clinical vocabulary the DSL may reference.
# --------------------------------------------------------------------------- #
ENUMS: Dict[str, Dict[str, int]] = {
    "risk": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "antibiotic": {"NONE": 0, "Cefepime": 1, "PipTazo": 2, "Vancomycin": 3,
                   "Meropenem": 4, "Ceftriaxone": 5, "Ciprofloxacin": 6,
                   # the cage's code 5; without it an order for Levofloxacin could only
                   # be proven as a DIFFERENT fluoroquinolone
                   "Levofloxacin": 7},
}
# name -> ("real" | "int-enum" | "bool")
VARS: Dict[str, str] = {
    "risk": "int-enum",
    "antibiotic": "int-enum",
    "fluid_rate": "real",
    "vasopressor_dose": "real",
    "antibiotic_time_hr": "real",          # planned time-to-administration (Hour-1)
    "patient.CHF": "bool",
    "patient.penicillin_allergy": "bool",
    "patient.renal_failure": "bool",
}


def _z3_var(name: str):
    t = VARS[name]
    safe = name.replace(".", "_")
    return Real(safe) if t == "real" else Int(safe) if t == "int-enum" else Bool(safe)


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #
_UNICODE = {"→": "->", "≤": "<=", "≥": ">=", "≠": "!=", "∧": "and", "∨": "or", "¬": "not"}
_TOKEN_RE = re.compile(r"""
    (?P<NUMBER>\d+\.\d+|\d+)
  | (?P<ARROW>->)
  | (?P<LE><=) | (?P<GE>>=) | (?P<NE>!=) | (?P<EQ>=)
  | (?P<LT><) | (?P<GT>>)
  | (?P<LP>\() | (?P<RP>\)) | (?P<COLON>:)
  | (?P<NAME>[A-Za-z_][A-Za-z0-9_.]*)
""", re.VERBOSE)
_KEYWORDS = {"RULE", "ALWAYS", "WITHIN", "and", "or", "not", "true", "false"}


@dataclass
class Tok:
    kind: str
    val: str


def lex(text: str) -> List[Tok]:
    for u, a in _UNICODE.items():
        text = text.replace(u, a)
    # strip comments (# ...)
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    toks: List[Tok] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise SyntaxError(f"unexpected character {text[i]!r} at {i}")
        kind = m.lastgroup
        val = m.group()
        if kind == "NAME" and val in _KEYWORDS:
            kind = val.upper() if val in ("RULE", "ALWAYS", "WITHIN") else val
        toks.append(Tok(kind, val))
        i = m.end()
    toks.append(Tok("EOF", ""))
    return toks


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #
class Node: ...
@dataclass
class Rule(Node):
    name: str
    body: Node
    text: str
@dataclass
class Always(Node):
    inner: Node
@dataclass
class Within(Node):
    hours: float
    inner: Node
@dataclass
class BinBool(Node):
    op: str            # and | or | implies
    lhs: Node
    rhs: Node
@dataclass
class NotNode(Node):
    inner: Node
@dataclass
class Cmp(Node):
    op: str            # = != <= >= < >
    lhs: str           # variable name
    rhs: Any           # number or enum-name
@dataclass
class BoolVar(Node):
    name: str


# --------------------------------------------------------------------------- #
# Parser (recursive descent + precedence)
# --------------------------------------------------------------------------- #
class Parser:
    def __init__(self, toks: List[Tok], src: str):
        self.t = toks
        self.i = 0
        self.src = src

    def peek(self) -> Tok:
        return self.t[self.i]

    def eat(self, kind: Optional[str] = None) -> Tok:
        tok = self.t[self.i]
        if kind and tok.kind != kind:
            raise SyntaxError(f"expected {kind}, got {tok.kind} {tok.val!r}")
        self.i += 1
        return tok

    def parse_program(self) -> List[Rule]:
        rules: List[Rule] = []
        while self.peek().kind == "RULE":
            rules.append(self.parse_rule())
        self.eat("EOF")
        if not rules:
            raise SyntaxError("no RULEs found")
        return rules

    def parse_rule(self) -> Rule:
        self.eat("RULE")
        name = self.eat("NAME").val
        self.eat("COLON")
        start = self.i
        body = self.parse_expr()
        text = " ".join(t.val for t in self.t[start:self.i])
        return Rule(name, body, text)

    # implication (lowest precedence, right-assoc)
    def parse_expr(self) -> Node:
        lhs = self.parse_or()
        if self.peek().kind == "ARROW":
            self.eat("ARROW")
            return BinBool("implies", lhs, self.parse_expr())
        return lhs

    def parse_or(self) -> Node:
        lhs = self.parse_and()
        while self.peek().kind == "or":
            self.eat("or")
            lhs = BinBool("or", lhs, self.parse_and())
        return lhs

    def parse_and(self) -> Node:
        lhs = self.parse_unary()
        while self.peek().kind == "and":
            self.eat("and")
            lhs = BinBool("and", lhs, self.parse_unary())
        return lhs

    def parse_unary(self) -> Node:
        k = self.peek().kind
        if k == "not":
            self.eat("not")
            return NotNode(self.parse_unary())
        if k == "ALWAYS":
            self.eat("ALWAYS")
            self.eat("LP")
            inner = self.parse_expr()
            self.eat("RP")
            return Always(inner)
        if k == "WITHIN":
            self.eat("WITHIN")
            num = float(self.eat("NUMBER").val)
            unit = self.eat("NAME").val.lower()       # hr | h | min
            hours = num / 60.0 if unit.startswith("min") else num
            return Within(hours, self.parse_unary())
        if k == "LP":
            self.eat("LP")
            inner = self.parse_expr()
            self.eat("RP")
            return inner
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self.eat("NAME")
        name = tok.val
        # comparison?
        if self.peek().kind in ("EQ", "NE", "LE", "GE", "LT", "GT"):
            op = {"EQ": "=", "NE": "!=", "LE": "<=", "GE": ">=", "LT": "<", "GT": ">"}[
                self.eat().kind]
            rt = self.eat()
            rhs: Any = float(rt.val) if rt.kind == "NUMBER" else rt.val
            if name not in VARS:
                raise SyntaxError(f"unknown variable {name!r}")
            return Cmp(op, name, rhs)
        # bare boolean variable
        if name not in VARS or VARS[name] != "bool":
            raise SyntaxError(f"{name!r} is not a boolean variable; needs a comparison")
        return BoolVar(name)


def parse(text: str) -> List[Rule]:
    return Parser(lex(text), text).parse_program()


# --------------------------------------------------------------------------- #
# Compiler: AST -> Z3 BoolRef
# --------------------------------------------------------------------------- #
def _resolve_value(varname: str, raw: Any):
    t = VARS[varname]
    if t == "int-enum":
        if isinstance(raw, str):
            enum = ENUMS[varname]
            if raw not in enum:
                raise SyntaxError(f"{raw!r} is not a valid {varname} value {list(enum)}")
            return IntVal(enum[raw])
        return IntVal(int(raw))
    if t == "real":
        return RealVal(float(raw))
    raise SyntaxError(f"cannot compare boolean variable {varname!r} to a value")


@_z3_guarded
def compile_node(n: Node) -> BoolRef:
    # Guarded: building terms touches Z3's shared, non-thread-safe context. This
    # was unguarded — compiling a policy on a request thread while the ward loop
    # solved corrupted the context ("UNEXPECTED CODE WAS REACHED") and the prover
    # failed closed for the rest of the process. Re-entrant, so the recursion and
    # the guarded callers above cost only lock traffic.
    if isinstance(n, Always):
        return compile_node(n.inner)                  # bounded to the decision point
    if isinstance(n, Within):
        # ψ must hold AND the planned administration time is within the bound.
        return And(compile_node(n.inner), _z3_var("antibiotic_time_hr") <= RealVal(n.hours))
    if isinstance(n, BinBool):
        l, r = compile_node(n.lhs), compile_node(n.rhs)
        return {"and": And, "or": Or, "implies": Implies}[n.op](l, r)
    if isinstance(n, NotNode):
        return Not(compile_node(n.inner))
    if isinstance(n, BoolVar):
        return _z3_var(n.name) == BoolVal(True)
    if isinstance(n, Cmp):
        v = _z3_var(n.lhs)
        rhs = _resolve_value(n.lhs, n.rhs)
        return {"=": v == rhs, "!=": v != rhs, "<=": v <= rhs,
                ">=": v >= rhs, "<": v < rhs, ">": v > rhs}[n.op]
    raise TypeError(f"cannot compile node {n!r}")


@dataclass
class CompiledRule:
    name: str
    text: str
    formula: BoolRef

    @_z3_guarded
    def sexpr(self) -> str:
        return self.formula.sexpr()                   # printing a term reads the context


@_z3_guarded
def compile_rules(text: str) -> List[CompiledRule]:
    return [CompiledRule(r.name, r.text, compile_node(r.body)) for r in parse(text)]


# --------------------------------------------------------------------------- #
# Binding a concrete protocol + patient context to Z3 constraints.
# --------------------------------------------------------------------------- #
def _bindings(protocol: Dict[str, Any], context: Dict[str, Any]) -> List[BoolRef]:
    b: List[BoolRef] = []
    risk_map = ENUMS["risk"]
    if "risk_level" in context:
        b.append(_z3_var("risk") == IntVal(risk_map.get(str(context["risk_level"]).upper(), 0)))
    if "antibiotic_code" in protocol:
        b.append(_z3_var("antibiotic") == IntVal(int(protocol["antibiotic_code"])))
    for k in ("fluid_rate", "vasopressor_dose", "antibiotic_time_hr"):
        if k in protocol and protocol[k] is not None:
            b.append(_z3_var(k) == RealVal(float(protocol[k])))
    for ctxk, var in (("chf", "patient.CHF"),
                      ("penicillin_allergy", "patient.penicillin_allergy"),
                      ("renal_failure", "patient.renal_failure")):
        if ctxk in context:
            b.append(_z3_var(var) == BoolVal(bool(context[ctxk])))
    return b


# --------------------------------------------------------------------------- #
# Proof certificate + the verifier.
# --------------------------------------------------------------------------- #
@dataclass
class RuleCertificate:
    rule: str
    text: str
    formula: str                  # z3 s-expression (machine-readable, auditable)
    z3_status: str                # "unsat" = proven satisfied | "sat" = violated
    satisfied: bool
    counterexample: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "text": self.text, "formula": self.formula,
                "z3_status": self.z3_status, "satisfied": self.satisfied,
                "counterexample": self.counterexample}


@dataclass
class PolicyCertificate:
    ok: bool
    rules: List[RuleCertificate]
    n_rules: int
    n_satisfied: int

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "n_rules": self.n_rules, "n_satisfied": self.n_satisfied,
                "rules": [r.to_dict() for r in self.rules]}


@_z3_guarded
def verify_protocol(rules: List[CompiledRule], protocol: Dict[str, Any],
                    context: Dict[str, Any]) -> PolicyCertificate:
    """Prove a concrete protocol satisfies every compiled rule for this patient.

    For each rule we assert the bindings and the rule's *negation*; ``unsat`` means
    the protocol provably satisfies the rule. ``sat`` returns a counterexample."""
    binds = _bindings(protocol, context)
    certs: List[RuleCertificate] = []
    for r in rules:
        s = Solver()
        s.add(*binds)
        s.add(Not(r.formula))
        status = s.check()
        if status == unsat:
            certs.append(RuleCertificate(r.name, r.text, r.sexpr(), "unsat", True))
        else:
            m = s.model()
            ce = {str(d.name()): str(m[d]) for d in m.decls()}
            certs.append(RuleCertificate(r.name, r.text, r.sexpr(), "sat", False, ce))
    n_ok = sum(c.satisfied for c in certs)
    return PolicyCertificate(n_ok == len(certs), certs, len(certs), n_ok)


@_z3_guarded
def check_consistency(rules: List[CompiledRule],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Is the rule set jointly satisfiable for this patient? (Detects a
    self-contradictory policy — rules no protocol can satisfy at once.)"""
    s = Solver()
    if context:
        s.add(*_bindings({}, context))
    for r in rules:
        s.add(r.formula)
    status = s.check()
    out: Dict[str, Any] = {"consistent": status == sat,
                           "z3_status": "sat" if status == sat else "unsat"}
    if status == sat:
        m = s.model()
        out["example_protocol"] = {str(d.name()): str(m[d]) for d in m.decls()}
    return out


# --------------------------------------------------------------------------- #
# A default hospital policy (authored in the DSL).
# --------------------------------------------------------------------------- #
DEFAULT_POLICY = """
# Hour-1 sepsis bundle: high risk demands timely antibiotics.
RULE sepsis_antibiotic:
    ALWAYS (risk = HIGH -> WITHIN 1hr antibiotic != NONE)

# Penicillin allergy: never piperacillin-tazobactam.
RULE allergy_guard:
    ALWAYS (patient.penicillin_allergy -> antibiotic != PipTazo)

# Congestive heart failure: cap fluid resuscitation.
RULE fluid_cap:
    ALWAYS (patient.CHF -> fluid_rate <= 5.0)

# Renal failure: bound vasopressor dose.
RULE renal_vaso_cap:
    ALWAYS (patient.renal_failure -> vasopressor_dose <= 0.2)
"""


def _demo() -> None:
    rules = compile_rules(DEFAULT_POLICY)
    print(f"compiled {len(rules)} rules")
    safe = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5, "fluid_rate": 4.0,
            "vasopressor_dose": 0.1}
    ctx = {"risk_level": "HIGH", "chf": True, "penicillin_allergy": True, "renal_failure": False}
    cert = verify_protocol(rules, safe, ctx)
    print(f"safe protocol ok={cert.ok} ({cert.n_satisfied}/{cert.n_rules})")

    bad = {"antibiotic_code": 2, "antibiotic_time_hr": 3.0, "fluid_rate": 30.0}
    cert2 = verify_protocol(rules, bad, ctx)
    print(f"bad protocol ok={cert2.ok} ({cert2.n_satisfied}/{cert2.n_rules})")
    for c in cert2.rules:
        if not c.satisfied:
            print(f"  VIOLATED {c.rule}: {c.text}")
    print("consistency:", check_consistency(rules, ctx)["consistent"])


if __name__ == "__main__":
    _demo()
