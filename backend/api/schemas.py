"""Request bodies for the AegisOS REST surface (one place, shared by every router)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from backend.aegis.clinical_dsl import DEFAULT_POLICY


class TelemetryIn(BaseModel):
    telemetry: Dict[str, float]
    source: str = "wearable"
    camera_vitals: Optional[Dict[str, Any]] = None
    signal_quality: float = 1.0
    # Explicit source contract for AERIS. Omitted means no additional source is
    # assumed available; the advisory router returns WAIT rather than inventing one.
    evidence_capabilities: Optional[Dict[str, bool]] = None


class ReplayIn(BaseModel):
    outcome_id: Optional[str] = None       # replay at the earliest actionable node upstream
    node_id: Optional[str] = None          # …or replay at a specific branch point
    treatment: Dict[str, float] = {"fluid_rate": 5.0, "vasopressor_dose": 0.0,
                                    "antibiotic_code": 3}
    horizon: Optional[int] = None
    bed: Optional[str] = None              # which patient (live session or cohort bed)


class IdentitySwapIn(BaseModel):
    target: Optional[str] = None         # whose monitor
    source: Optional[str] = None         # whose stream is (mis)connected


class PharmaScreenIn(BaseModel):
    drugs: List[str] = ["warfarin", "fluconazole", "ciprofloxacin", "ondansetron"]
    context: Dict[str, Any] = {"renal_failure": False}


class DslVerifyIn(BaseModel):
    policy: str = DEFAULT_POLICY
    protocol: Dict[str, Any] = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                "fluid_rate": 4.0, "vasopressor_dose": 0.1}
    context: Dict[str, Any] = {"risk_level": "HIGH", "chf": True,
                               "penicillin_allergy": True, "renal_failure": False}


class RbacLoginIn(BaseModel):
    user: str = "clinician"
    role: str


class RbacAuthIn(BaseModel):
    token: str
    capability: str


class GovernIn(BaseModel):
    layer: str = "bed"
    action: str                                    # treat | escalate | pin | actuator
    payload: Dict[str, Any] = {}
    target: Optional[str] = None                   # bed id


class ProofVerifyIn(BaseModel):
    certificate: Dict[str, Any]
    require_signature: bool = True


class CoreProofIn(BaseModel):
    clinical_history: Dict[str, Any] = {"chf": True, "renal_failure": False,
                                        "penicillin_allergy": True}
    vitals: Dict[str, Any] = {"map": 62.0}
    risk_level: str = "HIGH"
    protocol: Dict[str, Any] = {"fluid_rate": 4.0, "vasopressor_dose": 0.2,
                                "antibiotic_code": 3}


class ChaosIn(BaseModel):
    node: str
    action: str = "crisis"            # crisis | clear | noise
    intensity: float = 1.0


class PKProjectIn(BaseModel):
    bed: Optional[str] = None                    # derive patient + order from a bed
    protocol: Optional[Dict[str, Any]] = None    # or state the order explicitly
    patient: Optional[Dict[str, Any]] = None     # or the covariates explicitly
    adverse: bool = True                         # also run the side-effect projection
    n_points: int = 121


class DeviceAssignIn(BaseModel):
    bed_id: str
    label: str = ""


class DeviceActionIn(BaseModel):
    action: str                           # calibrate | reboot | ota | wake | decommission


class SyscallIn(BaseModel):
    syscall: str                                  # treat | escalate | crisis | clear
    bed: str
    args: Dict[str, Any] = {}
    caller: str = "console"


class VerifyIn(BaseModel):
    name: Optional[str] = None             # a driver from the catalogue, or…
    source: Optional[str] = None           # …raw eBPF assembly to verify
    win_len: int = 2


class KernelFaultIn(BaseModel):
    slot: str                              # device key ("dev_0"), bay index, or label
    reason: str = "sensor disconnected (injected)"


class PkgIn(BaseModel):
    name: str


class HDKernelRunIn(BaseModel):
    workload: str = "mixed"                 # mixed | pipeline | fanout | stress
    policy: str = "HDSched"                 # FIFO | RR | CFS | EDF | HDSched


class MLTrainIn(BaseModel):
    name: str
    feature_set: str = "wearable"          # wearable | lab
    horizon: Optional[int] = None          # early-warning hours before onset (specialise)
    limit: int = 1000
    folds: int = 3


class MLUseIn(BaseModel):
    name: str


class TrackIn(BaseModel):
    command: str


class HistoryIn(BaseModel):
    text: str = ""
    source: str = "manual"


class LinkIn(BaseModel):
    source: str = "mock_hospital"
    patient_ref: Optional[str] = None        # source-specific id (e.g. FHIR patient id / MRN)


class SynthIn(BaseModel):
    text: str = ""


class LLMAskIn(BaseModel):
    question: str


class DoctorConsultIn(BaseModel):
    question: str
    bed: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    agentic: bool = True                  # may drive the OS to answer


class SpeakIn(BaseModel):
    text: str
    voice: Optional[str] = None


class AgentInvokeIn(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


class VasodilationIn(BaseModel):
    alpha: float                           # 0 = native PPG, 1 = dicrotic notch fully eroded


class ScenarioIn(BaseModel):
    name: str
    arc: str = "septic_shock"
    los_hours: int = 48
    onset_hour: Optional[int] = 30
    ramp_hours: int = 12
    severity: float = 1.0
    age: float = 68.0
    gender: str = "M"
    weight: float = 78.0
    lab_interval_hours: int = 8
    vitals_dropout: float = 0.04
    #: [[hour, kind], ...] with kind in fluids | antibiotics | vasopressors.
    interventions: List[List[Any]] = []
    seed: int = 7
    notes: str = ""
    #: [{"kind": ..., "value": ..., "because": ...}, ...]
    expectations: List[Dict[str, Any]] = []


class KillIn(BaseModel):
    pid: int
    signal: str = "SIGTERM"


class LoginIn(BaseModel):
    username: str
    passphrase: Optional[str] = None


class UnlockIn(BaseModel):
    passphrase: Optional[str] = None


class BreakGlassIn(BaseModel):
    reason: str
    minutes: float = 10.0


class BreakGlassReviewIn(BaseModel):
    id: str
    verdict: str
    note: str = ""


class RevokeIn(BaseModel):
    sid: str


class GateModeIn(BaseModel):
    mode: str


class FsWriteIn(BaseModel):
    path: str
    content: str = ""
    append: bool = False


class FsPathIn(BaseModel):
    path: str



# --- the paging OS ------------------------------------------------------------ #
class PagingAlertIn(BaseModel):
    text: str
    room: Optional[str] = None
    source: str = "nurse"                  # nurse | voice


class PagingRespondIn(BaseModel):
    action: str                            # accept | decline
    reason: Optional[str] = None
    clinician: Optional[str] = None        # only an operator may answer for someone else


class PagingApproveIn(BaseModel):
    clinician: Optional[str] = None        # override the planned assignee


class PagingCancelIn(BaseModel):
    reason: str = ""


class PagingStatusIn(BaseModel):
    status: str                            # available | in_procedure | on_case | off_shift


class PagingSettingsIn(BaseModel):
    patch: Dict[str, Any]


# --- the integration plane ------------------------------------------------------ #
class InteropAdoptIn(BaseModel):
    concept: Optional[str] = None          # the vocabulary key to adopt the code as
    conversion: Optional[Any] = None       # "declared" | "suggested" | "identity" | {"factor", "offset"}


class InteropRejectIn(BaseModel):
    reason: str = ""


class InteropSettingsIn(BaseModel):
    patch: Dict[str, Any]


class InteropDrillIn(BaseModel):
    kind: str                              # see backend/aegis/interop/sender.py DRILLS, or "clear"
    on: bool = True


class InteropMessageIn(BaseModel):
    message: str                           # one HL7 v2 message, ER7 text


class WebhookIn(BaseModel):
    url: str                               # https://... (or internal://echo)
    events: List[str] = []                 # empty = every event type
    description: str = ""
