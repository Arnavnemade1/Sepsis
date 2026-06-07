"""Physiological signal analysis shared across the camera and clinical paths.

The headline module is ``hrv``: beat detection (ECG R-peaks and PPG systolic
peaks), RR-interval cleaning, and a full standard HRV feature set (time-domain,
frequency-domain, and nonlinear — including the HeRO sample-entropy/asymmetry
descriptors that are clinically validated for early sepsis).

This is the bridge that lets us validate the camera path honestly: the same HRV
features can be computed from a camera's rPPG pulse, a wearable's PPG, or a
gold-standard ECG, so we can measure how faithfully one reproduces another on
real human recordings (see ``backend.physio.data.bidmc`` + the validation
script).
"""
