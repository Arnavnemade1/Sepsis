"""Run a YAIB baseline exactly as YAIB runs it, and keep its per-hour test predictions.

YAIB writes ``pred_indicators.csv`` (stay id, hour, label, p0, p1) only when the model
wrapper's class attribute ``BaseModule.debug`` is true (``MLWrapper.test_step``). That
attribute is separate from the ``--debug`` command-line flag, which also subsamples the
data to 1%, so the flag cannot be used. This driver sets only the attribute and then
calls YAIB's own ``main`` with the arguments given: the data, splits, preprocessing,
model and metric are YAIB's.

Run it in an environment where YAIB is installed, from YAIB's checkout (it reads its
``configs/`` relative to the working directory)::

    python -m backend.ml.yaib.run_baseline train -d <cohort> -n <name> \\
        -t BinaryClassification -tn Sepsis -m LGBMClassifier --seed 1234 -l <logs>
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    import icu_benchmarks.models.wrappers as W
    from icu_benchmarks.run import main as yaib_main

    W.BaseModule.debug = True          # save per-hour test predictions; nothing else
    yaib_main(tuple(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
