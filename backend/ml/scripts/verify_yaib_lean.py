"""Checks for the lean YAIB baseline driver that need neither YAIB nor data.

The real gate is deploy/physionet/yaib_lean_check.sh (stock vs lean on YAIB's demo,
byte-identical predictions). This only pins the argument handling.

python -m backend.ml.scripts.verify_yaib_lean
"""
from __future__ import annotations

import unittest

from backend.ml.scripts.yaib_lean_baseline import RAM_CACHE_OFF, release, with_ram_cache_off


class LeanArguments(unittest.TestCase):
    def test_binding_joins_an_existing_hp_list(self):
        # argparse keeps only the last -hp, so a second -hp would drop the probe's run subset
        args = ["train", "-m", "LGBMClassifier", "-hp", "a=1", "b=2", "--seed", "1"]
        out = with_ram_cache_off(args)
        self.assertEqual(out.count("-hp"), 1)
        self.assertEqual(out[out.index("-hp") + 1: out.index("--seed")], [RAM_CACHE_OFF, "a=1", "b=2"])

    def test_binding_is_added_when_there_is_no_hp(self):
        self.assertEqual(with_ram_cache_off(["train"])[-2:], ["-hp", RAM_CACHE_OFF])

    def test_long_flag_is_recognised(self):
        out = with_ram_cache_off(["train", "--hyperparams", "a=1"])
        self.assertEqual(out, ["train", "--hyperparams", RAM_CACHE_OFF, "a=1"])

    def test_a_caller_cannot_turn_the_item_cache_back_on(self):
        with self.assertRaises(ValueError):
            with_ram_cache_off(["train", "-hp", "train_common.ram_cache=True"])


class LeanRelease(unittest.TestCase):
    def test_release_empties_the_callers_dict(self):
        # execute_repeated_cv keeps its own name for the dict: only an in-place clear frees the frames
        fold = {"train": {"FEATURES": object(), "OUTCOME": object()}, "val": {"FEATURES": object()}}
        held_by_caller, train = fold, fold["train"]
        release(fold)
        self.assertEqual(held_by_caller, {})
        self.assertEqual(train, {})

    def test_release_of_nothing_is_a_no_op(self):
        release(None)
        release({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
