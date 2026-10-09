"""
test_drift.py — Unit tests for drift detection and transience classification
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import DriftDetectorConfig
from drift_detector import (
    ADWIN,
    PageHinkley,
    DDM,
    EnsembleDriftDetector,
    TransienceClassifier,
    LayerDivergenceTracker,
    ClientDriftDetector,
    DriftSignal,
    DriftType,
)


class TestDriftDetection(unittest.TestCase):

    def test_adwin_detection(self):
        adwin = ADWIN(delta=0.01)
        # Stationary baseline
        for _ in range(25):
            adwin.update(0.5 + float(np.random.normal(0, 0.02)))
        self.assertFalse(adwin.is_change_detected)

        # Abrupt jump in mean
        detected = False
        for _ in range(15):
            if adwin.update(3.5 + float(np.random.normal(0, 0.05))):
                detected = True
                break
        self.assertTrue(detected, "ADWIN must detect abrupt mean shift")

    def test_page_hinkley_detection(self):
        ph = PageHinkley(threshold=2.0, alpha=0.01)
        # Stationary sequence
        for _ in range(20):
            ph.update(0.4)
        self.assertFalse(ph.is_change_detected)

        # Cumulative increase
        detected = False
        for _ in range(15):
            if ph.update(2.5):
                detected = True
                break
        self.assertTrue(detected, "Page-Hinkley must detect monotonic loss accumulation")

    def test_ddm_detection(self):
        ddm = DDM(min_num_instances=3)
        # Low error baseline
        for _ in range(10):
            ddm.update(0.05)
        # Sudden error surge
        drift_found, _ = ddm.update(0.85)
        self.assertTrue(drift_found or ddm.update(0.90)[0], "DDM must detect high error rate surge")

    def test_transience_classifier_transient(self):
        tc = TransienceClassifier(transient_window=4, persistence_threshold=0.6)
        # Pulse of 1 drift round then reverting
        tc.observe(True, 1)
        tc.observe(False, 2)
        tc.observe(False, 3)
        res = tc.observe(False, 4)
        self.assertEqual(res, DriftType.TRANSIENT, "Short pulse must be classified as TRANSIENT")

    def test_transience_classifier_persistent(self):
        tc = TransienceClassifier(transient_window=4, persistence_threshold=0.6)
        # Sustained drift across all rounds
        tc.observe(True, 1)
        tc.observe(True, 2)
        tc.observe(True, 3)
        res = tc.observe(True, 4)
        self.assertEqual(res, DriftType.PERSISTENT, "Sustained alarms must be classified as PERSISTENT")

    def test_layer_divergence_tracker(self):
        layer_groups = {"base": [0, 1], "head": [2, 3]}
        tracker = LayerDivergenceTracker(layer_groups=layer_groups, divergence_threshold=0.3)
        
        # Baseline rounds
        for _ in range(5):
            tracker.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})

        # Head layers diverge significantly, base remains invariant
        tracker.update({0: 1.02, 1: 0.98, 2: 3.5, 3: 4.2})
        affected_groups = tracker.get_drift_affected_groups()
        self.assertIn("head", affected_groups)
        self.assertNotIn("base", affected_groups)

    def test_ensemble_detector(self):
        cfg = DriftDetectorConfig()
        cfg.ph_threshold = 2.0
        ensemble = EnsembleDriftDetector(cfg)
        
        # Normal training loss sequence
        for _ in range(5):
            sig, _ = ensemble.update(0.4, 0.05)
        self.assertEqual(sig, DriftSignal.STABLE)
        
        # Loss surge
        sig_drift, diag = ensemble.update(2.5, 0.70)
        self.assertEqual(sig_drift, DriftSignal.DRIFT)


if __name__ == "__main__":
    unittest.main()
