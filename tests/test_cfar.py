"""Tests for the CA-CFAR vessel detector."""

import numpy as np
import pytest

from portvolume.detection.cfar import ca_cfar_2d


class TestCACFAR:
    def test_detects_known_targets(self, synthetic_sar_linear, cfar_params):
        """CFAR should detect all 5 bright targets in synthetic image."""
        image, target_positions = synthetic_sar_linear

        mask = ca_cfar_2d(
            image,
            guard_cells=cfar_params.guard_cells,
            background_cells=cfar_params.background_cells,
            pfa=cfar_params.pfa,
        )

        # Each target should have at least some detected pixels
        for row, col in target_positions:
            region = mask[row - 3 : row + 4, col - 3 : col + 4]
            assert region.any(), f"Target at ({row}, {col}) not detected"

    def test_no_false_alarms_on_noise(self, cfar_params):
        """Pure background noise should produce very few false alarms."""
        rng = np.random.default_rng(123)
        noise = rng.exponential(scale=0.01, size=(256, 256))

        mask = ca_cfar_2d(
            noise,
            guard_cells=cfar_params.guard_cells,
            background_cells=cfar_params.background_cells,
            pfa=cfar_params.pfa,
        )

        # With Pfa = 1e-6 and ~65000 pixels, expected false alarms < 1
        false_alarm_count = mask.sum()
        assert false_alarm_count < 10, (
            f"Too many false alarms on pure noise: {false_alarm_count}"
        )

    def test_nan_handling(self, cfar_params):
        """NaN pixels (land mask) should not produce detections."""
        rng = np.random.default_rng(42)
        image = rng.exponential(scale=0.01, size=(256, 256))

        # Set a large region to NaN
        image[50:150, 50:150] = np.nan

        mask = ca_cfar_2d(
            image,
            guard_cells=cfar_params.guard_cells,
            background_cells=cfar_params.background_cells,
            pfa=cfar_params.pfa,
        )

        # No detections in NaN region
        assert not mask[50:150, 50:150].any(), "Detections found in NaN region"

    def test_parameter_sensitivity(self, synthetic_sar_linear):
        """Strong targets should be detected regardless of parameter variation."""
        image, target_positions = synthetic_sar_linear

        for guard in [3, 5, 8]:
            for bg in [10, 15, 20]:
                mask = ca_cfar_2d(image, guard_cells=guard, background_cells=bg, pfa=1e-5)
                detected = 0
                for row, col in target_positions:
                    region = mask[row - 3 : row + 4, col - 3 : col + 4]
                    if region.any():
                        detected += 1
                assert detected >= 4, (
                    f"Only {detected}/5 targets detected with guard={guard}, bg={bg}"
                )

    def test_invalid_window_raises(self):
        """Should raise ValueError for invalid window configuration."""
        image = np.ones((100, 100))
        with pytest.raises(ValueError, match="Invalid CFAR window"):
            ca_cfar_2d(image, guard_cells=50, background_cells=0, pfa=1e-6)
