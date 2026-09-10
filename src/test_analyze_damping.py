import unittest

import numpy as np

from analyze_damping import jacobian, modes, spectral_metrics


class DampingAnalysisTests(unittest.TestCase):
    def test_jacobian_and_sampled_mode(self):
        damping, frequency = -2.0, 5.0
        matrix = np.exp(damping * .005) * np.array([
            [np.cos(frequency * .005), -np.sin(frequency * .005)],
            [np.sin(frequency * .005), np.cos(frequency * .005)],
        ])
        np.testing.assert_allclose(jacobian(lambda state: matrix @ state, np.ones(2), 1e-5), matrix)
        pole = modes(matrix)[0]
        self.assertAlmostEqual(pole["continuous_real"], damping)
        self.assertAlmostEqual(pole["damped_frequency_Hz"], frequency / (2 * np.pi))
        self.assertAlmostEqual(pole["damping_ratio"], 2 / np.sqrt(29))

    def test_absolute_spectral_amplitudes(self):
        time = np.arange(1000) * .005
        signal = .1 * np.sin(2 * np.pi * 10 * time) + .02 * np.sin(2 * np.pi * 40 * time)
        metrics = spectral_metrics(signal, .005)
        self.assertAlmostEqual(metrics["dominant_frequency_Hz"], 10)
        self.assertAlmostEqual(metrics["band_rms_A"]["3_to_25Hz"], .1 / np.sqrt(2))
        self.assertAlmostEqual(metrics["band_rms_A"]["25_to_100Hz"], .02 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
