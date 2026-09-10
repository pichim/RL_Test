from dataclasses import replace
import unittest

from validate_smooth_current import CONFIG, current_metrics, horizon_metrics


class LongValidationTests(unittest.TestCase):
    def test_balance_loss_and_truncation(self):
        config = replace(CONFIG, episode_time=30.0)
        result = {"trace": [{"balanced": 1, "unsafe": 0} for _ in range(30001)]}
        self.assertEqual(horizon_metrics(result, config), {
            "success_at_5s": 1, "unsafe_by_5s": 0,
            "balance_lost_after_5s": 0, "completed_30s": 1,
        })
        result["trace"][6000]["balanced"] = 0
        self.assertEqual(horizon_metrics(result, config)["balance_lost_after_5s"], 1)
        result["trace"] = result["trace"][:4500]
        result["trace"][-1]["unsafe"] = 1
        metrics = horizon_metrics(result, config)
        self.assertEqual(metrics["success_at_5s"], 0)
        self.assertEqual(metrics["unsafe_by_5s"], 1)
        self.assertEqual(metrics["completed_30s"], 0)

    def test_late_window_keeps_preceding_command(self):
        result = {"trace": [{"command_current_A": 0.2} for _ in range(30001)]}
        late = current_metrics(result, CONFIG, 25, 30)
        self.assertEqual(late["command_samples"], 1000)
        self.assertEqual(late["change_rms_A"], 0.0)
        self.assertAlmostEqual(current_metrics(result, CONFIG, 0, 5)["total_variation_A"], 0.2)
        result["trace"] = result["trace"][:29000]
        self.assertIsNone(current_metrics(result, CONFIG, 25, 30))


if __name__ == "__main__":
    unittest.main()
