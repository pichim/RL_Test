import unittest

from feedback_swingup import summarize, transition_metrics


class FeedbackSwingupTests(unittest.TestCase):
    def test_pairing_and_failures(self):
        episodes = [{"suite": "test", "controller": label, "unsafe": int(label == "modified"),
                     "centered_success_5s": 0, "centered_success_end": int(label == "original"),
                     "centered_lost_after_5s": 0, "completed_horizon": int(label == "original"),
                     "centered_finish_s": 8 if label == "original" else None} for label in ("original", "modified")]
        summary = summarize(episodes, [])
        self.assertEqual(summary["test"]["modified"]["unsafe"], 1)
        self.assertEqual(summary["test"]["original"]["mean_finish_s"], 8)
        self.assertEqual(summary["test"]["modified"]["windows"]["first_5s"]["paired_cases"], 0)
        self.assertIsNone(summary["test"]["modified"]["windows"]["first_5s"]["current_rms_A"])

    def test_activity_measurements(self):
        result = transition_metrics([{"correction_action": value} for value in (0, .1, .2, 0)])
        self.assertEqual(result["correction_activity_transitions"], 2)
        self.assertEqual(result["correction_active_steps"], 2)
        self.assertEqual(result["max_correction_step_action"], .2)


if __name__ == "__main__":
    unittest.main()
