import unittest
import numpy as np
from analyze_brca1_screen import region_mean, choose_candidates

class AnalysisTests(unittest.TestCase):
    def test_region_coverage(self):
        a = np.arange(100, dtype=np.float32)
        self.assertEqual(region_mean(a, 20), 20)
        self.assertEqual(region_mean(a, 79), 79)
        for c in (-1, 19, 80, 100):
            self.assertIsNone(region_mean(a, c))

    def test_missing_site_is_not_a_zero_effect(self):
        rows = [dict(run_id=str(i), mean_relative_l2=float(i),
                     nearest_site_mean_relative_l2=None if i==3 else float(i),
                     screen_group='g', nearest_site_position=100,
                     position_grch38_1based=101 if i else 102) for i in range(4)]
        selected = choose_candidates(rows, 1)
        reasons = {r['run_id']:r['selection_reason'] for r in selected}
        self.assertIn('top_global', reasons['3'])
        self.assertIn('top_nearest_site', reasons['2'])
        self.assertNotIn('top_nearest_site', reasons['3'])
        self.assertEqual(len(reasons), len(selected))
        self.assertTrue(any('comparison_for:' in r for r in reasons.values()))

if __name__ == '__main__':
    unittest.main()
