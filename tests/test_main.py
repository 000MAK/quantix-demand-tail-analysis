import unittest
from main import empirical_quantile, exceedances, pot_quantile

class TailTests(unittest.TestCase):
    def test_exceedances_are_positive_excesses(self): self.assertEqual(exceedances([1, 2, 5], 2), [3.0])
    def test_linear_quantile_interpolation(self): self.assertAlmostEqual(empirical_quantile([0, 10], .25), 2.5)
    def test_empty_quantile_rejected(self):
        with self.assertRaises(ValueError): empirical_quantile([], .5)
    def test_exponential_tail_limit(self): self.assertAlmostEqual(pot_quantile(10, 0, 2, .1, .99), 10 + 2 * __import__('math').log(10))

if __name__ == "__main__": unittest.main()
