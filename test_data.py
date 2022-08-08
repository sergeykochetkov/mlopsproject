from data import load_dataset
import unittest


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.ticker = 'MSFT'
        self.length = 10
        self._d, self._x, self._y = load_dataset(self.ticker, self.length, period='1y', interval='1h')

    def test_dimensions(self):
        self.assertEqual(len(self._d), len(self._x))
        self.assertEqual(len(self._y), len(self._x))
        self.assertEqual(self._x.shape[1], self.length)
        self.assertEqual(len(self._y.shape), 1)

    def test_uniqueness(self):
        self.assertEqual(len(self._d), len(set(self._d)))
