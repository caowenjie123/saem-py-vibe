import unittest
import numpy as np
from saemix.utils import transphi, transpsi


class TestUtils(unittest.TestCase):
    def test_transphi_log_normal(self):
        phi = np.array([[0.0, 1.0]])
        tr = np.array([1, 0])
        psi = transphi(phi, tr)
        np.testing.assert_array_almost_equal(psi, np.array([[1.0, 1.0]]))
    
    def test_transpsi_log_normal(self):
        psi = np.array([[1.0, 2.0]])
        tr = np.array([1, 0])
        phi = transpsi(psi, tr)
        np.testing.assert_array_almost_equal(phi, np.array([[0.0, 2.0]]))


if __name__ == '__main__':
    unittest.main()