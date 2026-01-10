import unittest
import numpy as np
from saemix.model import saemix_model


def simple_model(psi, id, xidep):
    return xidep[:, 0] * psi[id, 0]


class TestSaemixModel(unittest.TestCase):
    def test_basic_creation(self):
        model = saemix_model(
            model=simple_model,
            psi0=np.array([[1.0, 2.0]]),
            description="Simple model"
        )
        
        self.assertEqual(model.n_parameters, 2)
        self.assertEqual(model.modeltype, "structural")
    
    def test_model_validation(self):
        def invalid_model(psi, id):
            return psi[id, 0]
        
        with self.assertRaises(ValueError):
            saemix_model(
                model=invalid_model,
                psi0=np.array([[1.0]])
            )


if __name__ == '__main__':
    unittest.main()