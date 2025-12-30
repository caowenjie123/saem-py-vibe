import unittest
import numpy as np
import pandas as pd
from saemix import saemix, saemix_data, saemix_model, saemix_control


def linear_model(psi, id, xidep):
    """
    简单的线性模型: y = psi[id, 0] * x + psi[id, 1]
    """
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


class TestIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
        n_subjects = 5
        n_obs_per_subject = 4
        
        data_list = []
        for i in range(n_subjects):
            x = np.linspace(0, 3, n_obs_per_subject)
            true_a = 2.0 + np.random.normal(0, 0.3)
            true_b = 1.0 + np.random.normal(0, 0.2)
            y = true_a * x + true_b + np.random.normal(0, 0.1, n_obs_per_subject)
            
            for j in range(n_obs_per_subject):
                data_list.append({
                    'Id': i + 1,
                    'X': x[j],
                    'Y': y[j]
                })
        
        self.data = pd.DataFrame(data_list)
        
        self.model = saemix_model(
            model=linear_model,
            psi0=np.array([[2.0, 1.0]]),
            description="Linear model test",
            name_modpar=["a", "b"]
        )
        
        self.saemix_data = saemix_data(
            name_data=self.data,
            name_group='Id',
            name_predictors=['X'],
            name_response='Y'
        )
    
    def test_basic_fit(self):
        control = saemix_control(nbiter_saemix=(10, 5), display_progress=False, warnings=False)
        
        result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control
        )
        
        self.assertIsNotNone(result.results.mean_phi)
        self.assertIsNotNone(result.results.omega)
        self.assertEqual(result.results.mean_phi.shape[0], self.saemix_data.n_subjects)
        self.assertEqual(result.results.mean_phi.shape[1], self.model.n_parameters)
    
    def test_map_estimation(self):
        control = saemix_control(nbiter_saemix=(10, 5), map=True, display_progress=False, warnings=False)
        
        result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control
        )
        
        self.assertIsNotNone(result.results.map_phi)
        self.assertIsNotNone(result.results.map_psi)
        self.assertEqual(result.results.map_phi.shape[0], self.saemix_data.n_subjects)
        self.assertEqual(result.results.map_phi.shape[1], self.model.n_parameters)
    
    def test_predict(self):
        control = saemix_control(nbiter_saemix=(10, 5), map=True, display_progress=False, warnings=False)
        
        result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control
        )
        
        ppred = result.predict(type="ppred")
        self.assertIsNotNone(ppred)
        self.assertEqual(len(ppred), len(self.data))
        
        ipred = result.predict(type="ipred")
        self.assertIsNotNone(ipred)
        self.assertEqual(len(ipred), len(self.data))
        
        np.testing.assert_array_almost_equal(ppred.shape, (len(self.data),))
        np.testing.assert_array_almost_equal(ipred.shape, (len(self.data),))
    
    def test_fim(self):
        control = saemix_control(nbiter_saemix=(10, 5), fim=True, display_progress=False, warnings=False)
        
        result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control
        )
        
        self.assertIsNotNone(result.results.fim)
        self.assertIsInstance(result.results.fim, np.ndarray)
    
    def test_psi_phi_eta_methods(self):
        control = saemix_control(nbiter_saemix=(10, 5), map=True, display_progress=False, warnings=False)
        
        result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control
        )
        
        psi_mode = result.psi(type="mode")
        self.assertIsNotNone(psi_mode)
        
        phi_mode = result.phi(type="mode")
        self.assertIsNotNone(phi_mode)
        
        psi_mean = result.psi(type="mean")
        self.assertIsNotNone(psi_mean)


if __name__ == '__main__':
    unittest.main()