import unittest
import pandas as pd
import numpy as np
from saemix.data import saemix_data


class TestSaemixData(unittest.TestCase):
    def test_basic_creation(self):
        data = pd.DataFrame({
            'Id': [1, 1, 2, 2],
            'Time': [0, 1, 0, 1],
            'Concentration': [10, 5, 12, 6]
        })
        
        saemix_data_obj = saemix_data(
            name_data=data,
            name_group='Id',
            name_predictors=['Time'],
            name_response='Concentration'
        )
        
        self.assertEqual(saemix_data_obj.n_subjects, 2)
        self.assertEqual(saemix_data_obj.n_total_obs, 4)
        self.assertTrue('index' in saemix_data_obj.data.columns)
    
    def test_index_mapping(self):
        data = pd.DataFrame({
            'Id': [10, 10, 20, 20],
            'Time': [0, 1, 0, 1],
            'Concentration': [10, 5, 12, 6]
        })
        
        saemix_data_obj = saemix_data(
            name_data=data,
            name_group='Id',
            name_predictors=['Time'],
            name_response='Concentration'
        )
        
        unique_indices = np.unique(saemix_data_obj.data['index'].values)
        self.assertTrue(np.array_equal(unique_indices, [0, 1]))


if __name__ == '__main__':
    unittest.main()