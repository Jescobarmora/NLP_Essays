import unittest
import os
import pandas as pd
from src.data_loader import load_data
from src.pipeline import TextPipeline

class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline(self):
        # Construir la ruta al archivo de datos
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training_set_rel3.tsv')
        data = load_data(file_path)
        pipeline = TextPipeline(data)
        pipeline.run_pipeline()
        
        # Verificar que el archivo CSV se ha creado
        self.assertTrue(os.path.exists('roc_auc_scores.csv'))
        
        # Verificar contenido del CSV
        results = pd.read_csv('roc_auc_scores.csv')
        self.assertIsNotNone(results)
        self.assertListEqual(list(results.columns), ['Method', 'Accuracy', 'ROC AUC'])
        
        # Verificar que los ROC AUC están ordenados de mayor a menor
        self.assertTrue(results['ROC AUC'].is_monotonic_decreasing)
