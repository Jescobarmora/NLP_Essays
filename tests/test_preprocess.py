import unittest
from src.preprocess import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextPreprocessor()
        self.sample_text = "This is a sample essay with some Stopwords and punctuation!"

    def test_preprocess(self):
        processed = self.processor.preprocess(self.sample_text)
        self.assertIsInstance(processed, list)
        
        # Verificar que no haya stopwords
        stop_words = set(['this', 'is', 'a', 'with', 'some'])
        self.assertFalse(any(word in stop_words for word in processed))
        
        # Verificar que todos los tokens sean alfabéticos y lematizados
        for word in processed:
            self.assertTrue(word.isalpha())
