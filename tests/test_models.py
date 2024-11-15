import unittest
from src.models.lda_model import train_lda_model, get_topic_probabilities
from src.models.bert_model import get_bert_embeddings
from src.models.fasttext_model import prepare_fasttext_data, train_fasttext_model
from src.models.evaluator import calculate_accuracy, calculate_roc_auc
from src.data_loader import load_data
from src.preprocess import TextPreprocessor
from sklearn.model_selection import train_test_split
from gensim import corpora

class TestModels(unittest.TestCase):
    def setUp(self):
        # Cargar un subconjunto de datos para pruebas
        data = load_data('data/training_set_rel3.tsv').head(100)
        self.preprocessor = TextPreprocessor()
        data['processed_essay'] = data['essay'].apply(self.preprocessor.preprocess)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data['processed_essay'], data['topic'], test_size=0.2, random_state=42
        )
        
        self.dictionary = corpora.Dictionary(self.X_train)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.X_train]

    def test_lda_model(self):
        lda_model = train_lda_model(self.corpus, self.dictionary)
        self.assertIsNotNone(lda_model)
        self.assertEqual(lda_model.num_topics, 8)

    def test_bert_embeddings(self):
        embeddings = get_bert_embeddings(self.X_train.head(5))
        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.shape[1], 768)  # Tamaño típico de BERT embeddings

    def test_fasttext_model(self):
        prepare_fasttext_data(self.X_train, self.y_train, 'train_test.txt')
        model = train_fasttext_model('train_test.txt')
        self.assertIsNotNone(model)
        
        labels, probs = model.predict([' '.join(text) for text in self.X_test.head(5)], k=1)
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(probs), 5)
