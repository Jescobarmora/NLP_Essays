from .models.lda_model import train_lda_model, get_topic_probabilities
from .models.bert_model import get_bert_embeddings
from .models.fasttext_model import prepare_fasttext_data, train_fasttext_model
from .models.evaluator import calculate_accuracy, calculate_roc_auc
from .preprocess import TextPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from gensim import corpora, models
from sklearn.preprocessing import label_binarize

class TextPipeline:
    def __init__(self, data):
        self.data = data
        self.preprocessor = TextPreprocessor()
        self.num_topics = 8
        self.results = None
    
    def preprocess_data(self):
        self.data['processed_essay'] = self.data['essay'].apply(self.preprocessor.preprocess)
    
    def encode_labels(self):
        self.label_encoder = LabelEncoder()
        self.data['topic_encoded'] = self.label_encoder.fit_transform(self.data['topic'])
    
    def split_data(self):
        X = self.data['processed_essay']
        y = self.data['topic_encoded']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_lda(self):
        # Crear diccionario y corpus
        self.dictionary = corpora.Dictionary(self.X_train)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.X_train]
        # Entrenar modelo LDA sin TF-IDF
        self.lda_model = train_lda_model(self.corpus, self.dictionary, num_topics=self.num_topics)
        # Entrenar modelo LDA con TF-IDF
        self.tfidf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = self.tfidf[self.corpus]
        self.lda_tfidf_model = train_lda_model(self.corpus_tfidf, self.dictionary, num_topics=self.num_topics)
    
    def evaluate_lda(self):
        # Preparar datos de prueba
        self.corpus_test = [self.dictionary.doc2bow(text) for text in self.X_test]
        self.corpus_test_tfidf = self.tfidf[self.corpus_test]
        # LDA sin TF-IDF
        self.test_topic_probs = get_topic_probabilities(self.lda_model, self.corpus_test, self.num_topics)
        self.lda_accuracy = calculate_accuracy(self.y_test, np.argmax(self.test_topic_probs, axis=1))
        self.lda_roc_auc = calculate_roc_auc(self.y_test, self.test_topic_probs, self.num_topics)
        # LDA con TF-IDF
        self.test_topic_probs_tfidf = get_topic_probabilities(self.lda_tfidf_model, self.corpus_test_tfidf, self.num_topics)
        self.lda_tfidf_accuracy = calculate_accuracy(self.y_test, np.argmax(self.test_topic_probs_tfidf, axis=1))
        self.lda_tfidf_roc_auc = calculate_roc_auc(self.y_test, self.test_topic_probs_tfidf, self.num_topics)
    
    def train_evaluate_bert(self):
        # Obtener embeddings
        self.X_train_embeddings = get_bert_embeddings(self.X_train)
        self.X_test_embeddings = get_bert_embeddings(self.X_test)
        # Entrenar clasificador
        self.bert_classifier = SVC(probability=True, random_state=42)
        self.bert_classifier.fit(self.X_train_embeddings, self.y_train)
        # Evaluación
        y_pred = self.bert_classifier.predict(self.X_test_embeddings)
        self.bert_accuracy = calculate_accuracy(self.y_test, y_pred)
        y_scores = self.bert_classifier.predict_proba(self.X_test_embeddings)
        self.bert_roc_auc = calculate_roc_auc(self.y_test, y_scores, len(self.label_encoder.classes_))
    
    def train_evaluate_fasttext(self):
        prepare_fasttext_data(self.X_train, self.y_train, 'train.txt')
        prepare_fasttext_data(self.X_test, self.y_test, 'test.txt')
        
        self.fasttext_model = train_fasttext_model('train.txt')
        
        result = self.fasttext_model.test('test.txt')
        self.fasttext_accuracy = result[1]
        
        labels_k, probabilities_k = self.fasttext_model.predict([' '.join(text) for text in self.X_test], k=self.num_topics)
        y_test_binarized = label_binarize(self.y_test, classes=np.arange(self.num_topics))
        scores_matrix = np.zeros((len(self.X_test), self.num_topics))
        for i, (label_list, prob_list) in enumerate(zip(labels_k, probabilities_k)):
            for label, prob in zip(label_list, prob_list):
                class_index = int(label.replace('__label__', ''))
                scores_matrix[i, class_index] = prob
                
        roc_auc_scores = []
        for i in range(self.num_topics):
            try:
                roc_auc = roc_auc_score(y_test_binarized[:, i], scores_matrix[:, i])
                roc_auc_scores.append(roc_auc)
            except ValueError:
                roc_auc_scores.append(np.nan)
        self.fasttext_roc_auc = np.nanmean(roc_auc_scores)
    
    def compile_results(self):
        self.results = pd.DataFrame({
            'Method': ['LDA', 'LDA_TFIDF', 'BERT', 'FastText'],
            'Accuracy': [self.lda_accuracy, self.lda_tfidf_accuracy, self.bert_accuracy, self.fasttext_accuracy],
            'ROC AUC': [self.lda_roc_auc, self.lda_tfidf_roc_auc, self.bert_roc_auc, self.fasttext_roc_auc]
        })
        self.results = self.results.sort_values(by='ROC AUC', ascending=False)
        self.results.to_csv('roc_auc_scores.csv', index=False)
    
    def run_pipeline(self):
        print()
        print("Iniciando pipeline...")
        
        print("1. Preprocesando datos...")
        self.preprocess_data()
        print("   - Datos preprocesados correctamente.")
        
        print("2. Codificando etiquetas...")
        self.encode_labels()
        print("   - Etiquetas codificadas correctamente.")
        
        print("3. Dividiendo datos en conjuntos de entrenamiento y prueba...")
        self.split_data()
        print("   - División de datos completada.")
        
        print("4. Entrenando modelo LDA...")
        self.train_lda()
        print("   - Modelo LDA entrenado correctamente.")
        
        print("5. Evaluando modelo LDA...")
        self.evaluate_lda()
        print("   - Evaluación del modelo LDA completada.")
        
        print("6. Entrenando y evaluando modelo BERT...")
        self.train_evaluate_bert()
        print("   - Modelo BERT entrenado y evaluado correctamente.")
        
        print("7. Entrenando y evaluando modelo FastText...")
        self.train_evaluate_fasttext()
        print("   - Modelo FastText entrenado y evaluado correctamente.")
        
        print("8. Compilando resultados finales...")
        self.compile_results()
        print("   - Resultados compilados con éxito.")
        
        print("Pipeline ejecutado exitosamente. Los resultados se han guardado en 'roc_auc_scores.csv'.")