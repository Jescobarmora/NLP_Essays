import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        # Convertir a minúsculas
        text = text.lower()
        # Tokenización
        tokens = word_tokenize(text)
        # Eliminación de stopwords y tokens no alfabéticos
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        # Lematización
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens