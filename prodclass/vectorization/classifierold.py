import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

default_vectorizer_settings = {
    'input': 'content',
    'encoding': 'utf-8',
    'decode_error': 'strict',
    'strip_accents': None,
    'lowercase': True,
    'preprocessor': None,
    'tokenizer': None,
    'analyzer': 'word',
    'stop_words': None,
    'token_pattern': r"(?u)\b\w\w+\b",
    'ngram_range': (1, 1),
    'max_df': 1.0,
    'min_df': 1,
    'max_features': None,
    'vocabulary': None,
    'binary': False,
    'dtype': 'float64',
    'norm': 'l2',
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': False
}


class ProductVectorizer:
    """
    A classifier for product descriptions using text vectorization and machine learning techniques.
    """
    def __init__(self, method='tfidf', ngram_range=(1, 1), norm = None, query = 'binary', query_norm = None):
        """
        Initializes the ProductClassifier with a specific text vectorization method.
        """
        self.method = method
        self.query_method = query
        self.ngram_range = ngram_range
        self.category_index = {}
        self.index_category = {}
        self.model = None        
        self.norm = norm
        self.norm_query = query_norm
        self.fixtfidf = (self.method in {'tfidf'})
        self.vectorizer = self.initialize_vectorizer()
        self.query = self.initialize_query()

    def initialize_vectorizer(self):
        """
        Initialize the TfidfVectorizer based on the specified method.
        """
        
        vectorizer_options = {
            'binary': {'binary': True, 'use_idf': False},
            'termfrequency': {'binary': False, 'use_idf': False},
            'tfsmooth': {'use_idf': False, 'sublinear_tf': True},
            'tfidf': {'use_idf': True, 'smooth_idf': False},
            'tfidfscikit': {},
        }

        options = vectorizer_options.get(self.method, {})
        return TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
            strip_accents='ascii',
            ngram_range=self.ngram_range,
            norm = self.norm,
            **options
        )
    
    def initialize_query(self):
        """
        Initialize the TfidfVectorizer based on the specified method.
        """
        
        query_options = {
            'binary': {'binary': True, 'use_idf': False},
            'termfrequency': {'binary': False, 'use_idf': False},
            'tf_smooth': {'use_idf': False, 'sublinear_tf': True},
            'tfidf': {'use_idf': True, 'smooth_idf': False},
            'tfidfscikit': {}
        }

        options = query_options.get(self.query_method, {})
        return TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
            strip_accents='ascii',
            ngram_range=self.ngram_range,
            norm = self.norm_query,
            **options
        )

    def fit(self, X, y=None):
        """
        Fit the model with the provided data.
        """
        self.original_descriptions = np.array(X)
        self.documents = X.copy()
        if y is not None:
            self.original_categories = np.array(y)
            self.build_category_mapping(y)
            self.group_descriptions_by_category()

        self.vectorizer.fit(self.documents)
        if self.fixtfidf:
           self.vectorizer.idf_-=1

        self.query.fit(self.documents)


    def build_category_mapping(self, categories):
        """
        Build mapping dictionaries for categories.
        """
        unique_categories = set(categories)
        self.category_index = {category: idx for idx, category in enumerate(unique_categories)}
        self.index_category = {idx: category for category, idx in self.category_index.items()}

    def group_descriptions_by_category(self):
        """
        Group descriptions by their categories.
        """
        self.documents = [' '.join(self.original_descriptions[self.original_categories == category]) 
                          for category in self.category_index]
        
    def get_term_document_matrix(self):
      X = self.vectorizer.transform(self.documents)
      if hasattr(self, 'original_categories'):
        df = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.category_index.keys())
      else:
        df = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())
      return df.T
    
    def get_term_query_matrix(self,X):
       X = self.query.transform(X)
       df = pd.DataFrame(X.toarray(), columns=self.query.get_feature_names_out())
       return df.T
    
    def transform(self, X):
       X = np.array(X)
       return self.query.transform(X).T
    
    def predict(self,X, out = 'number'):
        y = self.predict_score(X)
        y = np.argmax(y, axis=0)
        y = np.array(y).flatten() if out == 'number' else self.get_category_from_index(np.array(y).flatten())
        return y
    
    def predict_score(self, X):
        A = self.vectorizer.transform(self.documents)
        q = self.query.transform(X)
        y = A@q.T
        return y

    
    def get_index_from_category(self, categories):
      return np.vectorize(lambda x: self.category_index.get(x,-1))(categories)

    def get_category_from_index(self, index):
      return np.vectorize(lambda x: self.index_category.get(x,-1))(index)
    
# Example usage
# classifier = ProductClassifier(method='binary', ngram_range=(1, 2))
# classifier.fit(X, y)

