from prodclass.vectorization.product_vectorizer import ProductVectorizer
import numpy as np

class ArgmaxClassifier(ProductVectorizer):
    """
    Classificador que herda de ProductVectorizer e pode lidar tanto com texto quanto com dados numéricos.
    Para dados textuais, utiliza as funcionalidades de vetorização e classificação da classe pai.
    Para dados numéricos, aplica um algoritmo de classificação.
    """
    def __init__(self, method='tfidf', ngram_range=(1, 1), norm=None, query='binary', query_norm=None):
        """
        Inicializa o ArgmaxClassifier com configurações de vetorização e o classificador escolhido.
        """
        super().__init__(method=method, ngram_range=ngram_range, norm=norm, query=query, query_norm=query_norm)

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados fornecidos. Se os dados forem textuais, utiliza a lógica da classe pai.
        Caso contrário, ajusta o classificador numérico aos dados.

        Args:
            X (array-like): Dados de entrada.
            y (array-like): Rótulos de saída.
        """
        # Verifica se X e y são textuais
        super().fit(X, y)

    def predict(self, X):
        """
        Realiza predições com base nos dados fornecidos. Utiliza a lógica de vetorização e classificação
        da classe pai para texto ou o classificador numérico para dados numéricos.

        Args:
            X (array-like): Dados de entrada para realizar predições.

        Returns:
            array: Rótulos preditos.
        """
        return super().predict(X, out='name')
        