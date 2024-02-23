import sqlite3
import pandas as pd
import hashlib
import json
import itertools
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

class ExperimentManager:
    """Gestor de experimentos para avaliação de modelos de machine learning.

    Attributes:
        project_name (str): Nome do projeto, utilizado para nomear o arquivo do banco de dados.
        db_file (str): Caminho para o arquivo do banco de dados.
    """

    def __init__(self, project_name):
        """Inicializa o gestor de experimentos, criando as tabelas no banco de dados se não existirem.

        Args:
            project_name (str): Nome do projeto.
        """
        self.project_name = project_name
        self.db_file = f'{project_name}.db'
        self.create_tables()

    def connect_db(self):
        """Estabelece conexão com o banco de dados do projeto.

        Returns:
            Connection: Objeto de conexão SQLite.
        """
        return sqlite3.connect(self.db_file)

    def create_tables(self):
        """Cria as tabelas no banco de dados se não existirem."""
        with self.connect_db() as conn:
            # Tabela para Experimentos
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Experimentos (
                    ID INTEGER PRIMARY KEY,
                    Nome TEXT,
                    Projeto TEXT,
                    UNIQUE(Nome, Projeto)
                )''')
            # Tabela para Modelos (anteriormente chamada de Parametros)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Modelo (
                    ID INTEGER PRIMARY KEY,
                    ID_Experimento INTEGER,
                    Parametros TEXT,
                    Semente INTEGER,
                    Hash TEXT UNIQUE,
                    FOREIGN KEY (ID_Experimento) REFERENCES Experimentos(ID)
                )''')
            # Tabela para Execucao (anteriormente chamada de Metricas)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Execucao (
                    ID_Modelo INTEGER,
                    Rodada INTEGER,
                    Acuracia REAL,
                    F1ScoreMacro REAL,
                    TempoProcessamento REAL,
                    FOREIGN KEY (ID_Modelo) REFERENCES Modelo(ID)
                )''')

    def add_experiment(self, experiment_name):
        """Adiciona um novo experimento ao banco de dados, se não existir.

        Args:
            experiment_name (str): Nome do experimento.

        Returns:
            int: O ID do experimento adicionado.
        """
        with self.connect_db() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO Experimentos (Nome, Projeto) VALUES (?, ?) ON CONFLICT(Nome, Projeto) DO NOTHING", (experiment_name, self.project_name))
            return cur.lastrowid

    def add_model_variation(self, experiment_id, parameters, seed):
        """Adiciona uma variação de modelo ao banco de dados se não existir.

        Args:
            experiment_id (int): ID do experimento ao qual o modelo pertence.
            parameters (dict): Parâmetros do modelo.
            seed (int): Semente utilizada para a inicialização do modelo.

        Returns:
            int: O ID da variação do modelo adicionada.
        """
        with self.connect_db() as conn:
            # Gera o hash baseado nos parâmetros e na semente
            params_hash = self.generate_hash({**parameters, "seed": seed})
            cur = conn.cursor()
            # Verifica se a variação do modelo já existe baseado no hash
            cur.execute("SELECT ID FROM Modelo WHERE Hash = ?", (params_hash,))
            existing_model = cur.fetchone()
            
            if existing_model:
                # Retorna o ID do modelo existente se encontrado
                return existing_model[0]
            else:
                # Insere a nova variação do modelo se não existir
                cur.execute("INSERT INTO Modelo (ID_Experimento, Parametros, Semente, Hash) VALUES (?, ?, ?, ?)", 
                            (experiment_id, json.dumps(parameters), seed, params_hash))
                return cur.lastrowid

    def record_execution(self, model_id, round, accuracy, f1_score_macro, processing_time):
        """Registra os resultados e o tempo de processamento de uma execução do modelo no banco de dados.

        Args:
            model_id (int): ID do modelo que foi executado.
            round (int): Número da rodada de validação cruzada.
            accuracy (float): Acurácia alcançada na rodada.
            f1_score_macro (float): F1-Score Macro alcançado na rodada.
            processing_time (float): Tempo de processamento da execução em segundos.
        """
        with self.connect_db() as conn:
            conn.execute('''INSERT INTO Execucao 
                            (ID_Modelo, Rodada, Acuracia, F1ScoreMacro, TempoProcessamento) 
                            VALUES (?, ?, ?, ?, ?)''', 
                            (model_id, round, accuracy, f1_score_macro, processing_time))
            

    def generate_hash(self, params):
        """Gera um hash MD5 a partir dos parâmetros do modelo.

        Args:
            params (dict): Parâmetros do modelo.

        Returns:
            str: Hash MD5 dos parâmetros.
        """
        hash_input = json.dumps(params, sort_keys=True).encode()
        return hashlib.md5(hash_input).hexdigest()

    def evaluate_model(self, experiment_id, classifier,  X, y, parameter_variation, folds=5, random_state=None):
        """Atualizado para calcular e registrar o tempo de processamento."""
        model_id = self.add_model_variation(experiment_id, parameter_variation, seed = random_state)
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            start_time = time.time()  # Início do cronômetro
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # # Aplicando o TextVectorizer para transformar os textos
            # X_train_transformed = vectorizer.fit_transform(X_train, y_train)
            # X_test_transformed = vectorizer.transform(X_test)

            # classifier.fit(X_train_transformed, y_train)
            # y_pred = classifier.predict(X_test_transformed)
            
            # accuracy = accuracy_score(y_test, y_pred)
            # f1_score_macro = f1_score(y_test, y_pred, average='macro')
            accuracy = len(X_train)
            f1_score_macro = len(X_test)

            end_time = time.time()  # Fim do cronômetro
            processing_time = end_time - start_time  # Cálculo do tempo de processamento            
            self.record_execution(model_id, i, accuracy, f1_score_macro, processing_time)

    def run(self, experiment_name, classifier, X, y, parameter_dict={}, folds=10, random_state=None):
        """Executa um experimento com todas as combinações de parâmetros especificadas.

        Args:
            experiment_name (str): Nome do experimento.
            classifier (classifier): O classificador a ser usado no experimento.
            X (array-like): Dados de entrada para o modelo.
            y (array-like): Rótulos de saída para o modelo.
            parameter_dict (dict): Dicionário contendo os parâmetros e suas variações a serem testadas.
            folds (int): Número de dobras para a validação cruzada K-Fold.
            random_state (int, optional): Semente para a reprodutibilidade dos resultados.
        """
        experiment_id = self.add_experiment(experiment_name)
        parameter_combinations = self.generate_parameter_combinations(parameter_dict)
        
        for parameter_variation in parameter_combinations:
            self.evaluate_model(experiment_id, classifier, X, y, parameter_variation, folds=folds, random_state=random_state)

    def get_experiment_results(self, experiment_name):
        """Recupera os resultados de um experimento específico.

        Args:
            experiment_name (str): Nome do experimento cujos resultados devem ser recuperados.

        Returns:
            pandas.DataFrame: DataFrame contendo os resultados do experimento.
        """
        with self.connect_db() as conn:
            query = '''
            SELECT M.Parametros, E.Rodada, E.Acuracia, E.F1ScoreMacro
            FROM Execucao E
            JOIN Modelo M ON E.ID_Modelo = M.ID
            JOIN Experimentos Exp ON M.ID_Experimento = Exp.ID
            WHERE Exp.Nome = ? AND Exp.Projeto = ?
            '''
            results = pd.read_sql_query(query, conn, params=(experiment_name, self.project_name))
        return results

    def generate_parameter_combinations(self, parameter_dict):
        """Gera todas as combinações possíveis dos parâmetros fornecidos.

        Args:
            parameter_dict (dict): Dicionário contendo os parâmetros e suas variações a serem combinadas.

        Returns:
            list of dict: Lista contendo dicionários de cada combinação de parâmetros.
        """
        keys, values = zip(*parameter_dict.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
