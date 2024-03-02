from table_models.tabela_experimentos import TabelaExperimentos
from table_models.tabela_modelos import TabelaModelos
from table_models.tabela_execucoes import TabelaExecucoes
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import itertools
import pandas as pd
import time


class ExperimentManager:
    """Gestor de experimentos para avaliação de modelos de machine learning."""

    def __init__(self, project_name, db_path):
        """Inicializa o gestor de experimentos e as tabelas associadas.

        Args:
            project_name (str): Nome do projeto.
            db_path (str): Caminho para o arquivo do banco de dados.
        """
        self.project_name = project_name
        self.db_path = db_path
        # Inicializa as classes de tabelas
        self.tabela_experimentos = TabelaExperimentos(self.db_path)
        self.tabela_modelos = TabelaModelos(self.db_path)
        self.tabela_execucoes = TabelaExecucoes(self.db_path)

    def add_experiment(self, experiment_name):
        """Adiciona um novo experimento ao banco de dados, se não existir.

        Args:
            experiment_name (str): Nome do experimento.

        Returns:
            int: O ID do experimento adicionado.
        """
        return self.tabela_experimentos.add(experiment_name, self.project_name)

    def add_model_variation(self, experiment_id, parameters, seed):
        """Adiciona uma variação de modelo ao banco de dados se não existir.

        Args:
            experiment_id (int): ID do experimento ao qual o modelo pertence.
            parameters (dict): Parâmetros do modelo.
            seed (int): Semente utilizada para a inicialização do modelo.

        Returns:
            int: O ID da variação do modelo adicionada.
        """
        return self.tabela_modelos.add(experiment_id, parameters, seed)

    def record_execution(self, model_id, round, accuracy, f1_score_macro, processing_time):
        """Registra os resultados e o tempo de processamento de uma execução do modelo no banco de dados.

        Args:
            model_id (int): ID do modelo que foi executado.
            round (int): Número da rodada de validação cruzada.
            accuracy (float): Acurácia alcançada na rodada.
            f1_score_macro (float): F1-Score Macro alcançado na rodada.
            processing_time (float): Tempo de processamento da execução em segundos.
        """
        estatisticas = {"accuracy": accuracy, "f1_score_macro": f1_score_macro}
        self.tabela_execucoes.add(model_id, round, accuracy, f1_score_macro, processing_time, estatisticas)


    def generate_parameter_combinations(self, parameter_dict):
        """Gera todas as combinações possíveis dos parâmetros fornecidos.

        Args:
            parameter_dict (dict): Dicionário contendo os parâmetros e suas variações a serem testadas.

        Returns:
            list of dict: Lista contendo dicionários de cada combinação de parâmetros.
        """
        return [dict(zip(parameter_dict, v)) for v in itertools.product(*parameter_dict.values())]

    def get_experiment_results(self, experiment_name):
        """Recupera os resultados de um experimento específico.

        Args:
            experiment_name (str): Nome do experimento cujos resultados devem ser recuperados.

        Returns:
            pd.DataFrame: DataFrame contendo os resultados do experimento.
        """
        # Correção: Use a consulta correta para recuperar o ID do experimento
        consulta_experimento = f"SELECT ID FROM Experimentos WHERE Nome = '{experiment_name}' AND Projeto = '{self.project_name}'"
        experiment_id_df = self.tabela_experimentos.query_sql(consulta_experimento)

        if not experiment_id_df.empty:
            experiment_id = experiment_id_df.iloc[0]['ID']
            
            # Constrói a consulta SQL para recuperar resultados associados ao experimento
            consulta_resultados = f'''
                SELECT m.Parametros, e.Rodada, e.Acuracia, e.F1ScoreMacro, e.TempoProcessamento, e.DataExecucao, e.Estatisticas
                FROM Execucoes e
                JOIN Modelos m ON e.ID_Modelo = m.ID
                WHERE m.ID_Experimento = {experiment_id}
            '''
            return self.tabela_execucoes.query_sql(consulta_resultados)
        else:
            return pd.DataFrame()

    def run(self, experiment_name, classifier, X, y, parameter_dict, folds=10, random_state=100, kind = 'numeric'):
        """Executa um experimento com todas as combinações de parâmetros especificadas.

        Args:
            experiment_name (str): Nome do experimento.
            classifier (any): O classificador a ser usado no experimento.
            X (array-like): Dados de entrada para o modelo.
            y (array-like): Rótulos de saída para o modelo.
            parameter_dict (dict): Dicionário contendo os parâmetros e suas variações a serem testadas.
            folds (int): Número de dobras para a validação cruzada K-Fold.
            random_state (int, optional): Semente para a reprodutibilidade dos resultados.
        """
        experiment_id = self.add_experiment(experiment_name)
        parameter_combinations = self.generate_parameter_combinations(parameter_dict)

        for params in parameter_combinations:
            model_id = self.add_model_variation(experiment_id, params, random_state)
            # Chama evaluate_model para cada combinação de parâmetros
            self.evaluate_model(model_id, classifier, X, y, params, folds, random_state, kind)

    def fit_predict(self, classifier, parameter_variation,X_train,X_test, y_train, y_test, kind = 'numeric'):
        if kind == 'numeric':
            classifier.fit(X_train,y_train)
            return classifier.predict(X_test)
        if kind == 'argmax':
            classifier.fit(X_train,y_train)
            return classifier.predict(X_test)       

            
    def evaluate_model(self, model_id, classifier, X, y, parameter_variation, folds=5, random_state=None, kind = 'numeric'):
        """Avalia o modelo com um conjunto específico de parâmetros usando KFold para validação cruzada.
        kind é o tipo de X e y dado na entrada, numeric, argmax, vectorize
        ...
        """
        # Configura o classificador com os parâmetros fornecidos
        classifier.set_params(**parameter_variation)

        # Configura o KFold
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            start_time = time.time()  # Inicia o cronômetro

            #---> logicas diferentes para tipos de tarefas
            if kind == 'numeric':
                #X e y numéricos,classificadir espera numerico, y_test é numerico
                classifier.fit(X_train, y_train)            
                y_pred = classifier.predict(X_test)
            elif kind == 'argmax':
                #X e y textuais
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test, out='string')                
            #--->Fim das lógicas diferenciadas para avaliação
            # classifier.fit(X_train, y_train)            
            # y_pred = classifier.predict(X_test)            

            end_time = time.time()  # Finaliza o cronômetro
            tempo_processamento = end_time - start_time

            accuracy = accuracy_score(y_test, y_pred)
            f1_score_macro = f1_score(y_test, y_pred, average='macro')

            estatisticas = {
                "quantidade_registros_treino": len(train_index),
                "quantidade_registros_teste": len(test_index)
            }

            # Registra os resultados da execução
            self.tabela_execucoes.add(model_id, fold_idx, accuracy, f1_score_macro, tempo_processamento, estatisticas)



