import sqlite3
import pandas as pd
import json
from datetime import datetime

class TabelaExecucoes:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self.connect_db()
        self.create_table()

    def connect_db(self):
        """Estabelece conexão com o banco de dados."""
        return sqlite3.connect(self.db_path)

    def create_table(self):
        """Cria a tabela Execucoes se não existir."""
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS Execucoes (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                ID_Modelo INTEGER,
                Rodada INTEGER,
                Acuracia REAL,
                F1ScoreMacro REAL,
                TempoProcessamento REAL,
                DataExecucao TEXT,
                Estatisticas TEXT,
                FOREIGN KEY (ID_Modelo) REFERENCES Modelos(ID)
            );
        '''
        self.conn.execute(create_table_sql)

    def add(self, id_modelo, rodada, acuracia, f1_score_macro, tempo_processamento, estatisticas):
        """Adiciona um registro de execução à tabela.

        Args:
            id_modelo (int): ID do modelo que foi executado.
            rodada (int): Número da rodada de validação cruzada.
            acuracia (float): Acurácia alcançada na rodada.
            f1_score_macro (float): F1-Score Macro alcançado na rodada.
            tempo_processamento (float): Tempo de processamento da execução em segundos.
            estatisticas (dict): Dicionário contendo estatísticas adicionais da rodada.
        """
        data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        estatisticas_json = json.dumps(estatisticas)
        
        insert_sql = '''
            INSERT INTO Execucoes (ID_Modelo, Rodada, Acuracia, F1ScoreMacro, TempoProcessamento, DataExecucao, Estatisticas)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        '''
        self.conn.execute(insert_sql, (id_modelo, rodada, acuracia, f1_score_macro, tempo_processamento, data_execucao, estatisticas_json))
        self.conn.commit()

    def to_dataframe(self):
        """Converte todos os registros de execuções em um DataFrame."""
        df = pd.read_sql_query("SELECT * FROM Execucoes", self.conn)
        return df

    def query(self, consulta):
        """Executa uma consulta personalizada e retorna os resultados em um DataFrame.

        Args:
            consulta (str): Consulta SQL para ser executada.

        Returns:
            pd.DataFrame: DataFrame contendo os resultados da consulta.
        """
        df = self.to_dataframe()
        return df.query(consulta)
    
    def query_sql(self, consulta):
        """Executa uma consulta personalizada e retorna os resultados em um DataFrame.

        Args:
            consulta (str): Consulta SQL para ser executada.

        Returns:
            pd.DataFrame: DataFrame contendo os resultados da consulta.
        """
        return pd.read_sql_query(consulta, self.conn)
