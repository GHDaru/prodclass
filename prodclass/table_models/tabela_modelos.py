import sqlite3
import pandas as pd
import json

class TabelaModelos:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self.connect_db()
        self.create_table()

    def connect_db(self):
        """Estabelece conexão com o banco de dados."""
        return sqlite3.connect(self.db_path)

    def create_table(self):
        """Cria a tabela Modelos se não existir."""
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS Modelos (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                ID_Experimento INTEGER,
                Parametros TEXT NOT NULL,
                Semente INTEGER,
                Hash TEXT UNIQUE,
                FOREIGN KEY (ID_Experimento) REFERENCES Experimentos(ID)
            );
        '''
        self.conn.execute(create_table_sql)

    def add(self, id_experimento, parametros, semente):
        """Adiciona uma nova variação de modelo à tabela ou retorna o ID se já existir.

        Args:
            id_experimento (int): ID do experimento associado ao modelo.
            parametros (dict): Parâmetros do modelo.
            semente (int): Semente utilizada para a inicialização do modelo.

        Returns:
            int: O ID da variação do modelo adicionada ou existente.
        """
        parametros_json = json.dumps(parametros, sort_keys=True)
        hash_modelo = self.generate_hash(parametros_json, semente)

        cursor = self.conn.cursor()
        cursor.execute("SELECT ID FROM Modelos WHERE Hash = ?", (hash_modelo,))
        existing_model = cursor.fetchone()

        if existing_model:
            return existing_model[0]

        cursor.execute("INSERT INTO Modelos (ID_Experimento, Parametros, Semente, Hash) VALUES (?, ?, ?, ?)",
                       (id_experimento, parametros_json, semente, hash_modelo))
        self.conn.commit()
        return cursor.lastrowid

    def generate_hash(self, parametros, semente):
        """Gera um hash único para a combinação de parâmetros e semente.

        Args:
            parametros (str): Parâmetros do modelo em formato JSON.
            semente (int): Semente utilizada para a inicialização do modelo.

        Returns:
            str: Hash MD5 gerado a partir dos parâmetros e semente.
        """
        import hashlib
        hash_input = (parametros + str(semente)).encode()
        return hashlib.md5(hash_input).hexdigest()

    def to_dataframe(self):
        """Converte todos os modelos em um DataFrame."""
        return pd.read_sql_query("SELECT * FROM Modelos", self.conn)

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
