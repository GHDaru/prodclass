import sqlite3
import pandas as pd

class TabelaExperimentos:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self.connect_db()
        self.create_table()

    def connect_db(self):
        """Estabelece conexão com o banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Erro ao conectar ao banco de dados SQLite: {e}")
            return None

    def create_table(self):
        """Cria a tabela Experimentos se não existir."""
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS Experimentos (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Nome TEXT NOT NULL,
                Projeto TEXT NOT NULL,
                UNIQUE(Nome, Projeto)
            );
        '''
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
        except sqlite3.Error as e:
            print(f"Erro ao criar a tabela Experimentos: {e}")

    def add(self, nome, projeto):
        """Adiciona um novo experimento à tabela ou retorna o ID se já existir.

        Args:
            nome (str): Nome do experimento.
            projeto (str): Nome do projeto.

        Returns:
            int: O ID do experimento adicionado ou existente.
        """
        # Primeiro, tenta encontrar um experimento existente com o mesmo nome e projeto
        search_sql = 'SELECT ID FROM Experimentos WHERE Nome = ? AND Projeto = ?'
        try:
            cursor = self.conn.cursor()
            cursor.execute(search_sql, (nome, projeto))
            existing_experiment = cursor.fetchone()

            if existing_experiment:
                print("Experimento já existe. Retornando o ID existente.")
                return existing_experiment[0]

            # Se o experimento não existir, insere um novo
            insert_sql = '''
                INSERT INTO Experimentos (Nome, Projeto)
                VALUES (?, ?);
            '''
            cursor.execute(insert_sql, (nome, projeto))
            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.Error as e:
            print(f"Erro ao adicionar ou buscar experimento: {e}")
            return None

    def to_dataframe(self):
            """Converte todos os experimentos em um DataFrame."""
            query = "SELECT * FROM Experimentos"
            return pd.read_sql_query(query, self.conn)

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