# data/data_utils.py

import pandas as pd
import os

def get_dataset(dataset='retailpt-br'):
    # Construir o caminho do arquivo relativo ao local deste script
    file_path = os.path.join(os.path.dirname(__file__), f'{dataset}.csv')

    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    # Ler o arquivo CSV
    data = pd.read_csv(file_path, sep=';', dtype={'nm_item': str})

    # Filtrar itens com problemas, se necessário
    if dataset == 'retailpt-br':
        return data[data.nm_product != 'ITENS COM PROBLEMA']

    return data
