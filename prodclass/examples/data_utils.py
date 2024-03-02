# data/data_utils.py

import pandas as pd
import os

def get_dataset(dataset='retailpt-br'):
    try:
        # Assumindo que o arquivo CSV está no diretório 'data'
        file_path = os.path.join(os.path.dirname(__file__), dataset+'.csv')
    except:
        raise ValueError("Unknown dataset")
    data = pd.read_csv(file_path, sep=';',  dtype = {'nm_item':str})

    if dataset == 'retailpt-br':    
        return data[data.nm_product != 'ITENS COM PROBLEMA']   
    
    return data
    
# Você pode adicionar mais opções para outros conjuntos de dados
