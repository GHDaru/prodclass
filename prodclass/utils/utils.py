import os
import requests

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Cria o diretório de destino se não existir

    filename = url.split('/')[-1]  # Assume que o nome do arquivo é a última parte da URL
    file_path = os.path.join(dest_folder, filename)

    # Faz o download do arquivo se ele ainda não existe
    if not os.path.exists(file_path):
        print(f"Baixando {filename}...")
        r = requests.get(url, allow_redirects=True)
        open(file_path, 'wb').write(r.content)
        print(f"Salvo em: {file_path}")
    else:
        print(f"Arquivo {filename} já existe.")

    return file_path

import os
import json

def percorrer_pasta_e_gravar(diretorio, arquivo_saida, prompt):
    arvore = []
    arquivos_py = []

    # Percorre o diretório recursivamente e coleta os caminhos dos arquivos .py
    for raiz, diretorios, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith('.py'):
                caminho_completo = os.path.join(raiz, arquivo)
                arvore.append(caminho_completo.replace(diretorio, ''))
                with open(caminho_completo, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                arquivos_py.append({'arquivo': caminho_completo, 'conteúdo': conteudo})
    
    # Preparar o texto de saída
    saida = [f"<<{prompt}>>\n", "Árvore da pasta:\n"]
    saida.extend([f"{caminho}\n" for caminho in arvore])
    saida.append("\nDetalhes dos Arquivos:\n")
    
    for arquivo in arquivos_py:
        saida.append(json.dumps(arquivo, indent=4, ensure_ascii=False))
        saida.append("\n")
    
    # Gravar no arquivo de saída
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.writelines(saida)



