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
