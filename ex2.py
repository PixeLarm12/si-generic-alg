import os
import pandas as pd

def read_files_from_directory(directory, prefix):
    """
    Lê arquivos de um diretório específico com um prefixo definido.
    Retorna uma lista de DataFrames contendo os dados dos arquivos.
    """
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    dataframes = []

    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    return dataframes

def process_datasets():
    ha30_directory = 'ha30/'  # Caminho para o diretório ha30
    uk12_directory = 'uk12/'  # Caminho para o diretório uk12

    # Lê os arquivos com prefixo ha30_
    ha30_dfs = read_files_from_directory(ha30_directory, 'ha30_')
    
    # Lê os arquivos com prefixo uk12_
    uk12_dfs = read_files_from_directory(uk12_directory, 'uk12_')

    # Processa os arquivos lidos
    for df in ha30_dfs:
        # Aqui, substituímos a verificação booleana inadequada
        if df.empty:
            print("DataFrame do diretório ha30 está vazio.")
        else:
            print("DataFrame do diretório ha30 lido com sucesso.")
            # Processa o DataFrame (adicione o processamento necessário aqui)
            print(df.head())

    for df in uk12_dfs:
        if df.empty:
            print("DataFrame do diretório uk12 está vazio.")
        else:
            print("DataFrame do diretório uk12 lido com sucesso.")
            # Processa o DataFrame (adicione o processamento necessário aqui)
            print(df.head())

# Chama a função para iniciar o processo
process_datasets()
