#%%
import pandas as pd
import numpy as np
from tqdm import tqdm

DIR = "../data/processed/"
INPUT_PATH_POSTS = DIR+"concatenated_title_selftext.csv"
INPUT_PATH_EMBEDDINGS = DIR+"embeddings_posts.npy"
OUTPUT_PATH = DIR+"subreddit_centroids.parquet"

# 1. Carregar metadados e matriz
print("Carregando dados...")
df = pd.read_csv(INPUT_PATH_POSTS)
embeddings = np.load(INPUT_PATH_EMBEDDINGS, mmap_mode='r')

# Garantir que o DF tem o mesmo tamanho da matriz
assert len(df) == embeddings.shape[0], "O número de posts não coincide com a matriz de embeddings!"

def calculate_centroids(df, embeddings):
    centroids = {}
    
    # Agrupa os índices das linhas por subreddit
    print("Agrupando índices por subreddit...")
    subreddit_indices = df.groupby('subreddit').groups
    print(f"Número de subreddits encontrados: {len(subreddit_indices)}")

    for sub, indices in tqdm(subreddit_indices.items(), desc="Calculando centroides"):
        # Extrai apenas os vetores deste subreddit da matriz NumPy
        # O NumPy faz isso de forma muito eficiente
        sub_vectors = embeddings[indices]
        
        # Cálculo do Centroide: Somatório / Total de Posts
        # axis=0 calcula a média de cada uma das 384 dimensões separadamente
        centroid = np.mean(sub_vectors, axis=0)
        
        centroids[sub] = centroid
        
    return centroids

# Executar cálculo
dict_centroids = calculate_centroids(df, embeddings)

# Transformar em DataFrame para facilitar o uso posterior
df_centroids = pd.DataFrame.from_dict(dict_centroids, orient='index')
df_centroids.index.name = 'subreddit'

# Salvar como Parquet (preserva a precisão dos floats e é rápido)
df_centroids.to_parquet(OUTPUT_PATH)

print(f"Centroides de {len(df_centroids)} subreddits calculados e salvos.")