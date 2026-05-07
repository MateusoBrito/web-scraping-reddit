#%%
import pandas as pd
import numpy as np
import torch
import re
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def clean_for_embeddings(text):
    # 1. Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 2. Remove espaços extras e quebras de linha
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

DIR = "../data/processed/"
INPUT_PATH = DIR+"concatenated_title_selftext.csv"
OUTPUT_PATH = DIR+"embeddings_posts.npy"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 128

def run_embeddings(input_path = INPUT_PATH,
                   output_path = OUTPUT_PATH,
                   model_name = MODEL_NAME,
                   batch_size = BATCH_SIZE):
    
    print("Carregando DataFrame...")
    df = pd.read_csv(input_path)
    df['text'] = df['text'].fillna("").apply(clean_for_embeddings)
    df.to_parquet(DIR+"text_to_embeddings.parquet", index=False)
    sentences = df['text'].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    model = SentenceTransformer(model_name, device=device)

    print(f"Iniciando cálculo de embeddings para {len(sentences)} posts...")
    embeddings = model.encode(
        sentences, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

    # 5. Salvar Resultados
    print(f"Salvando embeddings em: {output_path}")
    np.save(output_path, embeddings)
    print("Concluído com sucesso!")

if __name__ == "__main__":
    run_embeddings()