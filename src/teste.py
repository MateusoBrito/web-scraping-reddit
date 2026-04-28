#%%
import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

#%%
rows = []
erros = []

# Itera por cada profundidade
for depth in range(3 + 1):
    dir_ = f"../data/raw/expansao/{depth}/subreddits"
    if not os.path.exists(dir_):
        continue
    for filename in os.listdir(dir_):
        if filename.endswith(".json"):
            path = os.path.join(dir_, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    posts = json.load(f)
                # Adiciona a profundidade em cada post para rastreabilidade
                for post in posts:
                    post["depth"] = depth
                rows.extend(posts)
            except json.JSONDecodeError:
                erros.append(path)


df = pd.DataFrame(rows)

print(f"Arquivos com erro: {len(erros)}")
print(f"Quantidade de subreddits únicos: {df['subreddit'].nunique()}")
print(f"Quantidade de autores únicos: {df['author'].nunique()}")
print(f"Quantidade de posts carregados: {len(df)}")

# %%

df['selftext'] = df['selftext'].fillna('')
df['title'] = df['title'].fillna('')

df['text'] = df['title']+df['selftext']
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
sentences = df['text'].tolist()

embeddings = model.encode(
    sentences, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True
)

np.save("embeddings_posts.npy", embeddings)
# %%
