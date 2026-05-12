import os
import json
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

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

df['selftext'] = df['selftext'].fillna('')
df['title'] = df['title'].fillna('')

df['text'] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
df_not_en = df[df['lang'] != 'en']
print(f"Quantidade de posts classificados como não ingleses pelo langdetect {len(df_not_en)}")


device = 0 if torch.cuda.is_available() else -1
print(f"Usando: {'GPU' if device == 0 else 'CPU'}")

model_id = "papluca/xlm-roberta-base-language-detection"
lang_classifier = pipeline("text-classification", model=model_id, device=device)

texts = df_not_en['text'].astype(str).tolist()

batch_size = 128
roberta_results = []

print(f"Re-classificando {len(texts)} posts com RoBERTa...")

for i in tqdm(range(0,len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    batch = [t[:512] for t in batch]

    outpus = lang_classifier(batch)
    roberta_results.extend([out['label'] for out in outpus])

df_not_en['lang_roberta'] = roberta_results

corrigidos_para_en = df_not_en[df_not_en['lang_roberta'] == 'en']
print(f"Posts que o langdetect errou (eram inglês na verdade): {len(corrigidos_para_en)}")

df_not_en[['id','lang','lang_roberta']].to_csv("../reports/checking_lang_detect.csv")
