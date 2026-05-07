#%%
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np

DIR = "../data/processed/"

# Carregar os centroides (espaços vetoriais que representam as comunidades)
df_centroids = pd.read_parquet(DIR+"subreddit_centroids.parquet")
centroids_matrix = df_centroids.values
subreddit_names = df_centroids.index.tolist()

# Carregar textos pré-processados para o c-TF-IDF
df_text = pd.read_csv(DIR+"preprocess_text.csv")
df_meta = pd.read_csv(DIR+"concatenated_title_selftext.csv")

# Unir para ter o texto limpo vinculado ao nome do subreddit
df_full = df_meta[['id','subreddit']].merge(df_text[['id','text_clean']], on='id')
df_full['text_clean'] = df_full['text_clean'].fillna("").astype(str)

# Agrupar texto por subreddit (essencial para o c-TF-IDF funcionar por comunidade)
docs_per_subreddit = df_full.groupby('subreddit')['text_clean'].agg(lambda x: ' '.join(x)).reset_index()

# Garantir que a ordem dos textos seja a mesma dos centroides
docs_per_subreddit = docs_per_subreddit.set_index('subreddit').reindex(subreddit_names).fillna("")
documents = docs_per_subreddit['text_clean'].tolist()

umap_model = UMAP(
    n_neighbors=50,
    min_dist=0.0,
    n_components=3, 
    metric='cosine', 
    random_state=42)

hdbscan_model = HDBSCAN(
    min_cluster_size=50, 
    min_samples=1, 
    metric='euclidean', 
    cluster_selection_method='leaf')

topic_model = BERTopic(
    umap_model = umap_model,
    hdbscan_model=hdbscan_model,
    ctfidf_model = ClassTfidfTransformer(),
    verbose=True
)

topics, probs = topic_model.fit_transform(documents, centroids_matrix)

# %%
# 1. Obter informações dos tópicos e remover colunas pesadas para o resumo
df_topic_info = topic_model.get_topic_info()

# Selecionamos apenas o essencial para o dicionário
df_resumo = df_topic_info[['Topic', 'Count', 'Name', 'Representation']].copy()

# 2. Mapeamento de Subreddits (usando topics atualizados se houve reduce_outliers)
df_subreddit_mapping = pd.DataFrame({
    'subreddit': subreddit_names,
    'Topic': topics  # Certifique-se de usar new_topics se rodou o redutor de outliers
})

# Agrupar nomes de subreddits por tópico
topic_to_subs = df_subreddit_mapping.groupby('Topic')['subreddit'].apply(list).reset_index()
topic_to_subs.columns = ['Topic', 'subreddits_list']

# 3. Merge e Exportação para JSON (Leve e Estruturado)
df_dicionario_final = df_resumo.merge(topic_to_subs, on='Topic', how='left')
df_dicionario_final.to_json("../reports/topic_modeling/dicionario_topicos.json", orient="records", indent=4)

# %%
# 4. Visualizações 
n_clusters_total = len(df_topic_info) - 1
fig_words = topic_model.visualize_barchart(
    top_n_topics=n_clusters_total, 
    n_words=10,
    title="Assinaturas Linguísticas do Ecossistema (c-TF-IDF)"
)
fig_words.write_html("../reports/topic_modeling/visualizacao_topicos.html")

fig_2d = topic_model.visualize_topics(
    title="Topologia do Ecossistema: Distância entre Clusters Ideológicos"
)
fig_2d.write_html("../reports/topic_modeling/mapa_2d_topicos.html")

# %%
