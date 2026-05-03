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

#%%
new_topics = topic_model.reduce_outliers(documents, topics)

#%%
topic_info = topic_model.get_topic_info()
print(topic_info.head())
# %%
fig_words = topic_model.visualize_barchart(n_words=10)
fig_words.write_html("../reports/topic_modeling/visualizacao_topicos.html")
# %%
df_topic_info = topic_model.get_topic_info()
df_subreddit_mapping = pd.DataFrame({
    'subreddit': subreddit_names,
    'Topic': topics
})

topic_to_subs = df_subreddit_mapping.groupby('Topic')['subreddit'].apply(lambda x: ', '.join(x)).reset_index()
topic_to_subs.columns = ['Topic', 'subreddits_list']
topic_to_subs.to_csv("../reports/topic_modeling/dicionario_topicos_comunidades.csv", index=False)
# %%
