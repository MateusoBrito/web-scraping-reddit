---

# Análise de Ecossistemas Digitais: Mapeamento Topológico e Semântico do Reddit

Este projeto de pesquisa tem como objetivo mapear a estrutura de comunidades no Reddit, utilizando uma abordagem híbrida que combina **Ciência de Redes (Análise de Grafos)** e **Processamento de Linguagem Natural (NLP)**. O foco está na identificação de "nichos ideológicos" e na análise da dinâmica de usuários dentro desses ecossistemas.

## 🚀 Status do Projeto: 
Em progresso

---

## Arquitetura do Sistema

O pipeline de dados está dividido em cinco módulos principais:

### 1. Coleta de Dados (`scraper_snow_ball.py`)
* Implementação de um algoritmo **Snowball Sampling** de profundidade 3 via API PullPush.
* Extração de subreddits semente (*seed*) para expansão da rede baseada em autores comuns.
* Coleta automatizada de posts (título, texto, autor, score e metadados).

### 2. Engenharia de Grafos (`graph.py`)
* Construção de um grafo de subreddits onde as arestas representam a **interseção de autores**.
* Geração de arquivo `.graphml` para análise de centralidade e topologia em ferramentas como Gephi ou Cytoscape.

### 3. Representação Vetorial (`generate_embeddings.py` & `calculate_subreddit_centroids.py`)
* **Embeddings de Sentença:** Transformação de posts em vetores de 384 dimensões utilizando o modelo `paraphrase-multilingual-MiniLM-L12-v2`.
* **Cálculo de Centroides:** Agrupamento de todos os vetores de um subreddit para gerar uma "assinatura semântica" única (Média Aritmética Vetorial) por comunidade.

### 4. Modelagem de Tópicos e Clusterização (`topic_modeling.py`)
* **UMAP & HDBSCAN:** Redução de dimensionalidade e agrupamento denso dos centroides.
* **c-TF-IDF (Class-based TF-IDF):** Extração de palavras-chave que definem cada cluster ideológico, tratando o conjunto de subreddits como uma única classe.
* **Visualização Interativa:** Geração de mapas intermitentes de distância inter-tópica e gráficos de barras das assinaturas linguísticas.

---

## 📂 Estrutura de Arquivos Gerados

* `../data/processed/subreddit_centroids.parquet`: Base vetorial das comunidades.
* `../reports/topic_modeling/dicionario_topicos.json`: Mapeamento estruturado de Tópicos $\rightarrow$ Palavras-chave $\rightarrow$ Subreddits.
* `../reports/topic_modeling/mapa_2d_topicos.html`: Visualização interativa da topologia do ecossistema.
* `../data/processed/grafo.graphml`: Estrutura de conexões entre comunidades.

---

## 📊 Metodologia Acadêmica (Resumo)

### Cálculo do Centroide
Para um subreddit com $n$ posts, o centroide $C$ é definido por:
$$C = \frac{1}{n} \sum_{i=1}^{n} v_i$$

### Importância do Termo (c-TF-IDF)
$$W_{t, c} = tf_{t, c} \cdot \log \left( 1 + \frac{A}{df_t} \right)$$

---

### 🛠 Instalação e Requisitos

Para configurar o ambiente de desenvolvimento e executar os scripts deste projeto, siga os passos abaixo:

#### 1. Clonar o Repositório
```bash
git clone https://github.com/MateusoBrito/webmedia-ecosystem-Reddit.git
cd webmedia-ecosystem-Reddit
```

#### 2. Criar e Ativar um Ambiente Virtual (Recomendado)
É altamente recomendável o uso de um ambiente virtual para evitar conflitos de dependências:
```bash
# Criar o ambiente
python3 -m venv venv

# Ativar no Linux:
source venv/bin/activate

```

#### 3. Instalar Dependências
O projeto utiliza bibliotecas robustas para NLP e Ciência de Dados. Com o ambiente virtual ativo, execute:
```bash
pip install -r requirements.txt
```

> **Nota sobre Hardware:** Devido ao uso de *Sentence Embeddings* e *UMAP*, recomenda-se o uso de uma GPU (CUDA) para acelerar o processamento, embora o código esteja configurado para rodar em CPU automaticamente caso necessário.

---
**Desenvolvido por:** Mateus Brito, Thiago Braga, NaaN  
**Tecnologias:** Python, BERTopic, Sentence-Transformers, iGraph, Pandas, NumPy.