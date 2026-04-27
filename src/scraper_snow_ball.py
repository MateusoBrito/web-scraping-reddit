import requests
import json
import csv
from datetime import datetime, timezone
import time
import os
import re
import logging
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

INPUT_SEED = "../reports/relatorio_top_1000_subreddits.md"
OUTPUT_DIR = "../data/raw/expansao"
MAX_SNOWBALL_DEPTH = 3

MAX_POSTS_PER_SUBREDDIT = 300
MAX_USERS_PER_SUBREDDIT = 10
MAX_POSTS_PER_USER = 10

REQUEST_TIMEOUT = 20
DELAY_BETWEEN_REQUESTS = 2

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────

def extrair_seed_subreddits(caminho_arquivo):
    seed_subreddits = []
    
    # Padrão Regex: busca 'r/' seguido de caracteres alfanuméricos ou underscores
    padrao = r"r/(\w+)"
    
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
            # findall retorna apenas o que está dentro do grupo (\w+)
            # se quiser manter o 'r/', use r"(r/\w+)"
            seed_subreddits = re.findall(padrao, conteudo)
            
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado.")
    
    # Remove duplicatas mantendo a ordem (opcional)
    return list(dict.fromkeys(seed_subreddits))

HEADERS = {
    # PullPush não exige User-Agent específico, mas é boa prática identificar
    "User-Agent": "academic-scraper/1.0 (webmedia2026 research)"
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/scraper.log"),
        logging.StreamHandler()
    ]
)

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────

def save_checkpoint(depth):
    with open(f"{OUTPUT_DIR}/checkpoint.json", "w") as f:
        json.dump({"depth": depth}, f)

def load_checkpoint():
    path = f"{OUTPUT_DIR}/checkpoint.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["depth"]
    return 0

# ─────────────────────────────────────────────
# I/O — SUBREDDITS (JSON)
# ─────────────────────────────────────────────

def save_subreddit_data(all_data, depth, subreddit_name):
    dir_ = f"{OUTPUT_DIR}/{depth}/subreddits"
    os.makedirs(dir_, exist_ok=True)
    file = f"{dir_}/{subreddit_name}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logging.info(f"Dados salvos em {file} ({len(all_data)} posts)")

# ─────────────────────────────────────────────
# I/O — USUÁRIOS (CSV por profundidade)
# ─────────────────────────────────────────────

def get_users_csv_path(depth):
    return f"{OUTPUT_DIR}/{depth}/users_depth_{depth}.csv"

def save_user_data(posts, user_name, depth):
    path = get_users_csv_path(depth)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    CSV_FIELDS = ["depth", "user_name", "id", "subreddit"]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for post in posts:
            writer.writerow({**post, "depth": depth, "user_name": user_name})

    logging.info(f"  {user_name}: {len(posts)} posts adicionados em users_depth_{depth}.csv")

def users_scraped(depth):
    users = set()
    for d in range(depth + 1):
        path = get_users_csv_path(d)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                users.add(row["user_name"])
    return users

def sample_users(posts, n):
    """
    Seleciona os N usuários mais ativos (mais posts).
    Desempate: maior score total acumulado.
    """

    author_stats = defaultdict(lambda: {"post_count": 0, "total_score": 0})

    for post in posts:
        author = post.get("author")
        if not author or author == "[deleted]":
            continue
        author_stats[author]["post_count"] += 1
        author_stats[author]["total_score"] += int(post.get("score") or 0)

    sorted_authors = sorted(
        author_stats.items(),
        key=lambda x: (x[1]["post_count"], x[1]["total_score"]),
        reverse=True
    )

    return [author for author, _ in sorted_authors[:n]]

# ─────────────────────────────────────────────
# REQUEST — PullPush API
# Diferente do scraping HTML, a API retorna JSON diretamente
# Não precisa de BeautifulSoup
# ─────────────────────────────────────────────

def make_request(url: str, params: dict, retries: int = 10) -> list[dict] | None:
    """
    Faz requisição à API PullPush e retorna lista de posts (dicts).
    A paginação é feita via parâmetro 'before' (timestamp do último post).
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
            elif resp.status_code == 429:
                wait = (attempt + 1) * 10
                logging.info(f"  [429] Rate limited. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                logging.info(f"  [HTTP {resp.status_code}] {url}")
                return None
        except requests.RequestException as e:
            logging.error(f"  [Erro] {e} — tentativa {attempt + 1}/{retries}")
            time.sleep(5)
        except json.JSONDecodeError as e:
            # API retornou resposta não-JSON (raro mas possível)
            logging.error(f"  [JSON inválido] {e}")
            return None
    return None

# ─────────────────────────────────────────────
# PARSE — Adapta o dict da API para o formato do projeto
# Antes: extraía atributos HTML (data-author, data-score...)
# Agora: lê campos do dict JSON retornado pela API
# ─────────────────────────────────────────────

def parse_post(post: dict, features: list[str]) -> dict | None:
    """
    Converte um post retornado pela API PullPush para o formato padrão do projeto.
    O parâmetro features controla quais campos são extraídos.
    """
    try:
        result = {}

        if "post_id" in features:
            result["id"] = post.get("id")
        if "title" in features:
            result["title"] = post.get("title")
        if "selftext" in features:
            # Texto completo do post — None se for link externo
            selftext = post.get("selftext", "")
            result["selftext"] = selftext if selftext not in ("", "[removed]", "[deleted]") else None
        if "author" in features:
            result["author"] = post.get("author")
        if "subreddit" in features:
            subreddit = post.get("subreddit")
            result["subreddit"] = subreddit.lower() if subreddit else None
        if "score" in features:
            result["score"] = post.get("score")
        if "timestamp_raw" in features:
            created_utc = post.get("created_utc")
            if created_utc:
                result["timestamp"] = datetime.fromtimestamp(
                    int(created_utc), 
                    tz=timezone.utc
                ).isoformat()
        else:
            result["timestamp"] = None
        if "comments" in features:
            result["num_comments"] = post.get("num_comments")
        if "url" in features:
            permalink = post.get("permalink", "")
            result["url"] = f"https://reddit.com{permalink}" if permalink else post.get("url")
        if "is_self" in features:
            # True = post de texto, False = link externo
            result["is_self"] = post.get("is_self")
        if "over_18" in features:
            result["over_18"] = post.get("over_18")
        if "removed_by_category" in features:
            # Indica se foi removido e por quem (moderador, spam, etc.)
            result["removed_by_category"] = post.get("removed_by_category")

        return result if result else None

    except Exception as e:
        logging.error(f"Erro ao parsear post: {e}")
        return None

# ─────────────────────────────────────────────
# SCRAPING — SUBREDDITS
# Paginação via "before": passa o created_utc do último post
# para buscar posts mais antigos na próxima página
# ─────────────────────────────────────────────

def scrape_subreddits(subreddits, depth, max_posts=MAX_POSTS_PER_SUBREDDIT):
    FEATURES_SUBREDDIT = [
        "post_id", "title", "selftext", "author", "subreddit",
        "score", "timestamp_raw", "comments",
        "url", "is_self", "over_18", "removed_by_category"
    ]

    BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

    for subreddit in subreddits:
        seen_ids = set()
        all_data = []
        last_score = None  # cursor de paginação por score

        logging.info(f"Scraping subreddit: {subreddit}")

        while len(all_data) < max_posts:
            params = {
                "subreddit": subreddit,
                "size": min(100, max_posts - len(all_data)),  # máximo 100 por página
                "sort": "desc",
                "sort_type": "score",
            }
            if last_score is not None:
                params["score"] = f"<{last_score}"  # posts com score menor que o último

            posts = make_request(BASE_URL, params)

            if not posts:
                logging.info(f"  {subreddit}: sem mais posts, encerrando")
                break

            new_posts = []
            for post in posts:
                data = parse_post(post, FEATURES_SUBREDDIT)
                if data and data.get("id") and data["id"] not in seen_ids:
                    seen_ids.add(data["id"])
                    new_posts.append(data)

            if not new_posts:
                logging.info(f"  {subreddit}: sem posts novos, encerrando")
                break

            all_data.extend(new_posts)
            last_score = posts[-1].get("score")  # cursor para próxima página
            logging.info(f"  {len(all_data)} posts coletados...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

        save_subreddit_data(all_data, depth, subreddit)

# ─────────────────────────────────────────────
# SCRAPING — USUÁRIOS
# Coleta apenas subreddit + id dos posts do usuário
# suficiente para descobrir novas comunidades no snowball
# ─────────────────────────────────────────────

def scrape_users(depth, max_posts=MAX_POSTS_PER_USER):
    users = users_scraped(depth)
    FEATURES_USER = ["post_id", "subreddit"]
    BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

    # Agrega todos os posts de todos os subreddits da profundidade
    #all_posts = []
    for filename in os.listdir(f"{OUTPUT_DIR}/{depth}/subreddits"):
        if filename.endswith(".json"):
            path = os.path.join(f"{OUTPUT_DIR}/{depth}/subreddits", filename)
            with open(path, "r", encoding="utf-8") as f:
                posts = json.load(f)
            #all_posts.extend(posts)

        logging.info(f"{filename}: {len(posts)} posts")

        # Top N usuários mais ativos no total da profundidade
        sampled_users = sample_users(posts, MAX_USERS_PER_SUBREDDIT)
        logging.info(f"  Amostra: {len(sampled_users)} usuários (por subreddit)")

        for user_name in sampled_users:
            if not user_name or user_name == "[deleted]":
                continue

            if user_name in users:
                continue

            users.add(user_name)
            seen_ids = set()

            params = {
                "author": user_name,
                "size": max_posts,
                "sort": "desc",
                "sort_type": "score",
            }

            user_posts = make_request(BASE_URL, params)

            if not user_posts:
                logging.info(f"  {user_name}: sem posts, pulando")
                save_user_data([], user_name, depth)
                continue

            all_data = []
            for user_post in user_posts:
                data = parse_post(user_post, FEATURES_USER)
                if data:
                    data = {k: v for k, v in data.items() if v is not None}
                if data and data.get("id") and data["id"] not in seen_ids:
                    seen_ids.add(data["id"])
                    all_data.append(data)

            logging.info(f"  {user_name}: {len(all_data)} posts coletados...")
            time.sleep(DELAY_BETWEEN_REQUESTS)
            save_user_data(all_data, user_name, depth)

# ─────────────────────────────────────────────
# SNOWBALL — lógica idêntica ao scraper original
# ─────────────────────────────────────────────

def subreddits_to_scrape(depth):
    subreddits = set()
    subreddits.update(s.lower() for s in SEED_SUBREDDITS)

    for d in range(depth):
        path = get_users_csv_path(d)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subreddit = row.get("subreddit")
                if subreddit:
                    subreddits.add(subreddit.lower())

    for d in range(depth + 1):
        dir_ = f"{OUTPUT_DIR}/{d}/subreddits"
        if not os.path.exists(dir_):
            continue
        for filename in os.listdir(dir_):
            if filename.endswith(".json"):
                subreddits.discard(filename.replace(".json", "").lower())

    return subreddits


def snow_ball():
    start_depth = load_checkpoint()
    if start_depth > 0:
        logging.info(f"Retomando do checkpoint: profundidade {start_depth}")

    for depth in range(start_depth, MAX_SNOWBALL_DEPTH + 1):
        subreddits = subreddits_to_scrape(depth)

        logging.info(f"\n{'='*50}")
        logging.info(f"SNOWBALL — Profundidade {depth}/{MAX_SNOWBALL_DEPTH}")
        logging.info(f"Subreddits nesta rodada: {subreddits}")
        logging.info(f"{'='*50}")

        if not subreddits:
            logging.info("Nenhum subreddit novo. Encerrando snowball.")
            break

        scrape_subreddits(subreddits, depth)
        if depth < MAX_SNOWBALL_DEPTH:
            scrape_users(depth)
        save_checkpoint(depth)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

    try:
        seed_subreddits = extrair_seed_subreddits(INPUT_SEED)
        logging.info(f"Subreddits extraídos da seed: {seed_subreddits}")
        global SEED_SUBREDDITS
        SEED_SUBREDDITS = seed_subreddits
    except Exception as e:
        logging.error(f"Erro ao extrair seed: {e}")
        return

    try:
        snow_ball()
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        save_checkpoint(load_checkpoint())

if __name__ == "__main__":
    main()