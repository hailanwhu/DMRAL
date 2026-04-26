import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

def union_fun(dataset_name):
    mode = 'dev'
    tables_json_path = f'../../dataset/data/{dataset_name}/{mode}_tables.json'

    with open(tables_json_path, 'r') as f:
        tables = json.load(f)

    # Initialize the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 1: Embed schemas
    schema_embeddings = []
    table_ids = []
    table_titles = []
    column_counts = []

    def embed_schema(column_names):
        embeddings = model.encode(column_names, convert_to_tensor=True)
        return torch.mean(embeddings, dim=0)

    for tid, table_info in tqdm(tables.items(), desc="Embedding Schemas"):
        col_names = [name for name in table_info["column_names_original"] if isinstance(name, str) and name.strip()]
        if not col_names:
            continue
        schema_emb = embed_schema(col_names)
        schema_embeddings.append(schema_emb.cpu().numpy())
        table_ids.append(tid)
        table_titles.append(table_info["table_name"].lower())  # Normalize for string comparison
        column_counts.append(len(col_names))

    # Step 2: Compute pairwise unionability
    schema_embeddings_np = np.stack(schema_embeddings)
    similarity_matrix = cosine_similarity(schema_embeddings_np)
    threshold = 0.8
    title_threshold = 0.9  # adjustable

    unionable_pairs = []

    def title_similarity(title1, title2):
        return SequenceMatcher(None, title1, title2).ratio()

    for i in range(len(table_ids)):
        for j in range(i + 1, len(table_ids)):
            # Check embedding similarity, column count match, and title similarity
            if (
                similarity_matrix[i, j] >= threshold and
                column_counts[i] == column_counts[j] and
                title_similarity(table_titles[i], table_titles[j]) >= title_threshold
            ):
                unionable_pairs.append((table_ids[i], table_ids[j]))

    # Step 3: Construct unionable clusters using union-find
    parent = {tid: tid for tid in table_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for t1, t2 in unionable_pairs:
        union(t1, t2)

    clusters = {}
    for tid in table_ids:
        root = find(tid)
        clusters.setdefault(root, []).append(tid)

    # Assign group_id and collect unionable table groups
    cluster_dict = {}
    group_id = 0

    for tid in table_ids:
        root = find(tid)
        if root not in cluster_dict:
            cluster_dict[root] = {
                "group_id": f"group_{group_id}",
                "tables": []
            }
            group_id += 1
        cluster_dict[root]["tables"].append(tid)

    # Filter out singleton groups (only one table, not really unionable)
    final_clusters = {
        v["group_id"]: v["tables"]
        for v in cluster_dict.values()
    }

    output_path = f"../../dataset/data/{dataset_name}/{mode}_unionable_clusters.json"
    with open(output_path, 'w') as f:
        json.dump(final_clusters, f, indent=2)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True)
  args = parser.parse_args()

  dataset_name = args.dataset


