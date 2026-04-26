import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sqlite3
import os
import argparse
import pandas as pd
from datasketch import MinHash, MinHashLSH
from utils import read_json, read_pickle, write_json, get_corpus, merge, get_skip_idxs, get_corpus_schema, embed, sql_to_tables
import os
import sqlite3
from tqdm import tqdm
from collections import defaultdict
import json


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_table_columns(db_path, table_name, column_names, limit=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = f"SELECT {', '.join(f'`{c}`' for c in column_names)} FROM `{table_name}`"
    if limit:
        query += f" LIMIT {limit}"
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    # Transpose rows to column-wise format
    col_data = {col: set() for col in column_names}
    for row in rows:
        for i, val in enumerate(row):
            if val is not None:
                v = val.strip() if isinstance(val, str) else val
                col_data[column_names[i]].add(v)
    return col_data

def compute_jaccard(dataset: str, mode='dev'):
    tables = read_json(f'../../dataset/data/{dataset}/{mode}_tables.json')
    jaccard = {}
    fn = f'../../dataset/data/{dataset}/{mode}_jaccard.json'
    if os.path.isfile(fn):
        jaccard = read_json(fn)

    table_cache = {}
    for tid, table_info in tqdm(tables.items(), desc="Loading tables"):
        db_id = table_info["db_id"]
        # table_name = table_info["table_name"]
        col_names = table_info["column_names_original"]
        table_name = table_info['table_name']

        try:
          potential_df = pd.read_csv(f"../../dataset/datalake/{dataset}/{tid}.csv", index_col=0)
        except Exception as e:
          print(f"[ERROR] Failed to load table file")
          continue  # Skip this table and proceed

        if len(potential_df.columns) != len(col_names):
            print(f"[WARNING] Column mismatch in table {table_name}:")
            print("  CSV columns:", list(potential_df.columns))
            print("  Expected columns:", col_names)
            continue  # Skip mismatched tables

        # Cache each column as a set of its values
        table_cache[tid] = {
            col_names[i]: set(potential_df.iloc[:, i].dropna().tolist())
            for i in range(len(col_names))
        }

    # Step 2: Compute pairwise Jaccard similarity
    for t1 in tqdm(tables, desc="Comparing tables"):
        for t2 in tables:
            if t1 == t2:
                continue
            if f'{t1}-{t2}' in jaccard or f'{t2}-{t1}' in jaccard:
                continue

            if t1 not in table_cache or t2 not in table_cache:
                continue  # skip missing tables

            table_pair_key = f'{t1}-{t2}'
            jaccard[table_pair_key] = {}

            for c1 in tables[t1]['column_names_original']:
                r1 = table_cache[t1].get(c1, set())
                for c2 in tables[t2]['column_names_original']:
                    r2 = table_cache[t2].get(c2, set())
                    if not r1 or not r2:
                        continue

                    sim_score = len(r1 & r2) / min(len(r1), len(r2))
                    col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'
                    jaccard[table_pair_key][col_pair_key] = sim_score

    write_json(jaccard, fn)  # Save after each outer loop

    return jaccard


def get_uniqueness(dataset: str, mode='dev'):
  tables = read_json(f'../../dataset/data/{dataset}/{mode}_tables.json')
  
  u_scores = {}

  for t in tqdm(tables):
    db, t_name = tables[t]['db_id'], tables[t]['table_name_original']
    t_name = tables[t]['table_name']

    csv_path = f"../../dataset/datalake/{dataset}/{t}.csv"
    potential_df = pd.read_csv(csv_path)

    for i in range(len(tables[t]['column_names_original'])):
      c = tables[t]['column_names_original'][i]

      col_values = potential_df.iloc[:, i].dropna().tolist()  # Remove NaNs if any
    
      try:
          unique_values = set(col_values)
          uniqueness_score = len(unique_values) / len(col_values) if col_values else 0
      except Exception as e:
          print(f"Error processing column {c} in table {t}: {e}")
          uniqueness_score = 0

      u_scores[f'{t}#sep#{c}'] = uniqueness_score
       
  write_json(u_scores, f'../../dataset/data/{dataset}/{mode}_uniqueness.json')


def process_word_fast(word: str):
  return word.replace('_', ' ').replace('.', '').lower().strip()  

def overlap_coefficient(s1: str, s2: str):
  s1, s2 = process_word_fast(s1).split(' '), process_word_fast(s2).split(' ')

  s1, s2 = set(s1), set(s2)
  return len(s1 & s2) / min(len(s1), len(s2))

# the serialized format is 'db_id table_name column_name'
def get_cols_embeds(dataset, mode='dev'):
  tables = read_json(f'../../dataset/data/{dataset}/{mode}_tables.json')

  cols, cols_idxs = [], [0]

  for t in tables:
    db, t_name = tables[t]['db_id'], tables[t]['table_name_original']
    t_cols = [process_word_fast(f'{db} {t_name} {c}') for c in tables[t]['column_names_original']]
    cols += t_cols
    cols_idxs.append(cols_idxs[-1] + len(t_cols))

  cols_embeds = embed(cols, None, hide_progress=True)
  cols_embeds_dict = {}

  for t_idx, t in enumerate(tables):
    cols_embeds_dict[t] = cols_embeds[cols_idxs[t_idx]:cols_idxs[t_idx + 1]]

  return cols_embeds_dict

def get_col_sim(dataset, mode='dev'):
  tables = read_json(f'../../dataset/data/{dataset}/{mode}_tables.json')

  exact_sim, semantic_sim = {}, {}
  exact_sim_fn = f'../../dataset/data/{dataset}/{mode}_exact_col_sim.json'
  semantic_sim_fn = f'../../dataset/data/{dataset}/{mode}_semantic_col_sim.json'

  cols_embeds = get_cols_embeds(dataset, mode)

  for t1 in tqdm(tables):
    for t2 in tables:
      if t1 == t2:
        continue

      if f'{t1}-{t2}' in exact_sim or f'{t2}-{t1}' in exact_sim:
        continue
            
      table_pair_key = f'{t1}-{t2}'
      exact_sim[table_pair_key], semantic_sim[table_pair_key] = {}, {}

      db1, db2 = tables[t1]['db_id'], tables[t2]['db_id']
      t_name1, t_name2 = tables[t1]['table_name_original'], tables[t2]['table_name_original']
      cols1_embeds, cols2_embeds = cols_embeds[t1], cols_embeds[t2]

      semantic_sim_matrix = []
      for x in cols1_embeds:
        semantic_sim_matrix.append(F.cosine_similarity(x.unsqueeze(0), cols2_embeds, dim=1).unsqueeze(0))
      semantic_sim_matrix = torch.vstack(semantic_sim_matrix).tolist()

      for i1, c1 in enumerate(tables[t1]['column_names_original']):
        for i2, c2 in enumerate(tables[t2]['column_names_original']):
          col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'

          exact_score = overlap_coefficient(f'{db1} {t_name1} {c1}', f'{db2} {t_name2} {c2}')
          exact_sim[table_pair_key][col_pair_key] = exact_score
          semantic_sim[table_pair_key][col_pair_key] = semantic_sim_matrix[i1][i2]
  
    write_json(exact_sim, exact_sim_fn)
    write_json(semantic_sim, semantic_sim_fn)


def get_score(t1, col1, t2, col2, score_dict):
  if f'{t1}-{t2}' in score_dict:
    score = score_dict[f'{t1}-{t2}']
  else:
    score = score_dict[f'{t2}-{t1}']

  if f'{t1}#sep#{col1}-{t2}#sep#{col2}' in score:
    score = score[f'{t1}#sep#{col1}-{t2}#sep#{col2}']
  elif f'{t2}#sep#{col2}-{t1}#sep#{col1}' in score:
    score = score[f'{t2}#sep#{col2}-{t1}#sep#{col1}']
  else:
    score = 0
  
  return score


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True)
  args = parser.parse_args()

  dataset = args.dataset
  args.mode = 'dev'


  compute_jaccard(dataset, args.mode)
  print('jaccard done...')
  get_uniqueness(dataset, args.mode)
  print('uniquness done...')
  get_col_sim(dataset, args.mode)
  print('column similarity done...')
