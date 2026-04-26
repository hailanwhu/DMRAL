import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import chain
import os
import argparse
import os
import random
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle
from sql_metadata import Parser
import utils
from utils import dataset_config

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path)

def read_json(fn):
  with open(fn) as f:
    return json.load(f)


def write_json(obj, fn):
  with open(fn, 'w') as f:
    json.dump(obj, f, indent=2)

def read_pickle(fn):
  with open(fn, 'rb') as f:
    return pickle.load(f)

def write_pickle(r, fn):
  with open(fn, 'wb') as f:
    pickle.dump(r, f)

def mean_pooling(token_embeddings, mask):
  token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
  sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
  return sentence_embeddings

BATCH_SIZE = 200
def embed(texts, fn, hide_progress=False):
  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))
  
  tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
  model = AutoModel.from_pretrained('facebook/contriever-msmarco').cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break
    assert len(_texts) >= 1
    
    inputs = tokenizer(_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
      vec = model(**inputs)
    vec = mean_pooling(vec[0], inputs['attention_mask'])

    embeds.append(vec.cpu().numpy())
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds

def get_corpus(dataset):
  tables = read_json(f'../../dataset/data/{dataset}/dev_tables.json')
  return list(tables.keys())

def get_corpus_schema(dataset):
  tables = read_json(f'../../dataset/data/{dataset}/dev_tables.json')
  table_schemas = {}
  for ts, tval in tables.items():
    table_schemas[ts] = tval['column_names_original']

  return table_schemas

def sql_to_tables(sql: str, db_id: str):
  gold_ts = Parser(sql).tables
  gold_ts = [f'{db_id}#sep#{gold_t}' for gold_t in gold_ts]
  return gold_ts

# format should be either json or npy
def merge(num_partitions: int, _fn: str, format: str):
  fn = f'{_fn}.{format}'

  results = []

  individuals = [read_json(f'{_fn}_{partition}.json') if format == 'json' else np.load(f'{_fn}_{partition}.npy') for partition in range(num_partitions)]

  if format == 'json':
    for result in individuals:
      results += result
    write_json(results, fn)
  elif format == 'npy':
    results = np.vstack(individuals)
    np.save(fn, results)
  
  print(len(results))

def get_skip_idxs(dataset: str):
  qs = read_json(f'../../dataset/data/{dataset}/dev.json')
  skip_idxs = [i for i in range(len(qs)) if len(sql_to_tables(qs[i]['sql'], qs[i]['db_id'])) == 1]
  return skip_idxs


def decompose_schema(tables):
  r, col_nums = [], []
  for t in tables:
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for t_col in t_cols:
      r.append(f'{t_name}:{t_col}')
    col_nums.append(len(t_cols))
  return r, col_nums

def decompose_schema_full(tables):
  r, col_nums = [], []
  map_dict = {}
  for t in tables:
    f_t_name, f_t_cols = tables[t]['table_name'], tables[t]['column_names']
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for f_t_col, t_col in zip(f_t_cols, t_cols):
      r.append(f'{f_t_name}:{f_t_col}')
      map_dict[f'{f_t_name}:{f_t_col}'] = f'{t_name}:{t_col}'
    col_nums.append(len(t_cols))
  return r, col_nums, map_dict

def ravel_t_score(score, col_nums):
  r = []
  idx = 0
  for col_num in col_nums:
    r.append(score[idx:idx+col_num])
    idx += col_num
  
  assert(idx == len(score))
  return r

def get_sim_scores(dataset, dq: bool, cols_nums=None, mode=None):
  if dq:
    if mode!='full':
      save_fn = f'../../dataset/data/{dataset}/contriever/score_decomp.pkl'
      q_embeds_fn = f'../../dataset/data/{dataset}/contriever/q_decomp.npy'
      t_embeds_fn = f'../../dataset/data/{dataset}/contriever/t_decomp.npy'
    else:
      save_fn = f'../../dataset/data/{dataset}/contriever/score_decomp_f.pkl'
      q_embeds_fn = f'../../dataset/data/{dataset}/contriever/q_decomp.npy'
      t_embeds_fn = f'../../dataset/data/{dataset}/contriever/t_decomp_f.npy'
  else:
    save_fn = f'../../dataset/data/{dataset}/contriever/score.npy'
    q_embeds_fn = f'../../dataset/data/{dataset}/contriever/q.npy'
    t_embeds_fn = f'../../dataset/data/{dataset}/contriever/t.npy'

  q_embeds, t_embeds = torch.from_numpy(np.load(q_embeds_fn)), torch.from_numpy(np.load(t_embeds_fn))

  print(f'#q, #t: {q_embeds.shape[0]}, {t_embeds.shape[0]}')

  if not os.path.isfile(save_fn):
    sim_scores = []
    for q_embed in tqdm(q_embeds):
      sim_scores.append(F.cosine_similarity(q_embed.unsqueeze(0), t_embeds, dim=1).unsqueeze(0))
    sim_scores = torch.vstack(sim_scores).numpy()
    # print(sim_scores.shape)
    
    if not dq:
      np.save(save_fn, sim_scores)
  else:
    if save_fn.endswith('pkl'):
      sim_scores = pickle.load(open(save_fn,'rb'))
    elif save_fn.endswith('npy'):
      sim_scores = np.load(save_fn)


  if dq:
    # all subqueries are flattened --> for each subquery, compute similarity to all columns in all tables
    sim_scores = [ravel_t_score(score, cols_nums) for score in tqdm(sim_scores)]
    write_pickle(sim_scores, save_fn)
  
  return sim_scores

def serialize_table(table):
  db_id, table_name, cols = table['db_id'], table['table_name_original'], table['column_names_original']
  return ' '.join([db_id, table_name] + cols)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--query_decompose',  action='store_true')
  args = parser.parse_args()
  set_seed(1234)
  mode = 'full'

  model = ['tapas', 'contriever'][1]
  dataset = args.dataset
  dq = args.query_decompose

  if not os.path.exists(f'../../dataset/data/{dataset}/contriever/'):
    os.mkdir(f'../../dataset/data/{dataset}/contriever/')

  if dq:
    if mode != 'full':
      q_fn, t_fn = f'../../dataset/data/{dataset}/contriever/q_decomp.npy', f'../../dataset/data/{dataset}/contriever/t_decomp.npy'
    else:
      q_fn, t_fn = f'../../dataset/data/{dataset}/contriever/q_decomp.npy', f'../../dataset/data/{dataset}/contriever/t_decomp_f.npy'
  else:
    q_fn, t_fn = f'../../dataset/data/{dataset}/contriever/q.npy', f'../../dataset/data/{dataset}/contriever/t.npy'
  
  qs = read_json(f'../../dataset/data/{dataset}/dev.json')
  qs = [q['question'] for q in qs]

  if dq:
    qs = read_json(f'../../dataset/data/{dataset}/decomp.json')
    qs = list(chain.from_iterable(qs))

  tables = read_json(f'../../dataset/data/{dataset}/dev_tables.json')
  
  if dq:
    if mode != 'full':
      ts, col_nums = decompose_schema(tables)
    else:
      ts, col_nums, col_map_dict = decompose_schema_full(tables)
  else:
    ts, col_nums = [serialize_table(tables[t]) for t in tables], None

  embed(qs, q_fn)
  embed(ts, t_fn)

  get_sim_scores(dataset, dq, cols_nums=col_nums, mode=mode)


  
