from langchain_core.prompts import PromptTemplate
import utils
from utils import prompt, model_config
from model_config import llm
import json
import re
import time
import pickle
import tqdm
from tqdm import tqdm 
import random
import pandas as pd
import ast
import argparse

##########################################
#               PROMPTS
##########################################
# Please output revised sub-questions with explanation.
# 
mmqa_rewrite_prompt_table = prompt.collect_missing_table_prompt | llm


##########################################
#               FUNCTIONS
##########################################


def obtain_missing_tables(question, related_tables):
    response = mmqa_rewrite_prompt_table.invoke({"question": question, "database":related_tables})
    return response.content.strip()



def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def get_corpus_schema(tables_path):
  tables = read_json(tables_path)
  table_schemas = {}
  for ts, tval in tables.items():
    table_schemas[ts] = [tval['column_names_original'], tval['table_name_original']]
    # table_schemas[tval['table_name_original'].lower()] = [tval['column_names_original'], tval['table_name_original']]

  return table_schemas

def get_question_evidence(input_path):
  corpus = read_json(input_path)
  question_evds = {}
  for tval in corpus:
    question_evds[tval['question']] = tval['evidence']

  return question_evds

def get_question_goldsql(input_path):
  corpus = read_json(input_path)
  question_evds = {}
  for tval in corpus:
    question_evds[tval['question']] = tval['sql']

  return question_evds

def combine_related_tabs(related_tables, table_schemas):
  subq_solution_dict = []
  for subq, subr in zip(range(len(related_tables)), related_tables):
    # tab_str = f"{table_schemas[subr][1]} ({', '.join(table_schemas[subr][0])}) -- one sampled row {es.extract_row(subr)[0]}"
    tab_str = f"{table_schemas[subr][1]} ({', '.join(table_schemas[subr][0])})"
    subq_solution_dict.append(tab_str)
  return '\n'.join(subq_solution_dict)
    

  
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", required=True, type=str)
  parser.add_argument("--save_fpath", required=True, type=str)
  parser.add_argument('--qtab_fpath', required=True, type=str)
  parser.add_argument('--qsubqs_fpath', required=True, type=str)
  args = parser.parse_args()
  dataset_name = args.dataset_name
  
  topk = 5
  method = 'myqd'
  
  q_table_map = pickle.load(open(args.qtab_fpath,'rb'))
  subq_content = pickle.load(open(args.qsubqs_fpath, 'rb'))

  if type(q_table_map) == list:
    q_table_map = q_table_map[0]
  if type(subq_content) == list:
    subq_content = subq_content[0]


  tables_path = f"../../dataset/data/{dataset_name}/dev_tables.json"
  q_ev_path = f"../../dataset/data/{dataset_name}/dev.json"

  table_schemas = get_corpus_schema(tables_path)

  sql_saved_path = args.save_fpath
  # f"data/{dataset_name}_output/{dataset_name}_missing_tables.pkl"
  total_time = 0
  MAX_RETRIES = 3
  BACKOFF = 1.0   

  missing_table_dict = {}
  for q, retrieved_tabs in tqdm(q_table_map.items(), total=len(q_table_map)):
    
    if method in [ 'myqd']:
      singlehop_tabs = retrieved_tabs['multi_hop'][:topk]

      retrieved_tables = combine_related_tabs(list(set(singlehop_tabs)), table_schemas)

      success = False
      retries = 0
      while not success and retries < MAX_RETRIES:
          try:
              start_time = time.time()
              current_output = obtain_missing_tables(q, retrieved_tables)
              total_time += time.time() - start_time

              try:
                  missing_tables = json.loads(current_output)['Completing Tables']
              except:
                  missing_tables = ast.literal_eval(current_output)['Completing Tables']

              missing_table_dict[q] = missing_tables
              success = True  # ✅ success
          except Exception as e:
              print(f"⚠️ Retry {retries + 1} for question: {q} due to error: {e}")
              retries += 1
              time.sleep(BACKOFF * (2 ** retries))  # exponential backoff

      if not success:
          print(f"❌ Failed after {MAX_RETRIES} retries for question: {q}")

      missing_table_dict[q] = missing_tables

  print(f'infer total time is {total_time}')
  pickle.dump([missing_table_dict, total_time], open(sql_saved_path, 'wb') )
