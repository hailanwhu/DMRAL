import pickle
import json
from index_component import *
import os
import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from transformers import AutoTokenizer
from ragatouille import RAGPretrainedModel
import string
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='bird', type=str)
    parser.add_argument("--missing_tab_fpath", required=True, type=str)
    parser.add_argument("--qtab_fpath", required=True, type=str)
    parser.add_argument('--save_fpath', required=True, type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    tables = read_json(f'../../datasets/data/{dataset_name}/dev_tables.json')

    schema_elements = {t: serialize_table(tables[t]).strip() for t in tables}


    faiss_fpath  = f"index/{dataset_name}_faiss_store"
    if not os.path.exists(faiss_fpath):
        faiss_index = create_faiss_index(schema_elements, save_path=faiss_fpath)
    else:
        faiss_index = load_faiss_index(faiss_fpath)

    missing_tab_fpath = args.missing_tab_fpath
    retrieved_tab_fpath = args.qtab_fpath


    missing_tab_dict = pickle.load(open(missing_tab_fpath, 'rb'))
    if type(missing_tab_dict) == list:
        missing_tab_dict = missing_tab_dict[0]
    retrieved_tab_dict = pickle.load(open(retrieved_tab_fpath, 'rb'))
    if type(retrieved_tab_dict) == list:
        retrieved_tab_dict = retrieved_tab_dict[0]

    recall_tab_dict = {}
    start_time = time.time()
    for question, missing_tabs in tqdm(missing_tab_dict.items()):
        # gt_tabs = retrieved_tab_dict[question]['gt']
        previous_tabs = retrieved_tab_dict[question]['multi_hop']
        # failed_recall_tabs = [item for item in gt_tabs if item not in previous_tabs]

        if missing_tabs is None:continue
        all_tabs = []
        for missing_t in missing_tabs:
            res = retrieve_tables(missing_t, faiss_index, top_k=3)
            # retrieved_schemas = [_[0] for _ in res]
            retrieved_schemas = [_[-1] for _ in res]
            norerank_tables = [_['table_name'] for _ in retrieved_schemas if _ not in previous_tabs]
            all_tabs.extend(norerank_tables)

        recall_tab_dict[question] = all_tabs
            
    total_time = time.time() - start_time
    print(f"total retrieve time {total_time}")
    pickle.dump([recall_tab_dict, total_time], open(args.save_fpath, 'wb'))


