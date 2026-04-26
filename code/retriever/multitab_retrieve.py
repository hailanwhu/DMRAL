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
import argparse
import time


def identify_overlap_tables(jaccard_json_path, unique_json_path, csim_json_path, cemb_sim_json_path, dataset_name, sim_threshold=0.90, unique_threshold=0.98):
    tmp_str = json.load(open(jaccard_json_path, 'rb'))
    unique_str = json.load(open(unique_json_path, 'rb'))
    csim_str = json.load(open(csim_json_path, 'rb'))
    cembsim_str = json.load(open(cemb_sim_json_path, 'rb'))
    valid_tab_pairs = []
    cemb_threshold_dict = {'spiderdl': 0.55, 'birddl':0.65}

    for key, tab_pair_items in tmp_str.items():
        for tab_pair, score in tab_pair_items.items():
            
            if score>sim_threshold:    
                tab_arrs = tab_pair.split('-')
                if len(tab_arrs)!=2:
                    if '-'.join(tab_arrs[:2]) in unique_str:
                        unique_ratio = max(unique_str['-'.join(tab_arrs[:2])], unique_str['-'.join(tab_arrs[2:])])
                    else:
                        unique_ratio = max(unique_str[tab_arrs[0]], unique_str['-'.join(tab_arrs[1:])])
                else:
                    unique_ratio = max(unique_str[tab_arrs[0]], unique_str[tab_arrs[1]])
                if unique_ratio<unique_threshold:continue
                col_sim = csim_str[key][tab_pair]          
                
                
                if col_sim<0.5:continue
                cemb_sim = cembsim_str[key][tab_pair]
                
                if cemb_sim<cemb_threshold_dict[dataset_name]:continue
                

                all_tab_names = []
                for i in range(len(tab_arrs)):
                    # all_tab_names.append(tab_arrs[i])
                    if '#sep#' in tab_arrs[i]:
                        all_tab_names.append('#sep#'.join( tab_arrs[i].split('#sep#')[:-1] ))
                    # if '#sep#' in tab_arrs[i]:
                    #     all_tab_names.append( tab_arrs[i].split('#sep#')[1].lower() )
                if len(all_tab_names) != 2:
                    print('error ---')
                    print(all_tab_names, tab_arrs, tab_pair)
                    continue            
                # assert len(all_tab_names) == 2
                
                updated_tab_pair = sorted(all_tab_names)
                if updated_tab_pair not in valid_tab_pairs:
                    valid_tab_pairs.append(updated_tab_pair)

    return valid_tab_pairs


def find_valid_paths_from_pairs(matrix, neighbors):
    n, m = len(matrix), len(matrix[0])
    results = []

    # Preprocess neighbors into a mapping: (row, col) -> list of (row, col)
    neighbor_map = {}
    for (src, tgt) in neighbors:
        if src not in neighbor_map:
            neighbor_map[src] = []
        neighbor_map[src].append(tgt)

    def dfs(current, path, score):
        row, col = current
        # If we've reached the last column, add the current path and score.
        if col == m - 1:
            results.append((path[:], score))
            return
        # Look up the valid neighbors for the current node
        for next_node in neighbor_map.get((row, col), []):
            next_row, next_col = next_node
            # Allow same row in different columns; we check coordinate uniqueness.
            if next_node in path:
                continue
            next_score = score + matrix[next_row][next_col]
            dfs(next_node, path + [next_node], next_score)

    # Start DFS from all nodes in column 0
    for row in range(n):
        start = (row, 0)
        dfs(start, [start], matrix[row][0])

    return results

def pad_alpha(alpha, pad_value=0):
    max_len = max(len(a) for a in alpha)
    padded_alpha = [
        a + [pad_value] * (max_len - len(a))
        for a in alpha
    ]
    return np.array(padded_alpha)



def retrieve_multi_table_path(Q, tables, alpha, valid_tab_pairs, top_k=5):
    """
    Multi-Table Retrieval Algorithm
    Args:
        Q: list of sub-questions (length = n)
        tables: list of list of table strings per hop, shape [n][M]
        alpha: list of list of question-table relevance scores, shape [n][M]
        top_k: number of tables to retrieve

    Returns:
        List of top_k selected unique tables
    """

    n = len(Q)
    M = len(tables[0])  # Assuming each hop has M candidate tables
    neighbors = []
    for i in range(1, n):
        for j in range(len(tables[i-1])):
            prev_table = tables[i-1][j]
            for k in range(len(tables[i])):
                curr_table = tables[i][k]
                if sorted([prev_table, curr_table]) in valid_tab_pairs or prev_table == curr_table:
                    neighbors.append(((j, i-1), (k, i)))


    alpha = pad_alpha(alpha)
    paths = find_valid_paths_from_pairs(alpha.T, neighbors)

    all_candidate_paths = []
    for p in paths:
        cur_path = []
        for t in p[0]:
            table_name = tables[t[1]][t[0]]
            # .split()[1].lower()
            cur_path.append(table_name)
        all_candidate_paths.append(cur_path)
    

    initial_tab_score_dict = {}
    for i in range(n):
        for j in range(len(tables[i])):
            table_name = tables[i][j]
            initial_tab_score_dict[table_name] = max(initial_tab_score_dict.get(table_name, 0), np.array(alpha)[i][j])


    tab_score_dict = {}
    for p in paths:
        for t in p[0]:
            table_name = tables[t[1]][t[0]]
            tab_score_dict[table_name] = max(tab_score_dict.get(table_name, 0), p[1])
    sorted_items = [_[0] for _ in sorted(tab_score_dict.items(), key=lambda x:x[-1], reverse=True)[:top_k]]
    if len(sorted_items)<top_k:
        other_items = [_[0] for _ in sorted(initial_tab_score_dict.items(), key=lambda x:x[-1], reverse=True)]
        for _ in other_items:
            if _ not in sorted_items:
                sorted_items.append(_)
            if len(sorted_items)==top_k:
                break
    return sorted_items, all_candidate_paths


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--decp_fpath", required=True, type=str)
    parser.add_argument('--rerank', action='store_true', help="Enable the rerank of table path")
    parser.add_argument('--save_fpath', required=True, type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    
    rerank = args.rerank
    decp_fpath = args.decp_fpath
    save_fpath = args.save_fpath

    tables = read_json(f'../../dataset/data/{dataset_name}/dev_tables.json')

    # schema_elements = {tables[t]['table_name_original']: serialize_table(tables[t]).strip() for t in tables}
    schema_elements = {t: serialize_table(tables[t]).strip() for t in tables}


    faiss_fpath  = f"index/{dataset_name}_faiss_store"
    if not os.path.exists(faiss_fpath):
        faiss_index = create_faiss_index(schema_elements, save_path=faiss_fpath)
    else:
        faiss_index = load_faiss_index(faiss_fpath)


    jaccard_json_path = f'../../dataset/data/{dataset_name}/dev_jaccard.json'
    uniqueness_json_path = f'../../dataset/data/{dataset_name}/dev_uniqueness.json'
    
    if args.dataset_name == 'spiderdl':
        col_sim_path = f'../../dataset/data/{dataset_name}/exact_col_sim.json'
        col_emb_sim_path = f'../../dataset/data/{dataset_name}/semantic_col_sim.json'
    else:
        col_sim_path = f'../../dataset/data/{dataset_name}/dev_exact_col_sim.json'
        col_emb_sim_path = f'../../dataset/data/{dataset_name}/dev_semantic_col_sim.json'
    estimated_overlap_tabpairs = identify_overlap_tables(jaccard_json_path, uniqueness_json_path, col_sim_path, col_emb_sim_path, dataset_name, 0.98)

    if dataset_name in ['birddl']:
        evidence_path = f'../{dataset_name}/output/evidence.pkl'
        question_evidence_map_dict = pickle.load(open(evidence_path, 'rb'))

    decomposed_data = pickle.load(open(decp_fpath, 'rb'))
    if type(decomposed_data) == list:
        decomposed_data = decomposed_data[0]

    training_path_dict = {'birddl':'checkpoint/birddl/colbert',\
            'spiderdl': "checkpoint/spiderdl/colbert"}

    if rerank:
        training_path = training_path_dict[dataset_name]
        RAG_our = RAGPretrainedModel.from_pretrained(training_path)
        tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")


    output_res = {}
    # estimated_rnks = []
    singlehop_time, multihop_time = 0, 0
    for question, subqs in tqdm(decomposed_data.items()):
        start_time = time.time()
        res = retrieve_tables(question, faiss_index, top_k=10)
        singlehop_time += time.time() - start_time

        # single_hop_res = [_[0].split()[1].lower() for _ in res]
        single_hop_res = [_[-1] for _ in res]

        multi_hop_retrieved_tables, multi_hop_retrieved_table_scores, min_val, max_hop = [], [], np.inf, -1
        multihop_norerank_tables = []
        
        # if len(subqs)==0:
        #     subqs= decomposed_old_data[question]

        for i in range(len(subqs)):
            subq = subqs[i]

            if dataset_name in ['birddl']:
                keywords = []
                if question in question_evidence_map_dict:
                    keywords = question_evidence_map_dict[question].get(subq, [])
                    keywords = filter_name_variants([_ for _ in keywords if _.lower() not in ['id']])

                if len(keywords)>0:
                    start_time = time.time()
                    res = retrieve_tables_bird(subq, load_faiss_index(faiss_fpath), schema_elements, embedder, top_k=8, keywords=keywords)
                    multihop_time += time.time() - start_time
                    if len(res)==0:
                        print('---',keywords, question, subq)
                    if len(res)<10:
                        enrich_res = retrieve_tables(subq, faiss_index, top_k=8)
                        for item in enrich_res:
                            if item[0] not in [_[0] for _ in res]:
                                res.append(item)
                            if len(res)==10:break

                else:
                    start_time = time.time()
                    res = retrieve_tables(subq, faiss_index, top_k=8)
                    multihop_time += time.time() - start_time
            else:
                start_time = time.time()
                res = retrieve_tables(subq, faiss_index, top_k=8)
                multihop_time += time.time() - start_time

            retrieved_schemas = [_[0] for _ in res]
            norerank_tables = [_[-1] for _ in res]
            content_dict = {_[0]:_[-1] for _ in res}
            multihop_norerank_tables.append(norerank_tables)


            start_time = time.time()
            if rerank:
                query_tok_cnt = len(tokenizer.tokenize(subq))
                res = [(content_dict[_['content']]['table_name'], _['score']/query_tok_cnt) for _ in RAG_our.rerank(query=subq, documents=retrieved_schemas, k=min(10, len(retrieved_schemas)))]
            else:
                res = [(_[-1]['table_name'], _[1]) for _ in res]
            multihop_time += time.time() - start_time
            

            # output_tables, output_scores = [_[0].lower() for _ in res], [_[1] for _ in res]
            output_tables, output_scores = [_[0] for _ in res], [_[1] for _ in res]

            multi_hop_retrieved_tables.append(output_tables)
            multi_hop_retrieved_table_scores.append(output_scores)

        retrieved_schemas = []
        start_time = time.time()
        output_tables, output_paths = retrieve_multi_table_path(subqs, multi_hop_retrieved_tables, multi_hop_retrieved_table_scores, estimated_overlap_tabpairs, top_k=10)
        multihop_time += time.time() - start_time

        

        output_res[question] = {'single_hop':single_hop_res, 'multi_hop':output_tables, \
            'multi_hop_tables':multi_hop_retrieved_tables, \
            'multi_hop_paths':output_paths}

    print('total cost time: ', multihop_time)
    pickle.dump([output_res, singlehop_time, multihop_time], open(save_fpath, 'wb'))
