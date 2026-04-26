import utils
from utils import tool
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict
import json
import time
import argparse
import pickle


class UnionFind:
    def __init__(self):
        self.parent = dict()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

def build_table2group(estimated_overlap_tabpairs):
    uf = UnionFind()
    for t1, t2 in estimated_overlap_tabpairs:
        uf.union(t1, t2)

    group_map = defaultdict(list)
    for table in uf.parent:
        root = uf.find(table)
        group_map[root].append(table)

    table2group = {}
    for group_id, tables in enumerate(group_map.values()):
        for table in tables:
            table2group[table] = f"group{group_id}"
    
    return table2group

def convert_using_question(question, all_schema_list, nlp):

    doc = nlp(question)
    content_words = [token.text for token in doc if not token.is_stop and token.is_alpha]
    
    question_phrases = [
        ' '.join(content_words[i:j]) 
        for i in range(len(content_words)) 
        for j in range(i+1, min(i+3, len(content_words)+1))
    ]

    all_schema_q_map = {}
    for schema_list in all_schema_list:
        schema_texts = [s.replace(':', ' ') for s in schema_list]

        schema_embeddings = model.encode(schema_texts, convert_to_tensor=True)
        phrase_embeddings = model.encode(question_phrases, convert_to_tensor=True)

        for i, schema in enumerate(schema_list):
            sims = util.cos_sim(schema_embeddings[i], phrase_embeddings)[0]
            top_idx = int(sims.argmax())
            best_phrase = question_phrases[top_idx]
            best_score = sims[top_idx].item()
            all_schema_q_map[schema] = best_phrase
    return all_schema_q_map, content_words

def output_topk_schemas(score_pkl, cur_p, schema_map, k=30):
    arrays = score_pkl[cur_p]
    output_schemas = {}

    all_values = []
    for i, arr in enumerate(arrays):
        for j, val in enumerate(arr):
            all_values.append((val, i, j))
            
    # Sort and select top-k
    top_k = sorted(all_values, key=lambda x: -x[0])[:k]
    for item in top_k:
        # if item[0]<0.45:continue
        if schema_map is None:
            output_schemas[mini_schemas[item[1]][item[-1]]] = item[0]
        else:
            output_schemas[schema_map[mini_schemas[item[1]][item[-1]]]] = item[0]
    return output_schemas

def compute_best_group_v2(dev_json, decomposed_json, score_pkl, table2group, output_topk_schemas, schema_map, valid_questions, nlp, topk=50):
    """
    Args:
        decomposed_subqs: dict mapping qid -> list of sub-questions
        score_pkl: a pointer to global score data to feed output_topk_schemas
        table2group: dict mapping table name to group id (typically joinable cluster)
        output_topk_schemas: function(score_pkl, schema_cnt) -> dict of {schema_key: score}
        topk: number of top candidates to retrieve per sub-question

    Returns:
        dict: mapping qid -> list of selected schemas (one per sub-question)
    """
    result = {}
    schema_cnt = 0
    for i in range(len(dev_json)):
        qid = dev_json[i]['question']
        # if qid not in valid_questions:continue
        dq = decomposed_json[i]
        group_scores = defaultdict(lambda: defaultdict(tuple))  # group_id -> dq_index -> (schema, score)
        

        # Step 1: Get top-k schemas for each sub-question
        dq_schema_scores = []
        for j in range(len(dq)):
            schema_scores = output_topk_schemas(score_pkl, schema_cnt, schema_map)
            schema_cnt += 1
            dq_schema_scores.append(schema_scores)
            # if qid=="How many events of the Student_Club did Sacha Harrison attend in 2019?":
            #     print(j, dq[j], schema_scores)

            for schema_key, score in schema_scores.items():
                table, col = schema_key.split(":")
                group_id = table2group.get(table.lower(), table.lower())  # fallback to individual table if no group

                if j not in group_scores[group_id] or score > group_scores[group_id][j][1]:
                    group_scores[group_id][j] = (schema_key, score)

        # Step 2: Choose best-scoring group that covers all sub-questions
        best_group = None
        best_score = -1

        for group_id, dq_map in group_scores.items():
            # Extract table names from schema keys like "table:column"
            table_names = set(schema.split(":")[0] for schema, _ in dq_map.values())


            total_score = sum(score for _, score in dq_map.values())
            if total_score > best_score:
                best_score = total_score
                best_group = dq_map

        final_schemas = [best_group[j][0] for j in range(min(len(best_group),len(dq))) if j in best_group] if best_group else []

        table_grp = defaultdict(list)

        for a, b in zip(dq, final_schemas):
            table_name = b.split(":")[0]
            table_grp[table_name].append(a)
        

        if len(table_grp)==1:
            updated_table_grp = {}
            for b in list(table_grp.values())[0]:
                table_name = b.split(":")[0]
                if table_name not in updated_table_grp:
                    updated_table_grp[table_name] = []
                updated_table_grp[table_name].append(b)
            table_grp = updated_table_grp
    
        all_schema_list = list(table_grp.values())

        output, content_words = convert_using_question(qid, all_schema_list, nlp)
        result[qid] = [output, all_schema_list]

    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()


    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dataset_name = args.dataset

    basic_dir = f"../../dataset/data/{dataset_name}"
    training_data = json.load(open(f"{basic_dir}/dev.json", 'rb'))
    valid_questions = [item['question'] for item in training_data]
    tables = tool.read_json(f'{basic_dir}/dev_tables.json')
    decomposed_json = tool.read_json(f'{basic_dir}/decomp.json')
    dev_json = tool.read_json(f'{basic_dir}/dev.json')

    mode = 'full'
    if mode !='full':
        score_pkl = pickle.load(open(f'{basic_dir}/contriever/score_decomp.pkl','rb'))
        mini_schemas = tool.decompose_schema(tables)
        schema_map = None
    else:
        score_pkl = pickle.load(open(f'{basic_dir}/contriever/score_decomp_f.pkl','rb'))
        mini_schemas, _, schema_map = tool.decompose_schema_full(tables)

    jaccard_json_path = f'{basic_dir}/dev_jaccard.json'
    uniqueness_json_path = f'{basic_dir}/dev_uniqueness.json'

    col_sim_path = f'{basic_dir}/dev_exact_col_sim.json'
    col_emb_sim_path = f'{basic_dir}/dev_semantic_col_sim.json'


    estimated_overlap_tabpairs = tool.identify_overlap_tables(jaccard_json_path, uniqueness_json_path, col_sim_path, col_emb_sim_path, dataset_name)

    table2group = build_table2group(estimated_overlap_tabpairs)
    # start_time = time.time()
    all_output_res = compute_best_group_v2(dev_json, decomposed_json, score_pkl, table2group, output_topk_schemas, schema_map, valid_questions, nlp)
    # print(len(all_output_res))
    # print(time.time() - start_time)

    grp_path = f"{basic_dir}/wordlist"
    if not os.path.exist(grp_path):
        os.makedirs(grp_path)

    pickle.dump(all_output_res, open(f'{grp_path}/{dataset_name}_meta.pkl','wb'))



