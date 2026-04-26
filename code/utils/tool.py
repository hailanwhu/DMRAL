import duckdb
import json
import pickle

def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def decompose_schema(tables):
  r = []
  for t in tables:
    cur_r = []
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for t_col in t_cols:
      cur_r.append(f'{t_name}:{t_col}')
    r.append(cur_r)
  return r

def decompose_schema_full(tables):
  r, col_nums = [], []
  map_dict = {}
  for t in tables:
    cur_r = []
    f_t_name, f_t_cols = tables[t]['table_name'], tables[t]['column_names']
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for f_t_col, t_col in zip(f_t_cols, t_cols):
      cur_r.append(f'{f_t_name}:{f_t_col}')
      map_dict[f'{f_t_name}:{f_t_col}'] = f'{t_name}:{t_col}'
    col_nums.append(len(t_cols))
    r.append(cur_r)
  return r, col_nums, map_dict
  
def execute_sql_on_dataframes(sql_query, tab_df_dict):
    con = duckdb.connect()
    try:
        for table_name, df in tab_df_dict.items():
            clean_name = table_name.replace('#sep#', '')
            con.register(clean_name, df)

        result_df = con.execute(sql_query).fetchdf()
        return result_df.iloc[0, 0], None  # Return result and no error

    finally:
        con.close()

# identify the tables with joinability
def identify_overlap_tables(jaccard_json_path, unique_json_path, csim_json_path, cemb_sim_json_path, dataset_name, sim_threshold=0.90, unique_threshold=0.98):
    tmp_str = json.load(open(jaccard_json_path, 'r'))
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
                # if '#sep#satscores#' in tab_pair and '#sep#frpm#' in tab_pair:
                #     print(tab_pair, score, col_sim, cemb_sim)
                
                if cemb_sim<cemb_threshold_dict[dataset_name]:continue
                

                all_tab_names = []
                for i in range(len(tab_arrs)):
                    # all_tab_names.append(tab_arrs[i])
                    if '#sep#' in tab_arrs[i]:
                        all_tab_names.append('#sep#'.join( tab_arrs[i].split('#sep#')[:-1] ))
                if len(all_tab_names) != 2:
                    print('error ---')
                    print(all_tab_names, tab_arrs, tab_pair)
                    continue            
                # assert len(all_tab_names) == 2
                
                updated_tab_pair = sorted(all_tab_names)
                if updated_tab_pair not in valid_tab_pairs:
                    valid_tab_pairs.append(updated_tab_pair)

    return valid_tab_pairs