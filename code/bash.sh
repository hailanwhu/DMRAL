#!/bin/bash

dataset_name=''

# Step into the preprocessing directory
cd code/preprocessing
python dump_raw_table.py --dataset "$dataset_name"
python clean_table.py --dataset "$dataset_name"
python table_join_rel.py --dataset "$dataset_name"
python table_union_rel.py --dataset "$dataset_name"

# Step into the decomposer directory
cd ../decomposer

# Run query decomposition
python contriever.py --dataset "$dataset_name" --query_decompose

# Group information needs
python group_information_needs.py --dataset "$dataset_name"

# Create output directory
mkdir -p "../$dataset_name/output"

# Run question decomposition
python question_decomposition.py \
  --dataset_name "$dataset_name" \
  --save_fpath "../$dataset_name/output/question_decomposed_dev_updated.pkl" \
  --decompose_type our

# Step into the retriever directory
cd ../retriever

# Run multi-table retrieval with reranking
python multitab_retrieve.py \
  --decp_fpath "../$dataset_name/output/question_decomposed_dev_updated.pkl" \
  --dataset_name "$dataset_name" \
  --rerank \
  --save_fpath "../$dataset_name/output/multitab_rerank.pkl"

# Rewrite queries with missing info
python my_query_rewrite_table.py \
  --dataset_name "$dataset_name" \
  --qtab_fpath "../$dataset_name/output/multitab_rerank.pkl" \
  --qsubqs_fpath "../$dataset_name/output/question_decomposed_dev_updated.pkl" \
  --save_fpath "../$dataset_name/output/rerank_missing_infos.pkl"

# Retrieve complementary tables
python complement_table_retrieve.py \
  --dataset_name "$dataset_name" \
  --missing_tab_fpath "../$dataset_name/output/rerank_missing_infos.pkl" \
  --qtab_fpath "../$dataset_name/output/multitab_rerank.pkl" \
  --save_fpath "../$dataset_name/output/rerank_missingtabs.pkl"

# Step into the reasoner directory
cd ../reasoner

# Run the reasoning module
python our_reasoning.py \
  --dataset_name "$dataset_name" \
  --rerank \
  --recall \
  --refine \
  --topk 5 \
  --sql_saved_path "../$dataset_name/output/myqdfull_sql_5.pkl"
