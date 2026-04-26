from langchain_core.prompts import PromptTemplate
import json
import re
import time
import pickle
import tqdm
from tqdm import tqdm 
import random
import pandas as pd
import ast
import utils
from utils import prompt, model_config
from model_config import llm
import argparse
import time
import openai


mmqa_qd_enhance_tables_prompt = PromptTemplate.from_template("""
You are an expert in multi-hop question decomposition for table-based question answering.

Your task is to decompose a complex question into a sequence of simpler sub-questions. Each sub-question must:

1. Use the key phrases grouped together in each entry from a provided list (called "word lists").
2. Preserve the full meaning and intent of the original question — do not drop important information.
3. Be minimal and non-redundant. Do not split logically inseparable reasoning steps that belong to the same word list.
4. Refer to previous sub-questions using "#1", "#2", etc., where appropriate, to support multi-hop reasoning.

Additional Notes:
- Each word list corresponds to exactly one sub-question.
- Do not merge or split word lists.
- Avoid generating overly fine-grained sub-questions that could naturally be expressed together.
- Use natural, fluent language for each sub-question.

Please return only a raw JSON object. Do not add code formatting (like triple backticks) or comments. Format: {{"Sub-questions": [...]}}

---

Here are some examples:

[Question]  
Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?  
[Word Lists]  
[['schools', 'SAT test', 'charter funded'], ['average score']]  
{{"Sub-questions": ["Which schools have an average score in Math over 560?", "How many of the schools from #1 are directly charter-funded in the SAT test?"]}}

[Question]  
Sum up the away team goal scored by both Daan Smith and Filipe Ferreira.  
[Word Lists]  
[['away team', 'goal'], ['Daan Smith', 'Filipe Ferreira']]]  
{{"Sub-questions": ["Which goals were scored by the away team?","What is the total number of goals from #1 that were scored by both Daan Smith and Filipe Ferreira?"]}}

---

Now decompose the following question:

[Question]  
{question}  
[Word Lists]  
{table_schemas}
""")

mmqa_qd_enhance_tables_prompt_two_version = PromptTemplate.from_template("""
You are an expert in multi-hop question decomposition for table-based question answering.

Your task is to decompose a complex question into exactly two simpler sub-questions. Each sub-question must:

1. Preserve the full meaning and intent of the original question — do not drop important information.
2. Be minimal and non-redundant. Do not split logically inseparable reasoning steps that belong to the same word list.
3. Refer to previous sub-questions using "#1", "#2", etc., where appropriate, to support multi-hop reasoning.

Additional Notes:
- Avoid generating overly fine-grained sub-questions that could naturally be expressed together.
- Use natural, fluent language for each sub-question.

Please return only a raw JSON object. Do not add code formatting (like triple backticks) or comments. Format: {{"Sub-questions": [...]}}

---

Here are some examples:

[Question]  
How many superheroes have blue eyes?
{{"Sub-questions": ["Which color is blue?", "How many superheros have eyes with color #1?"]}}
---

Now decompose the following question:

[Question]  
{question}  
""")


mmqa_qd_prompt = PromptTemplate.from_template("""
    You are an expert at multi-hop question decomposition, you need to decompose the given multi-hop question [Question] based on the given example. Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"Sub-questions": [...]}}

    [Question] {question}

    [Example]:
                                              
    {examples}  
""")
                                              
    

simple_qa_prompt_single = PromptTemplate.from_template("""
    You are answering a question about a research paper based on the following sections:

    {context}

    Question: {question}

    Give a concise, accurate answer based only on the information above.
""")


##########################################
#               CHAINS
##########################################


mmqa_qd_enhance_tables_prompt = prompt.mmqa_qd_enhance_tables_prompt | llm
mmqa_qd_enhance_tables_prompt_two_version = mmqa_qd_enhance_tables_prompt_two_version | llm
multi_hop_decomposition_prompt = prompt.multi_hop_decomposition_prompt | llm


##########################################
#               FUNCTIONS
##########################################

def obtain_decomposed_questions(question):
    response = multi_hop_decomposition_prompt.invoke({"question": question})
    return response.content.strip()

def obtain_decomposed_questions_enhanced(question, table_schemas):
    response = mmqa_qd_enhance_tables_prompt.invoke({"question": question, 'table_schemas':table_schemas})
    return response.content.strip()

def obtain_decomposed_questions_enhanced_two_version(question):
    response = mmqa_qd_enhance_tables_prompt_two_version.invoke({"question": question})
    return response.content.strip()


def obtain_decomposed_questions_withexamples(question, examples):
    response = mmqa_qd.invoke({"question": question, 'examples':examples})
    return response.content.strip()


def construct_query_data_enhance(pkl_fpath):
   question_table_schemas = pickle.load(open(pkl_fpath, 'rb'))
#    sampled_items = question_table_schemas.items()
   question_subqs_map = {}
   random.seed(0)
   sampled_items = random.sample(list(question_table_schemas.items()), 1000)

   for question, table_schema_item in tqdm(sampled_items):
      if table_schema_item['table_join_path'] is None:continue
    #   print(question, table_schema_item)
      if len(list(set(table_schema_item['table_join_path']))) != len(table_schema_item['table_join_path']):
        table_schema_item['table_join_path'] = list(set(table_schema_item['table_join_path']))
      try:
        sequence_table_schema_str = ', '.join([_+':'+str(table_schema_item['tables'][_]) for _ in table_schema_item['table_join_path']])
      except:
         continue
    #   print(question)
    #   print(sequence_table_schema_str)
    #   print(table_schema_item['table_join_path'])
      
      current_output = obtain_decomposed_questions_enhanced(question, sequence_table_schema_str)
    #   current_output = obtain_decomposed_questions(question)
    #   print(current_output)
      try:
        decomposed_sub_questions = json.loads(current_output)['Sub-questions']
      except:
        decomposed_sub_questions =  ast.literal_eval(current_output)['Sub-questions']

      question_subqs_map[question] = decomposed_sub_questions
    #   print(decomposed_sub_questions)
    #   print('------------------------')
   return question_subqs_map

def construct_query_data_withexamples(df, summarized_fpath):
   questions = df['question'].values.tolist()
   summarized_infos = pickle.load(open(summarized_fpath, 'rb'))
   constructed_example_str_lst = []
   for patten, quest_repr in summarized_infos.items():
       question = list(quest_repr.keys())[0]
       subqs = list(quest_repr.values())[0]

       current_str = f"[Question] {question}\n{{\"Sub-questions\": {subqs}}}"
       constructed_example_str_lst.append(current_str)
   example_str = '\n\n'.join(constructed_example_str_lst)

   question_subqs_map = {}
   for question in tqdm(questions):
      current_output = obtain_decomposed_questions_withexamples(question, example_str)
      try:
        decomposed_sub_questions = json.loads(current_output)['Sub-questions']
      except:
        decomposed_sub_questions = ast.literal_eval(current_output)['Sub-questions']
      question_subqs_map[question] = decomposed_sub_questions
    #   print(question)
    #   print(decomposed_sub_questions)
    #   break
   return question_subqs_map

def construct_query_data_similarexamples(df, similarcase_info):
   questions = df['question'].values.tolist()
   read_infos = {}
   for ques, summarized_infos in similarcase_info.items():
        constructed_example_str_lst = []
        for patten, quest_repr in summarized_infos.items():
            question = list(quest_repr.keys())[0]
            subqs = list(quest_repr.values())[0]

            current_str = f"[Question] {question}\n{{\"Sub-questions\": {subqs}}}"
            constructed_example_str_lst.append(current_str)
        example_str = '\n\n'.join(constructed_example_str_lst)
        read_infos[ques] = example_str

   question_subqs_map = {}
   for question in tqdm(questions):
      current_output = obtain_decomposed_questions_withexamples(question, read_infos[question])
      try:
        decomposed_sub_questions = json.loads(current_output)['Sub-questions']
      except:
        decomposed_sub_questions = ast.literal_eval(current_output)['Sub-questions']
      question_subqs_map[question] = decomposed_sub_questions
   return question_subqs_map

def decompose_ours(tmp):
   question_subq_dict = {}
   MAX_RETRIES = 3
   total_cost_time = 0

   for q, q_item in tqdm(tmp.items(), total=len(tmp)):
    #   if q not in df['question'].values.tolist():continue
      
      schema_qwords_map = q_item[0]
      schema_lst = q_item[1]
      output_q_words = []
      for ss in schema_lst:
          cur_q_words = []
          for s in ss:
              if schema_qwords_map[s] not in output_q_words:
                  cur_q_words.append(schema_qwords_map[s])
          output_q_words.append(cur_q_words)
      if len(schema_lst)==1:
         output_q_words = [[_] for _ in schema_lst[0]]
      output_q_words_cnt = len(output_q_words)
      output_q_words = str(output_q_words)
    #   output_q_words = json.dumps(output_q_words, ensure_ascii=False)
      
  
      start_time = time.time()
      flag = True
      try:
        current_output = obtain_decomposed_questions_enhanced(q, output_q_words)
        total_cost_time += time.time() - start_time
        try:
            decomposed_sub_questions = json.loads(current_output)['Sub-questions']
        except:
            decomposed_sub_questions =  ast.literal_eval(current_output)['Sub-questions']
      except:
        print(f"bad format for {output_q_words}")
        flag = False

      if len(decomposed_sub_questions) != output_q_words_cnt or not flag:
        print("---- not right...")
        
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"🔁 Retry attempt {attempt}...")
            current_output = obtain_decomposed_questions_enhanced_two_version(q)

            try:
                # Try direct JSON first
                decomposed_sub_questions = json.loads(current_output)['Sub-questions']
                break  # ✅ success
            except:
              try:
                  # Strip markdown/code block formatting if needed
                  cleaned_output = current_output.strip().strip('```').replace('json', '').strip()
                  decomposed_sub_questions = ast.literal_eval(cleaned_output)['Sub-questions']
                  break  # ✅ success
              except Exception as e:
                  print(f"⚠️ Parse failed on attempt {attempt}: {e}")
                  decomposed_sub_questions = []
                  time.sleep(0.5)  # optional: short delay before retry

      if not decomposed_sub_questions:
          print(f"❌ Failed to obtain valid decomposed sub-questions after {MAX_RETRIES} attempts.")
         
      question_subq_dict[q] = decomposed_sub_questions
   return question_subq_dict, total_cost_time

def decompose_previous(tmp, max_retries=3, backoff=1.0):
    question_subq_dict = {}
    total_cost_time = 0

    for q, q_item in tqdm(tmp.items(), total=len(tmp)):
        retries = 0
        success = False

        while retries < max_retries and not success:
            start_time = time.time()
            try:
                current_output = obtain_decomposed_questions(q)
                total_cost_time += time.time() - start_time

                try:
                    decomposed_sub_questions = json.loads(current_output)['Sub-questions']
                except Exception:
                    decomposed_sub_questions = ast.literal_eval(current_output)['Sub-questions']

                question_subq_dict[q] = decomposed_sub_questions
                success = True  # ✅ success, break retry loop

            except openai.BadRequestError as e:
                print(f"❌ BadRequestError for question: {q}\nReason: {e}")
                retries += 1
                time.sleep(backoff * (2 ** retries))  # exponential backoff

            except Exception as e:
                print(f"⚠️ Unexpected error for question: {q}\n{e}")
                retries += 1
                time.sleep(backoff * (2 ** retries))

        if not success:
            print(f"❌ Failed after {max_retries} attempts: {q}")
            question_subq_dict[q] = []  # fallback to empty sub-question list

    return question_subq_dict, total_cost_time

     
if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--dataset_name", required=True, type=str)
   parser.add_argument("--save_fpath", required=True, type=str)
   parser.add_argument('--decompose_type', required=True, type=str)
   args = parser.parse_args()
   dataset_name = args.dataset_name

   args.semantic_fpath = f"../../dataset/data/{dataset_name}/wordlist/{dataset_name}_meta.pkl"

   tmp = pickle.load(open(args.semantic_fpath, 'rb'))
   save_fpath = args.save_fpath

   if args.decompose_type == 'our':
     question_subq_dict, total_cost_time = decompose_ours(tmp)
   else:
     question_subq_dict, total_cost_time = decompose_previous(tmp)
   
   print(f'total time for question decomposition is {total_cost_time}')  
#    print(question_subq_dict)                         
   pickle.dump([question_subq_dict, total_cost_time], open(save_fpath, 'wb'))
