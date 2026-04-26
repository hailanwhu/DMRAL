import utils
from utils import prompt, model_config, tool
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
import openai
from langchain_core.prompts import PromptTemplate
import os

##########################################
#               PROMPTS
##########################################

mmqa_sql_prompt_single = PromptTemplate.from_template("""
You are an expert in SQL generation for question answering over structured tables.

Given:
A natural language question;
One table schema;
Any relevant external knowledge

Your task is to generate a correct SQL query that answers the question using the provided table schema and knowledge.

[Constraints]
- You must only use the columns and table names exactly as they appear in the provided schema — no guessing or inference
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
                                               
Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"SQL": [...]}}

[Example]:
[Question] What is the most common type of grade span served in the city of Adelanto?       
[Table Schema] schools (CDSCode, NCESDist, NCESSchool, StatusType, City, School, GSoffered, GSserved, Virtual)               
[External Knowlege] 
{{"SQL": "SELECT GSserved FROM schools WHERE City = 'Adelanto' GROUP BY GSserved ORDER BY COUNT(GSserved) DESC LIMIT 1"}} 
                                               
Question] How many of the schools from #1 are exclusively virtual?       
[Table Schema] schools (CDSCode, NCESDist, NCESSchool, StatusType, City, School, GSoffered, GSserved, Virtual)              
[External Knowlege] Exclusively virtual refers to Virtual = 'F'
{{"SQL": "SELECT COUNT(DISTINCT School) from schools where Virtual = 'F'"}} 
  

[Input]:
[Question] {question}
[Table Schema]
{table_schemas}    
[External Knowlege] {knowledge}                                                                                                                                                                     
""")

mmqa_multi_subsqls_combine = PromptTemplate.from_template("""
You are an expert in generating SQL for multi-table question answering over structured tabular data.

Given:
- A natural language question
- A set of retrieved tables with their schemas
- A set of decomposed sub-questions (which may be inaccurate or incomplete)
- Optional external knowledge (e.g., mappings between question phrases and schema columns)

【Your Task】
1. Carefully read the original question and the retrieved table schemas.
2. Review the provided decomposed sub-questions. If they are inaccurate, incomplete, or not aligned with the retrieved tables, revise or rewrite them based on the question and schema.
3. For each revised sub-question, generate the corresponding SQL query using only the retrieved tables and any external knowledge.
4. If the final sub-question's SQL fully answers the original question, that SQL is the final output.  
   Otherwise, compose a final SQL query that integrates multiple sub-queries to answer the question completely.

【Constraints】
- In `SELECT <column>`, include only the columns necessary for answering the question.
- In `FROM <table>` or `JOIN <table>`, avoid including unnecessary tables.
- When using aggregation functions like `MAX()` or `MIN()`, perform all necessary joins first.
- If a column contains values like `'None'` or `None`, use filters like `WHERE <column> IS NOT NULL`.
- When using `ORDER BY`, include `GROUP BY` if required to ensure distinctness or correct aggregation.
- Do not reference any columns or tables that are not in the provided table schemas.
- Ensure the final output is a valid JSON object.

【Output Format】
Return your result as a JSON object with the following keys:
- "reasoning": Your step-by-step reasoning including any revised sub-questions and intermediate SQL if applicable.
- "Final SQL": The final SQL query string that answers the question.

Do not include any explanation or formatting outside the JSON. Do not wrap in triple backticks.

[Example]
[Retrieved Tables]
enrollment(enrollment_id, student_id, course_id) 
student(student_id, name, age, gender)                                                         
[Question] List the names of female students who have enrolled in more than 3 courses.
[External Knowledge]
"female" refers to gender = 'F'
[Decomposed Sub-questions]
['Which students have enrolled in more than 3 courses?', 'What are the names of female students from #1? ']
{{
  "reasoning": "Step 1: Revised sub-question 1: Which students have enrolled in more than 3 courses? SQL: SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3. Step 2: Revised sub-question 2: What are the names of female students from #1? SQL: SELECT name FROM student WHERE student_id IN (...) AND gender = 'F'.",
  "Final SQL": "SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3) AND gender = 'F'"
}}

========

[Retrieved Tables]
{table_schemas}
[Question]
{question}
[External Knowledge]
{knowledge}
[Decomposed Sub-questions]
{decomposed_subqs}
""")




mmqa_sql_prompt_multiple = PromptTemplate.from_template("""
You are an expert in generating SQL for multi-table question answering over structured tabular data.

Given:
- A natural language question
- A set of decomposed sub-questions, each paired with its corresponding table schema
- Optional external knowledge (e.g., mappings between question phrases and schema columns)

Your task:
Think step by step to generate an accurate final SQL query that answers the original question using multiple tables.

Instructions:
1. For each sub-question:
   - Carefully examine the sub-question and its associated schema.
   - Use any available external knowledge to help identify the correct columns (e.g., "number of test takers" → "NumTstTakr").
   - Generate an appropriate SQL snippet that retrieves the necessary information.
2. After generating SQL snippets for all sub-questions:
   - Reflect on how to combine them logically.
   - Ensure that the final composed SQL query satisfies the overall question intent and information need.

Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"SQL": [...]}}

[Example]:
[Question] List the names of female students who have enrolled in more than 3 courses.

[Decomposed Sub-questions and Schemas]
Sub-question 1: Which students have enrolled in more than 3 courses?
Schema: enrollment(enrollment_id, student_id, course_id)

Sub-question 2: What are the names of female students from #1?  
Schema: student(student_id, name, age, gender)

[External Knowledge]
"female" refers to gender = 'F'

Chain of Thought:
Step 1: Generate SQL1 for Sub-question 1 using schema + external knowledge  
SQL1: SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3

Step 2: Generate SQL2 for Sub-question 2 using schema + external knowledge  
SQL2: SELECT name FROM student where gender = 'F'

Step 3: Compose final SQL query and verify it meets the original question intent  
Final SQL: SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3) and gender = 'F'

Output:
{{"SQL": ["SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3) and gender = 'F'"]}}

[Input]:
[Question] {question}

[Decomposed Sub-questions and Schemas]
{subqs_schemas}

[External Knowlege] 
{knowledge} 
""")

sql_refine_prompt = PromptTemplate.from_template("""
【Instruction】
You are a SQL correction assistant for multi-table question answering over structured tabular data. The SQL query below encountered execution errors. 

Your task is to revise the SQL query using multiple tables based on the provided question, table schemas, and error message.
You may reason step by step if needed.
After generating the SQL, verify that it aligns with the original question. Include verifiable evidence or justification if relevant.
Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"SQL": [...]}}
                                                                                               
[Constraints]
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
[Question]
{question}
[Evidence]
{evidence}
[Table Schemas]
{table_schemas}
[Error SQL]
{sql}
[SQLite error] 
{sqlite_error}

Now please fixup old SQL and generate new SQL again.
[correct SQL]
""")


# TODO: prompt related the questions to extract the inforamtion

##########################################
#               CHAINS
##########################################


mmqa_sql_prompt_single = mmqa_sql_prompt_single | llm
mmqa_sql_prompt_multiple = mmqa_sql_prompt_multiple | llm
sql_refine_prompt = sql_refine_prompt | llm
mmqa_multi_subsqls_combine = mmqa_multi_subsqls_combine | llm


##########################################
#               FUNCTIONS
##########################################
def obtain_sql(question, related_tables, evidence):
    response = mmqa_sql_prompt_single.invoke({"question": question, "table_schemas":related_tables, "knowledge": evidence})
    return response.content.strip()

def combine_sqls(related_tables, question, decomposed_subqs, evidence):
    response = mmqa_multi_subsqls_combine.invoke({"table_schemas":related_tables, "question": question, "decomposed_subqs":decomposed_subqs, "knowledge": evidence})
    return response.content.strip()

def safe_combine_sqls(tables, question, subqs, knowledge, max_retries=3, retry_delay=1):
    for attempt in range(1, max_retries + 1):
        try:
            return combine_sqls(tables, question, subqs, knowledge)
        except openai.BadRequestError as e:
            print(f"⚠️ BadRequestError on attempt {attempt} for question: {question}")
            print("Sub-questions (truncated):", str(subqs)[:300])
            print("Knowledge (truncated):", str(knowledge)[:300])
            if attempt == max_retries:
                raise e
            time.sleep(retry_delay)
        except Exception as e:
            print(f"❌ Unexpected error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise e
            time.sleep(retry_delay)

def obtain_sql_multi(question, subqs_schemas, evidence):
    response = mmqa_sql_prompt_multiple.invoke({"question": question, "subqs_schemas":subqs_schemas, "knowledge": evidence})
    return response.content.strip()

def refine_sql(question, evidence, table_schemas, sql, sql_error):
    response = sql_refine_prompt.invoke({"question": question, \
                                         "evidence":evidence, \
                                        "table_schemas": table_schemas, "sql":sql, "sqlite_error":sql_error })
    return response.content.strip()

def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def get_corpus_schema(tables_path):
  tables = read_json(tables_path)
  table_schemas = {}
  for ts, tval in tables.items():
    table_schemas[ts] = [tval['column_names_original'], tval['table_name_original']]

  return table_schemas

def get_question_evidence(input_path):
  corpus = read_json(input_path)
  question_evds = {}
  for tval in corpus:
    question_evds[tval['question']] = tval['evidence']

  return question_evds

def combine_related_tabs(related_tables, table_schemas):
  subq_solution_dict = []
  for subq, subr in zip(range(len(related_tables)), related_tables):
    # tab_str = f"{table_schemas[subr][1]} ({', '.join(table_schemas[subr][0])})"
    tab_str = f"{subr.replace('#sep#', '_')} ({', '.join(table_schemas[subr][0])})"
    subq_solution_dict.append(tab_str)
  return '\n'.join(subq_solution_dict)

def get_question_goldsql(input_path):
  corpus = read_json(input_path)
  question_evds = {}
  for tval in corpus:
    question_evds[tval['question']] = tval['sql']

  return question_evds

def iter_all_subqs(subqs, related_tables, evidence, table_schemas):
  subq_solution_dict = {}
  for subq, subr in zip(subqs, related_tables):
    tab_str = f"{table_schemas[subr][1]} ({', '.join(table_schemas[subr][0])})"
    current_output = obtain_sql(subq, tab_str, evidence)

    try:
      generated_sql = json.loads(current_output)['SQL']
    except:
      try:
        generated_sql =  ast.literal_eval(current_output)['SQL']
      except:
        print(f'error---{current_output}')
        generated_sql = current_output

    subq_solution_dict[subq] = {'table_schema':tab_str, 'sql': generated_sql}
  return subq_solution_dict
    

  
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", default='bird', type=str)
  parser.add_argument("--rerank", action='store_true', help="rerank the retrieved tables") #rerank or norerank
  parser.add_argument('--recall', action='store_true', help="add the missing tables")
  parser.add_argument('--refine', action='store_true', help="refine the sql")
  parser.add_argument('--raw', action='store_true', help="whether use the raw decomposed subquestions")
  # parser.add_argument("--topk", default = 5, type=int)
  parser.add_argument("--topk", required=True, type=int)
  parser.add_argument("--sql_saved_path", required=True, type=str)
  args = parser.parse_args()
  dataset_name = args.dataset_name
  topk = args.topk
  sql_saved_path = args.sql_saved_path

  method = 'myqdcomplement'


  if args.rerank:
    load_table_path = f'../{dataset_name}/output/multitab_rerank.pkl'
  if args.raw:
    load_table_path = f'../{dataset_name}/output/multitab_rerank_raw.pkl'
  q_table_map = pickle.load(open(load_table_path, 'rb'))
  if type(q_table_map) == list:
     q_table_map = q_table_map[0]

  iterate_tab_dict = {}
  decom_fpath = f'../{dataset_name}/output/question_decomposed_dev_updated.pkl'
  if args.raw:
    decom_fpath = f'../{dataset_name}/output/question_decomposed_dev_raw_test.pkl'
  subq_content = pickle.load(open(decom_fpath, 'rb'))
  if type(subq_content) == list:
    subq_content = subq_content[0]

  if args.rerank:
    missing_tab_path = f"../{dataset_name}/output/rerank_missingtabs.pkl"
  if args.raw:
    missing_tab_path = f"../{dataset_name}/output/rerank_missingtabs_raw.pkl"
  missing_tab_map = pickle.load(open(missing_tab_path, 'rb'))
  if type(missing_tab_map) == list:
    missing_tab_map = missing_tab_map[0]

  tables_path = f"../../dataset/data/{dataset_name}/dev_tables.json"
  
  datalake_path = f"../../dataset/data/initial_dbs/{dataset_name}/"

  if dataset_name in ['bird', 'bird_dlr']:
    q_ev_path = f"../../dataset/data/{dataset_name}/dev.json"
    q_ev_dict = get_question_evidence(q_ev_path)
  else:
    q_ev_dict = {}
  table_schemas = get_corpus_schema(tables_path)


  quest_sql_dict, quest_reasoning_dict = {}, {}
  start_time = time.time()
  for q, retrieved_tabs in tqdm(q_table_map.items(), total=len(q_table_map)):
    singlehop_tabs = retrieved_tabs['multi_hop'][:topk]

    if method in['myqdcomplement']:
      ss = singlehop_tabs
      if args.recall:
        if q in missing_tab_map:
            for _ in missing_tab_map[q]:
             if _ not in ss:ss.append(_)

      retrieved_tables = combine_related_tabs(list(set(ss)), table_schemas)
      current_output = safe_combine_sqls(retrieved_tables, q, subq_content[q], q_ev_dict.get(q, ''))

      try:
          generated_sql = json.loads(current_output)['Final SQL']
          reasoning_procedure = json.loads(current_output)['reasoning']
      except:
          try:
            generated_sql =  ast.literal_eval(current_output)['Final SQL']
            reasoning_procedure = ast.literal_eval(current_output)['reasoning']
          except:
            print(f'error---{current_output}')
            generated_sql = current_output

      if not generated_sql:
        print(f"⚠️ Skipping question {q} due to missing 'Final SQL' in parsed output.")
        continue

      """
      # match = re.search(r"\*\*Final SQL\*\*:\s*```(?:sql)?\s*(.*?)\s*```", current_output, re.DOTALL)
      # match = re.search(r"\*\*Final SQL\*\*.*?:\s*```(?:sql)?\s*(.*?)\s*```", current_output, re.DOTALL)
      match = re.search(r"\*\*Final SQL\*\*.*?\n+```(?:sql)?\n(.*?)\n```", current_output, re.DOTALL)

      # Extracted result
      if match:
          generated_sql = [match.group(1).strip()]
          # print("✅ Extracted Final SQL:\n")
          # print(generated_sql[0])
      else:
          print("❌ Final SQL block not found.")
          print(current_output)
          continue
      """

    if args.refine:
        sql_to_execute = generated_sql if isinstance(generated_sql, str) else generated_sql[0]
        sql_to_execute = sql_to_execute.replace('#sep#', '_')
        tab_df_dict = {}

        for cur_tab in singlehop_tabs:
          if cur_tab in iterate_tab_dict:
            cur_tab_df = iterate_tab_dict[cur_tab]
          else:
            cur_tab_fpath = os.path.join(datalake_path, f"{cur_tab}.csv")
            cur_tab_df = pd.read_csv(cur_tab_fpath)
            iterate_tab_dict[cur_tab.replace('#sep#', '_')] = cur_tab_df
          tab_df_dict[cur_tab.replace('#sep#', '_')] = cur_tab_df
          

        status_flag, error = tool.execute_sql_on_dataframes(sql_to_execute, tab_df_dict)

        if status_flag == -1 or pd.isna(status_flag):
            updated_output = refine_sql(q, q_ev_dict.get(q, ''), retrieved_tables, generated_sql, error)
            try:
                generated_sql = json.loads(updated_output)['SQL']
            except:
                try:
                    generated_sql =  ast.literal_eval(updated_output)['SQL']
                except:
                    print(f'error---{updated_output}')
                    generated_sql = updated_output
            print('refine', generated_sql, q)
            print(error)
        else:
            print('pass',  generated_sql, q)
    quest_sql_dict[q] = generated_sql
    quest_reasoning_dict[q] = reasoning_procedure
    
  total_time = time.time() - start_time
  print(f'total time {total_time}')

  pickle.dump([quest_sql_dict, total_time, quest_reasoning_dict], open(sql_saved_path, 'wb') )