from langchain_core.prompts import PromptTemplate


##########################################
#               PROMPTS
##########################################

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

recover_column_headers = PromptTemplate.from_template(
      """You are given a table with partial metadata. Some column headers or the table title may be missing.

         Table Title: "{title_name}"
         Column Headers: {headers}
         Sample Rows (10 randomly sampled rows):
         {sampled_rows}

         Please infer the most likely missing column names or title elements.

         ⚠️ **Important Instructions**:
         - You must return exactly the same number of column headers as provided in the "Column Headers" list above.
         - Do not remove or add columns. Just replace ambiguous or missing ones.
         - Keep the order of the headers aligned with the input.

         Respond strictly in valid JSON format with the following structure:
         {{
         "updated_title": "...",
         "updated_headers": ["...", "...", ...]
         }}

         ❌ Do not include explanations, comments, or anything outside the JSON.
         ✅ Ensure the length of "updated_headers" matches the input headers exactly.
         """
    )


mmqa_sql_prompt = PromptTemplate.from_template("""
You are an expert in SQL generation for question answering over structured tables.

Given:
A natural language question;
One or more table schemas;
Any relevant external knowledge

Your task is to generate a correct SQL query to answer the question by following these steps:
1.Identify which tables contain the relevant columns.
2.Join the necessary tables if needed using only columns present in the tables.
3.Decide which columns to include in the SELECT clause to answer the original question.

Ensure the SQL is syntactically correct, and refer to columns using their associated tables to prevent ambiguity.
Be sure to think step by step and incorporate external knowledge if provided.  
                                               
Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"SQL": [...]}}

[Example]:
[Question] What is the grade span offered in the school with the highest longitude?        
[Table Schemas] 
frpm(CDSCode, Academic Year, County Code, District Code, School Code, Low Grade, High Grade)
satscores(cds, rtype, sname, dname)
schools(CDSCode, NCESDist, NCESSchool, Longitude, GSoffered, GSserved)
[External Knowlege] the highest longitude refers to the school with the maximum absolute longitude value. 
{{"SQL": "SELECT GSoffered FROM schools ORDER BY ABS(longitude) DESC LIMIT 1"}} 

[Question] What is the Percent (%) Eligible Free (K-12) in the school administered by an administrator whose first name is Alusine. List the district code of the school.   
[Table Schemas] 
frpm(CDSCode, Academic Year, County Code, District Code, School Code, Low Grade, High Grade, Free Meal Count (K-12), Enrollment (K-12))
satscores(cds, rtype, sname, dname)
schools(CDSCode, NCESDist, NCESSchool, Longitude, GSoffered, GSserved, AdmFName1)
[External Knowlege] Percent (%) Eligible Free (K-12) = `Free Meal Count (K-12)` / `Enrollment (K-12)` * 100%
{{"SQL": "SELECT T1.`Free Meal Count (K-12)` * 100 / T1.`Enrollment (K-12)`, T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.AdmFName1 = 'Alusine'"}}

[Input]:
[Question] {question}
[Table Schemas]
{table_schemas}    
[External Knowlege] {knowledge}                                                                                                                                                                     
""")

collect_missing_table_prompt = PromptTemplate.from_template("""
Given the following SQL tables, your job is to complete the possible left SQL tables given a user’s request.
Return None if no left SQL tables according to the user’s request.
Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"Completing Tables": [...]}}
                                                            
[Example]                                         
Quetsion: What was the growth rate of the total amount of loans across all accounts for a male client between 1996 and 1997?
Database: financial.client(client_id, gender, birth_date, location of branch) \n financial.loan(loan_id, account_id, date, amount, duration, monthly payments, status)
{{"Completing Tables": ["financial.account(account id, location of branch, frequency, date)", "financial.disp(disposition id, client_id, account_id, type)"]}}

Question: How many members did attend the event 'Community Theater' in 2019?
Database: student_club.Attendance(link to event, link to member) \n student_club.race(raceId, location, status, date)
{{"Completing Tables": ["student_club.Event(event id, event name, event date, type, notes, location, status)"]}}

Question: What's the Italian name of the set of cards with \"Ancestor's Chosen\" is in?
Database: card_games.cards(unique id number identifying the cards, artist, ascii Name, availability, border Color, card Kingdom Foil Id, card Kingdom Id, color Identity, color Indicator, colors, converted Mana Cost) \n card_games.set_translations(id, language, set code, translation)
{{"Completing Tables": None}}

Question: Which user have only one post history per post and having at least 1000 views?
Database: codebase_community.postHistory(Id, Post History Type Id, Post Id, Creation Date, User Id, Text, Comment, User Display Name)
{{"Completing Tables": ["codebase_community.users(Id, Reputation, Creation Date, Display Name, Website Url, Location, Views, UpVotes, DownVotes, Account Id, Profile Image Url)"]}}

Question: Among the atoms that contain element carbon, which one does not contain compound carcinogenic?
Database: toxicology.atom(atom id, molecule id, element) \n toxicology.molecule(molecule id, label)
{{"Completing Tables": None}}
                                                            
Question: What is the percentage of the customers who used AUS in 2012/8/25?
Database: debit_card_specializing.transactions_1k(Transaction ID, Date, Time, Customer ID, Card ID, Gas Station ID, Product ID, Amount, Price) \n debit_card_specializing.order(order_id, account_id, bank_to, account_to, amount, k_symbol)
{{"Completing Tables": ["debit_card_specializing.customers(CustomerID, client segment, Currency)"]}}

Question: How many female clients opened their accounts in Jesenik branch?
Database: financial.client(client_id, gender, birth_date, district_id) \n financial.disp(disposition id, client_id, account_id, type)
{{"Completing Tables": ["financial.district(district_id, district_name)"]}}

Question: Please list the leagues from Germany.
Database: european_football_2.League(id, country id, name)
{{"Completing Tables": ["european_football_2.Country(id, name)"]}}

Question: What sex is the patient who in a medical examination was diagnosed with PSS and in a laboratory examination had a blood level of C-reactive protein de 2+, createnine 1 and LDH 123?
Database: Patient(id, sex, birthday, admission, diagnosis) \n Laboratory(id, date, GOT, GPT, CRP, CRE, LDH)
{{"Completing Tables": ["Examination(id, examination date, diagnosis, symptoms)"]}}

[Input]
Question: {question}
Database: {database} 
""")

subqs_rewrite_prompt = PromptTemplate.from_template("""
You are an expert in multi-step reasoning for structured question answering.

Your task is to revise a list of decomposed sub-questions (maximum of three) by updating, adding, or merging existing sub-questions so that:
0. Use the provided sampled row(s) and table schemas to verify which values and filters are directly accessible. If a condition in a sub-question cannot be supported by the table content (e.g., filtering on a name when only IDs are available), insert an additional sub-question to retrieve the required linking information.
1. Each sub-question must refer to only one table.
2. Aggregation operations (e.g., count, sum, average) should be included in a sub-question only if the required information exists entirely in a single table.
3. Sub-questions must be clearly aligned with the retrieved tables. If the original decomposition implicitly relies on multiple tables, split it into smaller single-table-aligned sub-questions.
4. Do **not** list final answer calculation (e.g., "how many", "average") as a separate sub-question — because it reflects a reasoning step, not a retrieval step. **Instead, merge it into the last sub-question that accesses the necessary table and supports aggregation.**
5. The final set of revised sub-questions should fully cover the information needs of the original question. Every necessary lookup, filtering, or counting step should be explicitly supported by one of the sub-questions.
6. You must carefully verify that the fields referenced in each sub-question exist in the retrieved tables.
7. Do not assume entity names (e.g., event names, student names) are directly usable unless there is a column for them in the same table. Always check whether a value requires mapping (e.g., name → ID) and insert a sub-question to resolve it via the appropriate table.
8. If a sub-question refers to the output of a previous sub-question, use the format `#<number>` (e.g., `#1`, `#2`) to denote the dependency. Do not repeat the original entity (e.g., do not say "Lewis Hamilton" again if you already resolved the `driverId` in #1 — refer to it as `driver from #1` instead).
9. You must still include a sub-question that describes the required step (e.g., resolving an event name to its ID), even if the relevant table is not among the retrieved tables. If needed, refer generically to “a table that contains X” without naming a specific table.
10. When a value such as 'Female', 'Marvel Comics', or a named entity is needed for filtering, always retrieve its corresponding ID from the smallest available table (e.g., gender) before using it in a larger linking or fact table. Do not apply filters directly to large or relational tables without resolving ID mappings first.

**Important**: If the original question asks for a count/average, and this can be computed within a single table, you must express this directly in the relevant sub-question (e.g., "How many X meet condition Y using only table Z?").
                                               
Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"Sub-questions": [...]}}

[Examples]
[Question]    
How many points did Lewis Hamilton get in total in the 2020 Belgian Grand Prix?

[Original Decomposed Sub-questions]  
["Which race is 2020 Belgian Grand Prix?", "How many points did Lewis Hamilton get in total for #1?"]

[Retrieved Tables]  
results(resultId, raceId, driverId, points) -- one sampled row (1, 1, 1, 10)
race(raceId, raceName, year) -- one sampled row (1, Belgian Grand Prix, 2020)
                                                    
{{"Sub-questions": ["What is the driverId of Lewis Hamilton?", "What is the raceId of 2020 Belgian Grand Prix?", "What is the total number of points for the driver from #1 in race #2?"]}}


[Question]    
What is the average lap time of drivers in the 2020 Belgian Grand Prix?

[Original Decomposed Sub-questions]  
["Which race corresponds to the 2020 Belgian Grand Prix?", "What is the average lap time of all drivers in the race from #1 using laptimes and results?"]

[Retrieved Tables]  
races(raceId, year, name) -- one sampled row (1, 2020, 'Belgian Grand Prix')
laptimes(raceId, driverId, lap, time) -- one sampled row (1, 1, 1, '1:49.088')
 
{{"Sub-questions": ["What is the raceId of the 2020 Belgian Grand Prix", "What is the average lap time of all drivers in race #1?"}}


[Question]  
What is the average fastest lap time of the top 10 drivers in the 2006 United States Grand Prix?

[Original Decomposed Sub-questions]  
["Who were the top 10 drivers in the 2006 United States Grand Prix?", "What were the fastest lap times for each driver from #1?", "What is the average of the lap times from #2?"]

[Retrieved Tables]  
results(raceId, driverId, positionOrder, fastestLapTime) -- one sampled row (1, 1, 1, '1:49.088')
races(raceId, name, year) -- one sampled row (1, 'Belgian Grand Prix', 2020)

{{"Sub-questions": ["What is the raceId of the 2006 United States Grand Prix", "What is the average fastest lap time of the top 10 drivers by positionOrder in race #1?"}}

[Input]
[Question]  
{question}

[Original Decomposed Sub-questions]  
{subqs}

[Retrieved Tables]  
{table_schemas}
                                                                                                                                                                   
""")

multi_table_python_prompt = PromptTemplate.from_template("""
You are an expert Python developer specializing in data analysis using pandas.

Your task is to write a Python function named `execute_query_with_row_trace(tab_df_dict)` that answers the given natural language question by processing the provided dataframes.

Inputs:
- `question`: A natural language question. {question}
- `table_schemas`: A dictionary mapping table names to their schemas. Each schema is a list of column names. {tab_schema_dict}
- `table_dataframes`: A dictionary mapping table names to their corresponding pandas DataFrames. Each DataFrame is accessible via `tab_df_dict[table_name]`.
- `external_knowledge`: Additional information that may be necessary to answer the question. {external_knowledge}

Requirements:
1. Analyze the `question` and `external_knowledge` to determine the necessary operations and the relevant tables.
2. Utilize the `table_schemas` to understand the structure of each table.
3. Access the data using the `table_dataframes` provided in `tab_df_dict`.
4. Perform the necessary data processing using pandas to answer the question.
5. Ensure that the function returns the final result directly.

Guidelines:
- Do not include any import statements; assume all necessary libraries are already imported.
- Do not include any code outside the function definition.
- Do not include any explanations or comments; only provide the function code.
- Ensure that the function is self-contained and does not rely on any external variables or state.

Output Format:
Provide the output strictly as a JSON object in the following format:
{{"code": "<python_function_code>"}}
""")

multi_hop_decomposition_prompt = PromptTemplate.from_template("""
You are an expert at multi-hop question decomposition, you need to decompose the given multi-hop question [Question] based on the given example. Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"Sub-questions": [...]}}

[Question] {question}
""")


sql_to_python_prompt = PromptTemplate.from_template("""
You are a Python expert skilled in translating SQL into Pandas.

Given a SQL query, write a Python function named `execute_query_with_row_trace(tab_df_dict)` that executes the query logic using Pandas over a dictionary of DataFrames.

Inputs:
- SQL Query: {sql}
- `tab_df_dict` is a dictionary where each key is a table name, and each value is a pandas DataFrame.

Guidelines:
- Use only the data available in `tab_df_dict`.
- Assume necessary libraries are already imported.
- Do not include any imports or explanations.
- Return the final result from the function.
- Access DataFrames using `tab_df_dict['table_name']`.
- Use joins, filtering, and aggregation as needed to match the SQL logic.

Please output only the raw JSON object without any explanation or formatting — do not wrap the output in triple backticks or add a language label. Format: {{"code": "<python function>"}}
""")

mmqa_multi_subsqls_combine_before = PromptTemplate.from_template("""
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

[Example]:
[Retrieved Tables]
enrollment(enrollment_id, student_id, course_id) 
student(student_id, name, age, gender)                                                         
[Question] List the names of female students who have enrolled in more than 3 courses.
[External Knowledge]
"female" refers to gender = 'F'
[Decomposed Sub-questions]
['Which students have enrolled in more than 3 courses?', 'What are the names of female students from #1? ']

Step-by-step SQL generation:                                                         
**Revised Sub-question 1**: : 'Which students have enrolled in more than 3 courses?'
**SQL**:
```
SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3
```                                                                                                              
**Revised Sub-question 2**:  'What are the names of female students from #1?'
**SQL**:
```
SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3) and gender = 'F'
```
Question Solved. 
**Final SQL**:
```
SELECT name FROM student WHERE student_id IN (SELECT student_id FROM enrollment GROUP BY student_id HAVING COUNT(course_id) > 3) and gender = 'F'
```                                                     
========


[Input]:
[Retrieved Tables]
{table_schemas}
[Question] {question}
[External Knowlege] 
{knowledge} 
[Decomposed Sub-questions]
{decomposed_subqs}
                                                          
Step-by-step SQL generation: 
""")