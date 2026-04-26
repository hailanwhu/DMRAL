import pandas as pd
import glob
import tqdm
from tqdm import tqdm
from collections import defaultdict
import re
import json
import argparse
from uuid import uuid4
from llm_tool import run_structured_prompt
from utils import read_json, write_json


def remove_trailing_number(s):
    # Match ending digits and remove them if found
    return re.sub(r'\d+$', '', s)

def serialize_schema(table_name, schema):
    return f"Table: {table_name}\nColumns: {schema}"



def process_dataset(dataset_name):
    # Locate files
    fls = glob.glob(f'../../dataset/datalake/{dataset_name}/*.csv')

    all_dict = {}
    processed_table_names = []

    # Process each CSV file
    for f in tqdm(fls, total=len(fls)):
        updated_title_name = f.split('/')[-1][:-4]
        df = pd.read_csv(f)
        
        headers = list(df.columns)

        db_id = updated_title_name.split('#sep#')[0]

        column_types = [
            "number" if col in df.select_dtypes(include='number').columns else "text"
            for col in df.columns
        ]
        title_name = updated_title_name.split('#sep#')[-1]

        table_dict = {
            "db_id": db_id,
            "table_name_original": title_name,
            "table_name": ' '.join(title_name.split('_')),
            "column_names_original": headers,
            "column_names": [' '.join(_.split('_')) for _ in headers],
            "column_types": column_types
        }

        all_dict[updated_title_name] = table_dict
        # print(table_dict)

    # Save output
    output_path = f"../../dataset/data/{dataset_name}_dlr/dev_tables.json"
    with open(output_path, "w") as outfile:
        json.dump(all_dict, outfile, indent=2)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process table files and recover metadata for a given dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'spider')")
    args = parser.parse_args()

    process_dataset(args.dataset)


if __name__ == "__main__":
    main()