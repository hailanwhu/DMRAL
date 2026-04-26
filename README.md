# DMRAL

This repository contains the code and benchmarks for the paper **Decomposition-Driven Multi-Table Retrieval and Reasoning for Numerical Question Answering**.

---

## 📦 Benchmarks

We release two benchmarks to support evaluation:

- [**SpiderWild**](https://drive.google.com/drive/u/0/folders/1jtQQEOuRN7jz3LJ9f4QJlzrSXnGeiF41)  
- [**BirdWild**](https://drive.google.com/drive/u/0/folders/1FM2rXHoqVXXDhCCE0IadrRrzbxcnz8sL)

Each benchmark adapts an existing text-to-SQL dataset into a more realistic setting for multi-table question answering over large-scale table collections.


### Setup

- **Table collection:**  
  Download the benchmark files and place them in a `tables` subdirectory within the corresponding dataset folder.

- **Questions and Labels:**  
The associated questions, relevant tables, and answers are located in the `dataset/label` directory. 
---

## 🚀 Running the DMRAL Framework

### 1. Installation

Create and activate a conda environment, then install all required Python dependencies:

```bash
conda create -n dmral_env python=3.8 -y
conda activate dmral_env
pip install -r requirements.txt
```

### 2. Running the Full Pipeline
> **Note**: Before running the `bash.sh`, make sure to update the `dataset_name` variable by speficifying the dataset name (e.g., "spiderwild").

```bash
cd code
bash bash.sh
```

## Contact
If you have any issue, please contact feng.luo@student.rmit.edu.au.
