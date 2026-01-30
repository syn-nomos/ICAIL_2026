# ICAIL_2026: Multi-label Legal Text Classification Using Enriched Label Descriptions
Official Repository for 21st International Conference on Artificial Intelligence and Law. 
"Multi-label Legal Text Classification Using Enriched Label Descriptions"

tfidf_experiments.py
This script evaluates a TF-IDF retrieval baseline for multi-label legal text classification by ranking label concept IDs using enriched label descriptions (LLM-generated text).

By default, the script expects this structure relative to the repo root:
```text
ICAIL_2026/
├─ src
	├─ tfidf_experiments.py
	├─ EURLEX57K/
	│  └─ dataset/
	│     ├─ train/                       # many *.json files (one doc per file)
	│     ├─ dev/                         # (optional)
	│     └─ test/                        # many *.json files (one doc per file)
	├─ eurlex/
	│  └─ label_description_file.jsonl
	└─ eurlex/metrics/
```

## Environment Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```
Activate it:

For Linux / macOS
```bash
source .venv/bin/activate
```
For Windows
```bash
.venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -U pip
pip install numpy scipy scikit-learn tqdm
```
### Running the Experiment
```bash
python src/tfidf_experiments.py
```

## OF-LAN

Details for experiments are available at [Preserving Zero-shot Capability in Supervised Fine-tuning for Multi-label
Text Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/zero_shot_one_side_tuning/)

## RTS
Details for experiments are available at ["Structural Contrastive Representation Learning for Zero-shot Multi-label Text Classification" in Findings of EMNLP 2022](https://github.com/tonyzhang617/structural-contrastive-representation-learning)


## Datasets

### EURLEX57K
EURLEX57K could be downloaded from [EURLEX57K](https://huggingface.co/datasets/jonathanli/eurlex)

### Mulit-Eurlex
Mulit-Eurlex could be downloaded from [Mulit-Eurlex](https://huggingface.co/datasets/nlpaueb/multi_eurlex)