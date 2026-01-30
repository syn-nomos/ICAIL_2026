# ICAIL_2026: Multi-label Legal Text Classification Using Enriched Label Descriptions
Official Repository for 21st International Conference on Artificial Intelligence and Law. 
"Multi-label Legal Text Classification Using Enriched Label Descriptions"

##Abstract
*Multi-label text classification is important in the legal domain where
*documents naturally involve multiple topics and capturing all of
*them is essential for accurate search, review, compliance, and legal
*reasoning. In addition, the set of labels is constantly enriched and it
*is not feasible to obtain adequate training data for new categories.
*A common approach in this area is to attempt to represent both
*documents and legal label descriptions in a common embedding
*space to reveal the most relevant labels per document. To this end,
*short or very short label descriptions (e.g., a single word or a few
*words) are used while the label hierarchy, if available, can also be
*useful. In this paper, we propose the use of LLMs to provide enriched
*label descriptions, allowing existing multi-label text classification
*methods to better represent labels and estimate their relevance to
*documents. In the presented experiments, we used two datasets
*from the legal domain to demonstrate how the performance of
*existing methods is improved when the proposed enriched label
*descriptions are used. In addition, we examine how the performance
*is affected by using specific LLMs as well as the effect of exploiting
*label hierarchy when generating the label descriptions.

The tfidf_experiments.py script evaluates a TF-IDF retrieval baseline for multi-label legal text classification by ranking label concept IDs using enriched label descriptions (LLM-generated text).
All LLM Descriptions used in our experiments are available under LLM_Descriptions folder.
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
### Running the preparation script
```bash
python src/make_jsonl_from_zip.py /path/to/<input-LLM-descriptions>.zip -o eurlex/label_description_file.jsonl
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