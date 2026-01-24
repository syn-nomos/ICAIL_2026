# ICAIL_2026: Multi-label Legal Text Classification Using Enriched Label Descriptions
Official Repository for 21st International Conference on Artificial Intelligence and Law. 
"Multi-label Legal Text Classification Using Enriched Label Descriptions"

tfidf_experiments.py
This script evaluates a TF-IDF retrieval baseline for multi-label legal text classification by ranking label concept IDs using enriched label descriptions (LLM-generated text).

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