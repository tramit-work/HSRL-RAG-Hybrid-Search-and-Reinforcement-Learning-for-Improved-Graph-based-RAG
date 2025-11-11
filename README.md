# HSRL-RAG: Combining Hybrid Search and Reinforcement Learning for Graph-based Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-HSRL--RAG-blue.svg)](https://github.com/tramit-work/HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG)

---

## Description

**HSRL-RAG** introduces a hybrid retrieval-augmented generation (RAG) system integrating:

- **Knowledge Graph (KG) retrieval**  
- **Hybrid dense-sparse search** (BM25 + SBERT embeddings)  
- **Reinforcement Learning (RL)** for adaptive query and context refinement  

to enhance reasoning and factual consistency in multi-hop question answering.

**Authors:**  
Truong Ho-Viet Phan, Tram Ngoc-Bao Nguyen  
*Van Lang University, Ho Chi Minh City, Vietnam*

**Submitted to:** *PeerJ Computer Science (CS-2025:11:128338)*  

**Repository:**  
[https://github.com/tramit-work/HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG](https://github.com/tramit-work/HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG)

---

## Abstract

RAG systems enhance large language models (LLMs) by grounding outputs in retrieved evidence. However, current methods face limitations:

1. **Static retrieval** without adaptive refinement  
2. **Inefficient multi-hop reasoning** across semantically distant documents  

**HSRL-RAG** addresses these via:

- Construction of a **semantic knowledge graph (KG)** linking entities across documents  
- A **hybrid retriever** combining BM25 (sparse) and SBERT (dense) embeddings  
- A **reinforcement learning module** dynamically refining queries, context expansion, and generation

---

## Repository Structure

```bash
HSRL-RAG/
│
├── code/
│   ├── know_graph.ipynb      # KG construction & visualization (Neo4j + Llama-3)
│   ├── rl_main.ipynb         # RL-driven RAG training & evaluation loop
│
├── data/
│   ├── data_medicine.docx    # Raw textual knowledge base
│   ├── kg_medicine.json      # Extracted KG triples
│
├── requirements.txt          # Dependencies
├── README.md                 # This file
```

## Installation
### 1. Clone and Setup Environment
```bash
git clone https://github.com/tramit-work/HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG
cd HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG

conda create -n hsrl-rag python=3.10
conda activate hsrl-rag
pip install -r requirements.txt
```
### 2. Requirements
```bash
torch>=2.0.0
transformers>=4.38.0
accelerate>=0.27.0
huggingface_hub>=0.21.0
llama-index==0.10.33
llama-index-core==0.10.33
llama-index-legacy==0.9.48
llama-index-llms-huggingface==0.1.4
llama-index-embeddings-langchain==0.1.2
llama-index-embeddings-openai==0.1.9
llama-index-embeddings-huggingface>=0.1.5
langchain==0.1.16
langchain-core==0.1.46
langchain-community==0.0.34
neo4j>=5.15.0
pyvis>=0.3.2
sentence-transformers>=2.2.2
nltk>=3.9
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
pypdf>=4.2.0
docx2txt>=0.8
ipython>=8.22.0
tqdm>=4.66.0
evaluate>=0.4.2
rouge-score>=0.1.2
sacrebleu>=2.4.0
```
### 3. Methodology Overview
#### 1. Knowledge Graph Construction

Input documents (e.g., data_medicine.docx) are segmented into 512-token chunks.

Each chunk is processed using Llama to extract structured triples (head, relation, tail).

Triples are stored in Neo4j with SBERT embeddings for entity-level similarity.

#### 2. Hybrid Retrieval Mechanism

A weighted hybrid score balances sparse (BM25) and dense (SBERT) retrieval:

Score(q, d) = (1 - α) * BM25(q, d) + α * cos(v_q, v_d)

where:

- α = 0.5  
- Top-k entities (k = 3) are expanded via graph traversal depth l = 1

Explanation:

- BM25(q, d) — sparse lexical similarity
- cos(v_q, v_d) — dense semantic similarity via SBERT embeddings
- α — balances sparse and dense contributions

#### 3. Reinforcement Learning Pipeline

Implemented in rl_main.ipynb:

State: (query, retrieved_context, rewritten_query, generation_history)

Actions: rewrite_query, expand_context, generate_response

Reward: cosine similarity between generated and gold answers using SBERT

Training: 100 episodes with ε-greedy policy exploration.

#### 4. Evaluation Metrics

| Task       | Metric         | Description                                 |
|------------|----------------|---------------------------------------------|
| Retrieval  | F1             | Overlap between retrieved and gold evidence|
| Generation | BLEU, ROUGE-L  | Textual similarity to ground truth         |

---

## Running the System
### Step 1. Build the Knowledge Graph
```bash
python code/know_graph.py \
  --input data/data_medicine.docx \
  --output data/kg_medicine.json \
  --model meta-llama/Llama-3.2-3B-Instruct
```
### Step 2. Launch Neo4j Database 
```bash
docker run --name neo4j-hsrl \
  -p 7474:7474 -p 7687:7687 \
  -v $PWD/data/neo4j:/data \
  -e NEO4J_AUTH=neo4j/hsrl_password \
  neo4j:latest
```
### Step 3. Execute Reinforcement Learning RAG
```bash
python code/rl_main.ipynb
```
## Experimental Datasets
We evaluate on two multi-hop QA benchmarks:
| Dataset | #Examples | Train | Dev | Test | Type |
|--------|-----------|-------|-----|------|------|
| **2WikiMultihopQA** | 192,606 | 167,454 | 12,576 | 12,576 | Multi-hop QA over Wikipedia + Wikidata |
| **MuSiQue** | 24,814 | 19,938 | 2,417 | 2,459 | Multi-hop composed from SQuAD, NQ, etc. |

## Results
### Retrieval (F1)

| Model        | 2WikiMultihopQA | MuSiQue |
|-------------|----------------|---------|
| EfficientRAG | 51.64          | 21.18   |
| HSRL-RAG     | **83.15**      | **81.38** |

### Generation (2WikiMultihopQA)

| Model       | BLEU   | ROUGE-L |
|------------|--------|---------|
| TASE-CoT   | 45.38  | 37.42   |
| HSRL-RAG   | **78.10** | **81.00** |

## Reproducibility Notes
* Python version: 3.10

* Torch seed: 42, Numpy seed: 42

* GPU recommended; CPU possible for small datasets

* Neo4j version: 5.15.0

* FAISS version: 1.7.4

* Runtime: ~2h for KG construction on a 16GB GPU
  
## License & Contribution
This project is licensed under the MIT License - see LICENSE

**Contributions:**

1. Fork the repository

2. Create a branch: git checkout -b feature/xxx

3. Commit: git commit -m "Add xxx"

4. Push & open a Pull Request

Issues: GitHub Issues

## Citation

If you use this repository, please cite:
```bash
@article{phan2025hsrl,
  title={HSRL-RAG: Combining Hybrid Search and Reinforcement Learning to Enhance the Accuracy of Graph-based Retrieval-Augmented Generation System},
  author={Phan, Truong Ho-Viet and Nguyen, Tram Ngoc-Bao},
  journal={PeerJ Computer Science},
  year={2025},
  doi={10.7717/peerj-cs.XXXX},
  url={https://github.com/tramit-work/HSRL-RAG-Hybrid-Search-and-Reinforcement-Learning-for-Improved-Graph-based-RAG}
}
```
## Contact Information

Truong Ho-Viet Phan – truong.phv@vlu.edu.vn

Tram Ngoc-Bao Nguyen – tram.2274802010908@vanlanguni.vn

Institution: Van Lang University, Ho Chi Minh City, Vietnam

Last Updated: November 11, 2025











