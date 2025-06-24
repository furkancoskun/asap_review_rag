# asap_review_rag

## Data Analysis
```
python -m data_analyzers.data_analysis_v2
```



## Run
### 1 - Ollama
#### start ollama qwen3:8b model
```
source ollama_tools/run_ollama.sh
```

#### check running of ollama qwen3:8b model
```
source ollama_tools/request_ollama.sh
# or
python ollama_tools/check_ollama.py
```

### 2 - Dense Rag
You can select different paper ids by changing test_paper_ids in the code:

```
test_paper_id_rq1 = "ICLR_2019_1355" # Pick an ID to test
test_paper_id_rq2 = "NIPS_2019_367" # Pick an ID to test
```

run with following command:
```
python -m test_dense_rag
```

### 3 - Sparse Rag
```
python -m test_sparse_rag
```

## Evaluation
This repository uses ragas tool for evalation. Before running ragas metric calculators you need to set OPENAI_API_KEY.

#### run Ragas Context Relevance experiment
```
python -m metric_calculators.ragas_context_relevance
```

#### run Ragas full RAG system experiment - Dense Embedding
```
python -m metric_calculators.ragas_full_rag_system_dense
```

#### run Ragas full RAG system experiment - Sparse(Tf-Idf) Embedding
```
python -m metric_calculators.ragas_full_rag_system_sparse
```