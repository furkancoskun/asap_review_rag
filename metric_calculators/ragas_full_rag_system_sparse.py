import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import ContextRelevance, Faithfulness, AnswerRelevancy
from retrievers.tfidf_retriever import TfidfRetriever
from rag_pipeline.rag_system import RAGSystem, PROMPT_RQ1_JUSTIFICATION, PROMPT_RQ2_DISAGREEMENT
from metric_calculators.check_openai_key import check_openai_api_key

# os.environ["OPENAI_API_KEY"] = "SECRET_API_KEY"
DATA_SAVE_DIR = './data'
TFIDF_STORAGE_PATH = 'retriever_storage/tfidf_baseline'
FULL_RAGAS_RESULTS_FILE = "ragas_results/ragas_full_system_sparse_results.csv"
RETRIEVAL_INPUTS_FILE = f"{DATA_SAVE_DIR}/retrieval_evaluation_inputs.csv" # From previous eval step
FORCE_RERUN_FULL_RAGAS = False 

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

TOP_K_RETRIEVAL_FOR_EVAL = 5

if not check_openai_api_key():
    print("A valid OPENAI_API_KEY is not set. Aborting.")
    exit()

print("Loading processed data and evaluation paper IDs...")
try:
    reviews_df = pd.read_csv(f"{DATA_SAVE_DIR}/processed_reviews.csv")
    review_chunks_df = pd.read_csv(f"{DATA_SAVE_DIR}/processed_review_chunks.csv")
    eval_inputs_df = pd.read_csv(RETRIEVAL_INPUTS_FILE)
    if reviews_df.empty or review_chunks_df.empty or eval_inputs_df.empty:
        raise FileNotFoundError
except FileNotFoundError:
    print("Error: Necessary CSV files not found. Ensure previous steps are complete.")
    exit()

eval_queries_df = eval_inputs_df[eval_inputs_df['retriever_type'] == 'TF-IDF (Baseline)'].copy()
eval_queries_df = eval_queries_df[['paper_id', 'decision', 'rq_label', 'query']].drop_duplicates().reset_index(drop=True)
print(f"Loaded {len(eval_queries_df)} unique (paper_id, query) pairs for full RAG evaluation.")

print("\nInitializing Sparse Tf-Idf Retriever and RAG System...")
sparse_retriever_eval = TfidfRetriever(storage_path_prefix=TFIDF_STORAGE_PATH)
sparse_retriever_eval.pre_build_all_paper_tfidf(review_chunks_df, force_rebuild=False)

rag_system_eval = RAGSystem(retriever_instance=sparse_retriever_eval)

GENERATED_ANSWERS_FILE = f"{DATA_SAVE_DIR}/generated_answers_sparse_rag.csv"
all_generated_data = []

if os.path.exists(GENERATED_ANSWERS_FILE) and not FORCE_RERUN_FULL_RAGAS:
    print(f"Loading existing generated answers from {GENERATED_ANSWERS_FILE}...")
    all_generated_data_df = pd.read_csv(GENERATED_ANSWERS_FILE)
    # Convert 'contexts' back to list if stored as string
    if 'contexts' in all_generated_data_df.columns and isinstance(all_generated_data_df['contexts'].iloc[0], str):
        try:
            all_generated_data_df['contexts'] = all_generated_data_df['contexts'].apply(eval)
        except Exception:
             print("Warning: Could not evaluate 'contexts' column back to list. Re-generating might be needed.")
             all_generated_data_df = pd.DataFrame() # Force re-generation
    
    if not all_generated_data_df.empty:
        current_eval_ids = set(eval_queries_df.apply(lambda r: (r['paper_id'], r['rq_label']), axis=1))
        loaded_eval_ids = set(all_generated_data_df.apply(lambda r: (r['paper_id'], r['rq_label']), axis=1))
        if current_eval_ids == loaded_eval_ids:
             print("Loaded answers match current evaluation set.")
        else:
            print("Warning: Loaded answers do not fully match the current evaluation set. Re-generating answers.")
            all_generated_data_df = pd.DataFrame() # Force re-generation
else:
    if FORCE_RERUN_FULL_RAGAS: print("FORCE_RERUN_FULL_RAGAS is True. Re-generating answers.")
    all_generated_data_df = pd.DataFrame() 

if all_generated_data_df.empty:
    print(f"\nGenerating answers for {len(eval_queries_df)} evaluation items using RAG system...")
    for _, row in tqdm(eval_queries_df.iterrows(), total=len(eval_queries_df), desc="Generating RAG Answers"):
        paper_id = row['paper_id']
        original_query = row['query']
        rq_label = row['rq_label']

        prompt_template_to_use = PROMPT_RQ1_JUSTIFICATION if rq_label == "Q1_Justification" else PROMPT_RQ2_DISAGREEMENT

        answer, contexts, _ = rag_system_eval.generate_answer(
            paper_id=paper_id,
            original_query=original_query,
            prompt_template=prompt_template_to_use,
            top_k_retrieval=TOP_K_RETRIEVAL_FOR_EVAL
        )
        all_generated_data.append({
            "paper_id": paper_id,
            "rq_label": rq_label,
            "question": original_query, # RAGAS expects 'question'
            "answer": answer if answer is not None else "GENERATION_FAILED", # RAGAS expects 'answer'
            "contexts": contexts if contexts is not None else [], # RAGAS expects 'contexts'
            # "ground_truth": "Placeholder for RQ-specific ground truth" # If you create it
        })
    all_generated_data_df = pd.DataFrame(all_generated_data)
    all_generated_data_df.to_csv(GENERATED_ANSWERS_FILE, index=False)
    print(f"Generated answers saved to {GENERATED_ANSWERS_FILE}")

# Filter out rows where generation might have failed for RAGAS evaluation
ragas_eval_dataset_df = all_generated_data_df[all_generated_data_df['answer'] != "GENERATION_FAILED"].copy()
ragas_eval_dataset_df = ragas_eval_dataset_df[ragas_eval_dataset_df['contexts'].apply(lambda x: len(x) > 0)].copy()

if ragas_eval_dataset_df.empty:
    print("No valid data (generated answers and contexts) to evaluate with RAGAS. Exiting.")
    exit()

print(f"Prepared {len(ragas_eval_dataset_df)} items for full RAGAS evaluation.")

print("\nRunning full RAGAS evaluation (context_relevance, faithfulness, answer_relevancy)...")
full_ragas_summary_df = None

if os.path.exists(FULL_RAGAS_RESULTS_FILE) and not FORCE_RERUN_FULL_RAGAS:
    print(f"Loading existing full RAGAS results from {FULL_RAGAS_RESULTS_FILE}")
    try:
        full_ragas_summary_df = pd.read_csv(FULL_RAGAS_RESULTS_FILE)
        if 'context_relevance' not in full_ragas_summary_df.columns or \
           'faithfulness' not in full_ragas_summary_df.columns or \
           'answer_relevancy' not in full_ragas_summary_df.columns:
            print("Warning: Existing full RAGAS results file is missing expected metric columns. Will re-run.")
            full_ragas_summary_df = None
        else:
            print("Successfully loaded existing full RAGAS results.")
    except Exception as e:
        print(f"Error loading full RAGAS results: {e}. Will re-run.")
        full_ragas_summary_df = None
else:
    if FORCE_RERUN_FULL_RAGAS: print("FORCE_RERUN_FULL_RAGAS is True for full RAGAS evaluation.")


if full_ragas_summary_df is None:
    try:
        ragas_hf_dataset_full = Dataset.from_pandas(ragas_eval_dataset_df)

        metrics_to_evaluate = [
            ContextRelevance(),
            Faithfulness(),
            AnswerRelevancy()
        ]
        print(f"Evaluating with metrics: {[m.name for m in metrics_to_evaluate]}")

        full_score_results = evaluate(
            dataset=ragas_hf_dataset_full,
            metrics=metrics_to_evaluate,
            raise_exceptions=True
        )
        print("\nFull RAGAS Evaluation Raw Results:")
        print(full_score_results)

        full_ragas_summary_df = pd.DataFrame([full_score_results])
        full_ragas_summary_df['retriever_type'] = "TF-IDF (Baseline)"
        
        cols = ['retriever_type'] + [m.name for m in metrics_to_evaluate if m.name in full_ragas_summary_df.columns]
        full_ragas_summary_df = full_ragas_summary_df[cols]

        full_ragas_summary_df.to_csv(FULL_RAGAS_RESULTS_FILE, index=False)
        print(f"Full RAGAS results saved to {FULL_RAGAS_RESULTS_FILE}")

    except ImportError:
        print("RAGAS or datasets library not installed.")
    except Exception as e:
        print(f"Full RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        if full_ragas_summary_df is None: full_ragas_summary_df = pd.DataFrame()


if full_ragas_summary_df is not None and not full_ragas_summary_df.empty:
    print("\n--- Final Full RAG System RAGAS Summary (Sparse Retriever) ---")
    print(full_ragas_summary_df.to_string())
else:
    print("\nFull RAGAS evaluation was skipped or failed to produce results.")

print("\n--- Full RAG System Evaluation Script Complete ---")