import os
import pandas as pd
import numpy as np
import random
from ragas import evaluate
from ragas.metrics import ContextRelevance
from datasets import Dataset
from tqdm import tqdm
from retrievers.sentence_embedding_retriever import DenseRetriever
from retrievers.tfidf_retriever import TfidfRetriever
from metric_calculators.check_openai_key import check_openai_api_key

# os.environ["OPENAI_API_KEY"] = "SECRET_API_KEY"
DATA_SAVE_DIR = './data'
DENSE_STORAGE_PATH = 'retriever_storage/dense_all_minilm_l6_v2' 
TFIDF_STORAGE_PATH = 'retriever_storage/tfidf_baseline'
RAGAS_RESULTS_FILE = "ragas_results/ragas_context_relevance_results.csv"
FORCE_RERUN_RAGAS = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

if not check_openai_api_key():
    print("a valid OPENAI_API_KEY is not set. Aborting")
    exit()

print("Loading processed data...")
try:
    reviews_df = pd.read_csv(f"{DATA_SAVE_DIR}/processed_reviews.csv")
    review_chunks_df = pd.read_csv(f"{DATA_SAVE_DIR}/processed_review_chunks.csv")
    if reviews_df.empty or review_chunks_df.empty:
        raise FileNotFoundError
except FileNotFoundError:
    print("Error: 'processed_reviews.csv' or 'processed_review_chunks.csv' not found or empty.")
    print("Please run Step 1 (Data Preparation) first and ensure the files are created.")
    exit()
print(f"Loaded {len(reviews_df)} reviews and {len(review_chunks_df)} review chunks.")

N_EVAL_PAPERS = 100
N_ICLR_EVAL = 60 # Approximate target
N_NIPS_EVAL = 40 # Approximate target

evaluation_paper_ids = []

# ICLR Papers - stratified by decision if possible
iclr_papers = reviews_df[reviews_df['conference'] == 'ICLR'][['paper_id', 'decision']].drop_duplicates().reset_index(drop=True)
if not iclr_papers.empty:
    # Try to get a mix of 'Reject' and 'Accept' (any type of accept)
    iclr_rejects = iclr_papers[iclr_papers['decision'] == 'Reject']
    iclr_accepts = iclr_papers[iclr_papers['decision'].str.contains("Accept", case=False, na=False)]

    n_iclr_reject_sample = min(len(iclr_rejects), N_ICLR_EVAL // 2)
    n_iclr_accept_sample = min(len(iclr_accepts), N_ICLR_EVAL - n_iclr_reject_sample)

    if n_iclr_reject_sample > 0:
        evaluation_paper_ids.extend(iclr_rejects.sample(n_iclr_reject_sample, random_state=42)['paper_id'].tolist())
    if n_iclr_accept_sample > 0:
        evaluation_paper_ids.extend(iclr_accepts.sample(n_iclr_accept_sample, random_state=42)['paper_id'].tolist())
    
    # If we still need more ICLR papers (e.g., one category was too small)
    remaining_iclr_needed = N_ICLR_EVAL - len(evaluation_paper_ids)
    if remaining_iclr_needed > 0:
        remaining_iclr_papers = iclr_papers[~iclr_papers['paper_id'].isin(evaluation_paper_ids)]
        if len(remaining_iclr_papers) >= remaining_iclr_needed:
            evaluation_paper_ids.extend(remaining_iclr_papers.sample(remaining_iclr_needed, random_state=42)['paper_id'].tolist())
        else:
            evaluation_paper_ids.extend(remaining_iclr_papers['paper_id'].tolist())


# NIPS Papers (reportedly all 'Accept' or similar, so random sample)
nips_papers = reviews_df[reviews_df['conference'] == 'NIPS'][['paper_id']].drop_duplicates().reset_index(drop=True)
nips_needed = N_EVAL_PAPERS - len(evaluation_paper_ids) # How many more to reach 100
if nips_needed < 0 : nips_needed = N_NIPS_EVAL # If ICLR sampling was too generous
n_nips_sample = min(len(nips_papers), nips_needed)

if not nips_papers.empty and n_nips_sample > 0:
    evaluation_paper_ids.extend(nips_papers.sample(n_nips_sample, random_state=42)['paper_id'].tolist())

# Final check if we have less than N_EVAL_PAPERS, fill with any remaining unique papers
if len(evaluation_paper_ids) < N_EVAL_PAPERS:
    print(f"Warning: Could only select {len(evaluation_paper_ids)} diverse papers. Attempting to fill remaining...")
    all_unique_paper_ids = reviews_df['paper_id'].unique()
    remaining_pool = [pid for pid in all_unique_paper_ids if pid not in evaluation_paper_ids]
    fill_count = N_EVAL_PAPERS - len(evaluation_paper_ids)
    if len(remaining_pool) >= fill_count:
        evaluation_paper_ids.extend(np.random.choice(remaining_pool, fill_count, replace=False).tolist())
    else:
        evaluation_paper_ids.extend(remaining_pool)

evaluation_paper_ids = list(set(evaluation_paper_ids))
print(f"Selected {len(evaluation_paper_ids)} unique paper_ids for evaluation.")
# print("Sample evaluation paper IDs:", evaluation_paper_ids[:10])


print("\nInitializing retrievers...")
dense_retriever = DenseRetriever(sentence_transformer_model_name='all-MiniLM-L6-v2', storage_path_prefix=DENSE_STORAGE_PATH)
tfidf_retriever = TfidfRetriever(storage_path_prefix=TFIDF_STORAGE_PATH)

# This ensures that when we query for our eval papers, their indexes/models are ready.
print("\nBuilding retriever indexes/models for all papers (this may take time)...")
tfidf_retriever.pre_build_all_paper_tfidf(review_chunks_df)
dense_retriever.pre_build_all_paper_indexes(review_chunks_df)
print("Retriever indexes/models built.")

TOP_K_RETRIEVAL = 5
evaluation_results = []

# Define static queries for RQ1 and RQ2 for this evaluation

queries_for_rqs = {
    "Q1_Justification": "What are the main arguments, strengths, and weaknesses mentioned in the reviews that could justify the paper's decision?",
    "Q2_Disagreement": "Identify specific points of notable disagreement or contrasting opinions between reviewers regarding this paper's methods, results, or conclusions."
}

print(f"\nRunning evaluation loop for {len(evaluation_paper_ids)} papers...")
for paper_id in tqdm(evaluation_paper_ids, desc="Evaluating Papers"):
    # Get the actual decision for this paper for context (RQ1)
    paper_decision = reviews_df[reviews_df['paper_id'] == paper_id]['decision'].iloc[0] if not reviews_df[reviews_df['paper_id'] == paper_id].empty else "Unknown"

    for rq_label, query_text_template in queries_for_rqs.items():
        # Customize query slightly if needed, e.g. for RQ1 including decision
        query_text = query_text_template
        if rq_label == "Q1_Justification":
             query_text = f"Given the decision was '{paper_decision}', {query_text_template.lower()}"

        # Dense Retriever
        retrieved_dense_df = dense_retriever.retrieve_from_paper(query_text, paper_id, top_k=TOP_K_RETRIEVAL)
        if not retrieved_dense_df.empty:
            evaluation_results.append({
                "paper_id": paper_id,
                "decision": paper_decision,
                "rq_label": rq_label,
                "query": query_text,
                "retriever_type": "Dense (all-MiniLM-L6-v2)",
                "retrieved_contexts": retrieved_dense_df['text'].tolist(), # List of strings
                "retrieved_chunk_ids": retrieved_dense_df['chunk_id'].tolist(),
                "similarity_scores": retrieved_dense_df['similarity_score_L2_distance'].tolist()
            })
        else: # Handle cases where no documents are retrieved
             evaluation_results.append({
                "paper_id": paper_id,
                "decision": paper_decision,
                "rq_label": rq_label,
                "query": query_text,
                "retriever_type": "Dense (all-MiniLM-L6-v2)",
                "retrieved_contexts": [],
                "retrieved_chunk_ids": [],
                "similarity_scores": []
            })


        # TF-IDF Retriever (Baseline)
        retrieved_tfidf_df = tfidf_retriever.retrieve_from_paper(query_text, paper_id, top_k=TOP_K_RETRIEVAL)
        if not retrieved_tfidf_df.empty:
            evaluation_results.append({
                "paper_id": paper_id,
                "decision": paper_decision,
                "rq_label": rq_label,
                "query": query_text,
                "retriever_type": "TF-IDF (Baseline)",
                "retrieved_contexts": retrieved_tfidf_df['text'].tolist(), # List of strings
                "retrieved_chunk_ids": retrieved_tfidf_df['chunk_id'].tolist(),
                "similarity_scores": retrieved_tfidf_df['similarity_score_cosine'].tolist()
            })
        else:
            evaluation_results.append({
                "paper_id": paper_id,
                "decision": paper_decision,
                "rq_label": rq_label,
                "query": query_text,
                "retriever_type": "TF-IDF (Baseline)",
                "retrieved_contexts": [],
                "retrieved_chunk_ids": [],
                "similarity_scores": []
            })


evaluation_results_df = pd.DataFrame(evaluation_results)
print(f"\nEvaluation loop complete. Generated {len(evaluation_results_df)} retrieval results.")
print("Sample of evaluation results (first 5):")
print(evaluation_results_df.head())

# Save the retrieval results for later use (e.g., input to LLM and RAGAS)
evaluation_results_df.to_csv(f"{DATA_SAVE_DIR}/retrieval_evaluation_inputs.csv", index=False)
print("\nRetrieval results saved")

# RAGAS expects data in a Hugging Face Dataset format with specific column names.
# We need 'question' (our query) and 'contexts' (list of retrieved strings).
# We don't have 'answer' (LLM generated) or 'ground_truth' yet.

ragas_data_list = []
for _, row in evaluation_results_df.iterrows():
    ragas_data_list.append({
        "paper_id": row["paper_id"], 
        "rq_label": row["rq_label"], 
        "retriever_type": row["retriever_type"], 
        "question": row["query"],         
        "contexts": row["retrieved_contexts"] 
        # "answer": "Placeholder - LLM answer will go here", 
        # "ground_truth": "Placeholder - Ground truth answer will go here"
    })

ragas_input_df = pd.DataFrame(ragas_data_list)

ragas_input_df_filtered = ragas_input_df[ragas_input_df['contexts'].apply(lambda x: len(x) > 0)].copy()
if len(ragas_input_df_filtered) < len(ragas_input_df):
    print(f"Filtered out {len(ragas_input_df) - len(ragas_input_df_filtered)} entries with no retrieved contexts for RAGAS.")


print(f"\nAttempting to run or load RAGAS context_relevance (requires LLM environment for RAGAS if running)...")
ragas_summary_df = None

# Check if results file already exists and we don't want to force a rerun
if os.path.exists(RAGAS_RESULTS_FILE) and not FORCE_RERUN_RAGAS:
    print(f"Loading existing RAGAS results from: {RAGAS_RESULTS_FILE}")
    try:
        ragas_summary_df = pd.read_csv(RAGAS_RESULTS_FILE)
        if 'retriever_type' not in ragas_summary_df.columns or 'context_relevance' not in ragas_summary_df.columns:
            print("Warning: Existing RAGAS results file is missing expected columns. Will re-run RAGAS.")
            ragas_summary_df = None # Force re-run
        else:
            print("Successfully loaded existing RAGAS results.")
    except Exception as e:
        print(f"Error loading existing RAGAS results file: {e}. Will attempt to re-run RAGAS.")
        ragas_summary_df = None # Force re-run
else:
    if FORCE_RERUN_RAGAS:
        print(f"FORCE_RERUN_RAGAS is True. Will re-evaluate even if '{RAGAS_RESULTS_FILE}' exists.")
    else:
        print(f"'{RAGAS_RESULTS_FILE}' not found. Will run RAGAS evaluation.")


if ragas_summary_df is None: # Only run RAGAS if results weren't loaded or need re-running
    try:
        if not ragas_input_df_filtered.empty:
            results_accumulator = [] # Initialize here

            for retriever_t in ragas_input_df_filtered['retriever_type'].unique():
                print(f"\nEvaluating context_relevance for: {retriever_t}")
                subset_df = ragas_input_df_filtered[ragas_input_df_filtered['retriever_type'] == retriever_t]
                
                if not subset_df.empty:
                    subset_hf_dataset = Dataset.from_pandas(subset_df)
                    
                    try:
                        score = evaluate(
                            dataset=subset_hf_dataset,
                            metrics=[ContextRelevance()],
                            # llm=ragas_llm, # If not set, used default OpenAI llm
                            # embeddings=ragas_embeddings, # If not set, used default OpenAI embedding
                            raise_exceptions=True 
                        )
                        print(f"RAGAS Context Relevance Score for {retriever_t}:")
                        print(score)
                        results_accumulator.append({'retriever_type': retriever_t, 'context_relevance': score['nv_context_relevance']})
                    except Exception as e:
                        print(f"Could not run RAGAS evaluation for {retriever_t} due to LLM/environment error: {e}")
                        print("Please ensure your RAGAS LLM (e.g., OpenAI API key) is correctly configured.")
                        results_accumulator.append({'retriever_type': retriever_t, 'context_relevance': float('nan')})
                else:
                    print(f"No data to evaluate for {retriever_t}")
            
            if results_accumulator:
                ragas_summary_df = pd.DataFrame(results_accumulator)
                # Save the newly computed results
                try:
                    ragas_summary_df.to_csv(RAGAS_RESULTS_FILE, index=False)
                    print(f"RAGAS evaluation results saved to: {RAGAS_RESULTS_FILE}")
                except Exception as e:
                    print(f"Error saving RAGAS results to CSV: {e}")
            else:
                print("No RAGAS results to accumulate or save.")
                ragas_summary_df = pd.DataFrame(columns=['retriever_type', 'context_relevance']) # Empty df

        else:
            print("No valid data for RAGAS context_relevance evaluation (all entries might have had empty contexts).")
            ragas_summary_df = pd.DataFrame(columns=['retriever_type', 'context_relevance']) # Empty df


    except ImportError:
        print("RAGAS or datasets library not installed. Skipping RAGAS context_relevance evaluation.")
        print("Install with: pip install ragas datasets")
        if ragas_summary_df is None: ragas_summary_df = pd.DataFrame(columns=['retriever_type', 'context_relevance']) # Empty df
    except Exception as e:
        print(f"An error occurred during RAGAS setup or outer evaluation loop: {e}")
        print("Skipping RAGAS context_relevance evaluation. Check your LLM setup for RAGAS.")
        if ragas_summary_df is None: ragas_summary_df = pd.DataFrame(columns=['retriever_type', 'context_relevance']) # Empty df


# --- Display the final RAGAS summary (either loaded or newly computed) ---
if ragas_summary_df is not None and not ragas_summary_df.empty:
    print("\n--- Final RAGAS Context Relevance Summary ---")
    print(ragas_summary_df)
elif ragas_summary_df is not None and ragas_summary_df.empty and not (os.path.exists(RAGAS_RESULTS_FILE) and not FORCE_RERUN_RAGAS):
    print("\n--- RAGAS Context Relevance Summary ---")
    print("No RAGAS results were generated (e.g., due to empty input or errors during processing).")
elif not (os.path.exists(RAGAS_RESULTS_FILE) and not FORCE_RERUN_RAGAS):
     print("\n RAGAS evaluation was skipped or failed to produce results.")

print("\n--- Evaluation Data Generation and RAGAS Context Relevance Complete/Loaded ---")