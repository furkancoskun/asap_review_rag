import pandas as pd
from retrievers.tfidf_retriever import TfidfRetriever
from rag_pipeline.rag_system import RAGSystem, PROMPT_RQ1_JUSTIFICATION, PROMPT_RQ2_DISAGREEMENT

DATA_SAVE_DIR = './data'
TFIDF_STORAGE_PATH = 'retriever_storage/tfidf_baseline'

print("Loading review chunks for retriever setup...")
try:
    review_chunks_df_main = pd.read_csv(f"{DATA_SAVE_DIR}/processed_review_chunks.csv")
    reviews_df_main = pd.read_csv(f"{DATA_SAVE_DIR}/processed_reviews.csv") # For decision
    if review_chunks_df_main.empty:
        raise FileNotFoundError
except FileNotFoundError:
    print("Error: 'processed_review_chunks.csv' not found or empty. Run Step 1.")
    exit()

print("Initializing Tf-Idf Retriever and building indexes (if not already built)...")
sparse_retriever_main = TfidfRetriever(storage_path_prefix=TFIDF_STORAGE_PATH)
sparse_retriever_main.pre_build_all_paper_tfidf(review_chunks_df_main, force_rebuild=False) # Set force_rebuild=True if needed

# Initialize RAG System with the Sparse Tf-Idf Retriever
rag_sys = RAGSystem(retriever_instance=sparse_retriever_main)

# --- Example for RQ1 ---
test_paper_id_rq1 = "ICLR_2019_1355" # Pick an ID to test
paper_info_rq1 = reviews_df_main[reviews_df_main['paper_id'] == test_paper_id_rq1]

if not paper_info_rq1.empty:
    test_paper_decision_rq1 = paper_info_rq1['decision'].iloc[0]
    original_query_rq1 = f"Given the decision was '{test_paper_decision_rq1}', what are the main arguments, strengths, and weaknesses mentioned in the reviews that could justify this decision?"

    print(f"\n--- Testing RAG for RQ1: Decision Justification ---")
    print(f"Paper ID: {test_paper_id_rq1}, Decision: {test_paper_decision_rq1}")
    print(f"Original Query: {original_query_rq1}")

    answer_rq1, contexts_rq1, prompt_sent_rq1 = rag_sys.generate_answer(
        paper_id=test_paper_id_rq1,
        original_query=original_query_rq1, # This query will be used for retrieval
        prompt_template=PROMPT_RQ1_JUSTIFICATION,
        top_k_retrieval=5
    )

    if answer_rq1:
        print("\nRetrieved Contexts for RQ1:")
        for i, ctx in enumerate(contexts_rq1):
            print(f"  Snippet {i+1}: {ctx[:150]}...") # Print snippet preview
        # print(f"\nFull Prompt Sent to LLM for RQ1:\n{prompt_sent_rq1}") # Can be very long
        print(f"\nGenerated Justification (RQ1):\n{answer_rq1}")
    else:
        print("Failed to generate an answer for RQ1.")
else:
    print(f"Could not find paper info for {test_paper_id_rq1}")


# --- Example for RQ2 ---
test_paper_id_rq2 = "NIPS_2019_367" # Pick an ID to test
original_query_rq2 = "Identify specific points of notable disagreement or contrasting opinions between reviewers regarding this paper's methods, results, or conclusions."

print(f"\n--- Testing RAG for RQ2: Reviewer Disagreement ---")
print(f"Paper ID: {test_paper_id_rq2}")
print(f"Original Query: {original_query_rq2}")

answer_rq2, contexts_rq2, prompt_sent_rq2 = rag_sys.generate_answer(
    paper_id=test_paper_id_rq2,
    original_query=original_query_rq2, # This query will be used for retrieval
    prompt_template=PROMPT_RQ2_DISAGREEMENT,
    top_k_retrieval=7 # Maybe retrieve a bit more for disagreement to see more views
)

if answer_rq2:
    print("\nRetrieved Contexts for RQ2:")
    for i, ctx in enumerate(contexts_rq2):
        print(f"  Snippet {i+1}: {ctx[:150]}...")
    # print(f"\nFull Prompt Sent to LLM for RQ2:\n{prompt_sent_rq2}")
    print(f"\nGenerated Disagreement Summary (RQ2):\n{answer_rq2}")
else:
    print("Failed to generate an answer for RQ2.")