from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class TfidfRetriever:
    def __init__(self):
        self.paper_specific_tfidf_models = {} # Stores {'paper_id': {'vectorizer': tfidf_vectorizer, 'matrix': tfidf_matrix, 'chunks_df': paper_chunks_df}}

    def build_tfidf_for_paper(self, paper_id, paper_chunks_df):
        """
        Builds a TF-IDF model for the review chunks of a specific paper.
        Args:
            paper_id (str): The ID of the paper.
            paper_chunks_df (pd.DataFrame): DataFrame containing review chunks ONLY for this paper.
                                            Must have 'text' and 'chunk_id' columns.
        """
        if paper_chunks_df.empty:
            print(f"Warning: No chunks provided for paper {paper_id} (TF-IDF). Model not built.")
            return

        print(f"Building TF-IDF model for paper {paper_id} with {len(paper_chunks_df)} chunks...")
        chunk_texts = paper_chunks_df['text'].tolist()

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) # Use unigrams and bigrams
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)

        self.paper_specific_tfidf_models[paper_id] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'chunks_df': paper_chunks_df.copy()
        }
        print(f"TF-IDF model built for paper {paper_id}.")

    def retrieve_from_paper(self, query_text, paper_id, top_k=5):
        """
        Retrieves top_k relevant chunks for a given query from a specific paper's reviews using TF-IDF.
        Args:
            query_text (str): The query string.
            paper_id (str): The ID of the paper to retrieve reviews from.
            top_k (int): Number of chunks to retrieve.
        Returns:
            pd.DataFrame: DataFrame of the top_k retrieved chunks for that paper, or empty if no model/chunks.
        """
        if paper_id not in self.paper_specific_tfidf_models:
            print(f"Error: No TF-IDF model found for paper {paper_id}. Please build the model first.")
            return pd.DataFrame()

        paper_data = self.paper_specific_tfidf_models[paper_id]
        vectorizer = paper_data['vectorizer']
        tfidf_matrix = paper_data['matrix']
        paper_chunks_df = paper_data['chunks_df']

        if paper_chunks_df.empty:
            return pd.DataFrame()

        query_vector = vectorizer.transform([query_text])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Ensure top_k is not greater than the number of items
        num_indexed_items = len(paper_chunks_df)
        actual_top_k = min(top_k, num_indexed_items)

        if actual_top_k == 0:
            return pd.DataFrame()

        # Get top_k indices. Argsort sorts in ascending, so we take the last 'actual_top_k' elements after sorting.
        # Then reverse them to get descending order of similarity.
        top_indices = cosine_similarities.argsort()[-actual_top_k:][::-1]

        retrieved_chunks = paper_chunks_df.iloc[top_indices].copy()
        retrieved_chunks['similarity_score_cosine'] = cosine_similarities[top_indices] # Higher cosine similarity is better
        return retrieved_chunks.sort_values(by='similarity_score_cosine', ascending=False)

    def pre_build_all_paper_tfidf(self, all_review_chunks_df):
        """
        Builds TF-IDF models for all papers present in the review_chunks_df.
        Args:
            all_review_chunks_df (pd.DataFrame): DataFrame containing all review chunks.
                                              Must have 'paper_id', 'text', and 'chunk_id'.
        """
        if all_review_chunks_df is None or all_review_chunks_df.empty:
            print("Error: No review chunks data provided to pre_build_all_paper_tfidf.")
            return

        print("Pre-building TF-IDF models for all papers...")
        grouped_by_paper = all_review_chunks_df.groupby('paper_id')
        for paper_id, paper_chunks_df in tqdm(grouped_by_paper, desc="Building TF-IDF paper models"):
            self.build_tfidf_for_paper(paper_id, paper_chunks_df)
        print("All paper-specific TF-IDF models built.")
