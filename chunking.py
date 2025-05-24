import os
import json
import re
import nltk
import pandas as pd
from tqdm.auto import tqdm 

DATASET_ROOT = './dataset'
DATA_SAVE_DIR = './data'

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("'punkt' downloaded.")


def safe_load_json(filepath):
    """Safely loads a JSON file, returning None on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # print(f"Warning: File not found: {filepath}") # Can be noisy
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from: {filepath}")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred reading {filepath}: {e}")
        return None

def parse_score_from_string(score_string):
    """Parses numeric score from strings like '8: Top 50%' or '4: Confident'."""
    if not isinstance(score_string, str):
        return None
    match = re.match(r"^\s*(\d+)", score_string)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n') # Normalize newlines
    text = re.sub(r'\n\s*\n', '\n\n', text) # Normalize multiple newlines to double
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def chunk_text_into_sentences(text, review_id, paper_id):
    """Chunks text into sentences, adding metadata."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i, sentence in enumerate(sentences):
        cleaned_sentence = clean_text(sentence) # Clean individual sentences too
        if cleaned_sentence: # Avoid empty sentences
            chunks.append({
                'chunk_id': f"{review_id}_chunk_{i}",
                'review_id': review_id,
                'paper_id': paper_id,
                'text': cleaned_sentence,
                'chunk_index_in_review': i
            })
    return chunks


def preprocess_asap_review_data(dataset_root):
    """
    Loads, preprocesses, and chunks review data from the ASAP-Review dataset.
    """
    all_reviews_data = []
    all_review_chunks = []

    conference_prefixes = ['ICLR', 'NIPS']
    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root directory not found at '{dataset_root}'")
        return [], []

    # Pre-load paper decisions to avoid repeated file reads for reviews of the same paper
    paper_decisions_cache = {}

    # First pass: Collect all paper IDs and their decisions
    print("Pass 1: Caching paper decisions...")
    for prefix in conference_prefixes:
        for item in tqdm(os.listdir(dataset_root), desc=f"Scanning {prefix} folders"):
            if item.startswith(prefix) and os.path.isdir(os.path.join(dataset_root, item)):
                conf_year_path = os.path.join(dataset_root, item)
                # Paper files are usually directly within conf_year_path or one level deeper if organized by ID
                for root_dir, _, files in os.walk(conf_year_path):
                    for filename in files:
                        if filename.endswith('_paper.json'):
                            paper_id_from_filename = filename.replace('_paper.json', '')
                            if paper_id_from_filename not in paper_decisions_cache:
                                paper_filepath = os.path.join(root_dir, filename)
                                paper_meta = safe_load_json(paper_filepath)
                                if paper_meta and 'id' in paper_meta:
                                    paper_decisions_cache[paper_meta['id']] = paper_meta.get('decision', 'Unknown')
                                elif paper_meta: # If 'id' key is missing but file loads
                                     paper_decisions_cache[paper_id_from_filename] = paper_meta.get('decision', 'Unknown')
                                     print(f"Warning: Paper file {paper_filepath} loaded but 'id' key missing. Using filename as ID.")
                                else:
                                     paper_decisions_cache[paper_id_from_filename] = 'Unknown (File Load Error)'


    print(f"Cached decisions for {len(paper_decisions_cache)} papers.")
    print("\nPass 2: Processing reviews...")
    for prefix in conference_prefixes:
        conference_folders = [d for d in os.listdir(dataset_root) if d.startswith(prefix) and os.path.isdir(os.path.join(dataset_root, d))]
        for conf_year_folder in tqdm(conference_folders, desc=f"Processing {prefix} Conference Years"):
            conf_year_path = os.path.join(dataset_root, conf_year_folder)

            # Reviews are typically in _review.json files, potentially in subdirectories
            for root_dir, _, files in os.walk(conf_year_path):
                for filename in files:
                    if filename.endswith('_review.json'):
                        review_filepath = os.path.join(root_dir, filename)
                        review_collection_data = safe_load_json(review_filepath)

                        if not review_collection_data or 'id' not in review_collection_data or 'reviews' not in review_collection_data:
                            print(f"Warning: Skipping malformed review file: {review_filepath}")
                            continue

                        paper_id = review_collection_data['id']
                        decision = paper_decisions_cache.get(paper_id, 'Unknown (Paper Meta Not Found)')

                        for review_idx, review_content in enumerate(review_collection_data['reviews']):
                            review_id = f"{paper_id}_review_{review_idx}"
                            raw_review_text = review_content.get('review', '')
                            cleaned_review_text = clean_text(raw_review_text)

                            rating_str = review_content.get('rating')
                            confidence_str = review_content.get('confidence')

                            numerical_rating = parse_score_from_string(rating_str) if rating_str else None
                            numerical_confidence = parse_score_from_string(confidence_str) if confidence_str else None

                            review_meta = {
                                'review_id': review_id,
                                'paper_id': paper_id,
                                'conference': prefix,
                                'year': conf_year_folder.split('_')[-1] if '_' in conf_year_folder else 'UnknownYear',
                                'decision': decision,
                                'raw_review_text': raw_review_text,
                                'cleaned_review_text': cleaned_review_text,
                                'rating_str': rating_str,
                                'rating_score': numerical_rating,
                                'confidence_str': confidence_str,
                                'confidence_score': numerical_confidence,
                                'review_index_in_paper': review_idx
                            }
                            all_reviews_data.append(review_meta)

                            # Chunk the cleaned review text
                            if cleaned_review_text:
                                chunks = chunk_text_into_sentences(cleaned_review_text, review_id, paper_id)
                                all_review_chunks.extend(chunks)
    return all_reviews_data, all_review_chunks


if __name__ == "__main__":
    print("Starting Step 1: Data Preparation & Preprocessing...")

    if not os.path.exists(DATASET_ROOT) or not os.path.isdir(DATASET_ROOT):
        print(f"Error: The DATASET_ROOT '{DATASET_ROOT}' does not exist or is not a directory.")
        print("Please set the DATASET_ROOT variable at the top of the script to the correct path of your ASAP-Review dataset.")
    else:
        # Check for presence of ICLR/NIPS subfolders to give more specific guidance
        has_iclr = any(d.startswith("ICLR") for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)))
        has_nips = any(d.startswith("NIPS") for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)))

        if not (has_iclr or has_nips):
            print(f"Warning: Could not find typical ICLR or NIPS subdirectories in '{DATASET_ROOT}'.")
            print("Please ensure DATASET_ROOT points to the directory containing folders like 'ICLR_2017', 'NIPS_2016', etc.")
        else:
            print(f"Processing dataset from: {os.path.abspath(DATASET_ROOT)}")


            reviews_df_list, review_chunks_df_list = preprocess_asap_review_data(DATASET_ROOT)

            if reviews_df_list and review_chunks_df_list:
                reviews_df = pd.DataFrame(reviews_df_list)
                review_chunks_df = pd.DataFrame(review_chunks_df_list)

                print(f"\n--- Preprocessing Complete ---")
                print(f"Total reviews processed: {len(reviews_df)}")
                print(f"Total review sentence chunks created: {len(review_chunks_df)}")

                print("\nSample of Processed Reviews (first 5):")
                print(reviews_df.head())

                print("\nSample of Processed Review Chunks (first 5):")
                print(review_chunks_df.head())

                print("\nStatistics for ICLR reviews:")
                iclr_reviews = reviews_df[reviews_df['conference'] == 'ICLR']
                if not iclr_reviews.empty:
                    print(f"  Total ICLR Reviews: {len(iclr_reviews)}")
                    print(f"  ICLR Reviews with numeric rating: {iclr_reviews['rating_score'].notna().sum()}")
                    print(f"  ICLR Reviews with numeric confidence: {iclr_reviews['confidence_score'].notna().sum()}")
                    print(f"  ICLR Decision Distribution:\n{iclr_reviews['decision'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'}")
                else:
                    print("  No ICLR reviews found.")


                print("\nStatistics for NIPS reviews:")
                nips_reviews = reviews_df[reviews_df['conference'] == 'NIPS']
                if not nips_reviews.empty:
                    print(f"  Total NIPS Reviews: {len(nips_reviews)}")
                    print(f"  NIPS Reviews with numeric rating: {nips_reviews['rating_score'].notna().sum()} (Expected to be low/zero)")
                    print(f"  NIPS Reviews with numeric confidence: {nips_reviews['confidence_score'].notna().sum()}")
                    print(f"  NIPS Decision Distribution:\n{nips_reviews['decision'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'}")
                else:
                    print("  No NIPS reviews found.")

                reviews_df.to_csv(f"{DATA_SAVE_DIR}/processed_reviews.csv", index=False)
                review_chunks_df.to_csv(f"{DATA_SAVE_DIR}/processed_review_chunks.csv", index=False)
                print("\nProcessed data saved to CSV files (commented out by default).")
            else:
                print("No data was processed. Please check your DATASET_ROOT and file structure.")