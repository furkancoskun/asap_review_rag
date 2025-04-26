import os
import json
from collections import Counter
import re
import pandas as pd
from tqdm import tqdm
import statistics
import random

dataset_root = './dataset' 

def safe_load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON: {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return None

def parse_score(score_string):
    if not isinstance(score_string, str):
        return None
    match = re.match(r"^\s*(\d+)", score_string)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

paper_metadata = []
review_data = []
content_data = []
aspect_reviews_data = []
all_review_ids = set()
aspect_review_ids = set()
aspect_review_texts = {}
conference_years = []
file_counts = Counter()

for item in os.listdir(dataset_root):
    item_path = os.path.join(dataset_root, item)
    if os.path.isdir(item_path) and (item.startswith('ICLR_') or item.startswith('NIPS_')):
        conference_years.append(item)
        for root, _, files in os.walk(item_path):
            for filename in files:
                if filename.endswith('_paper.json'):
                    file_counts['paper'] += 1
                elif filename.endswith('_review.json'):
                    file_counts['review'] += 1
                elif filename.endswith('_content.json'):
                    file_counts['content'] += 1
                elif filename.endswith('.json') or filename.endswith('.pdf'):
                     file_counts['other_json_pdf'] += 1 

print(f"Conference-Years: {', '.join(sorted(conference_years))}")
print(f"Total Paper JSONs found: {file_counts['paper']}")
print(f"Total Review JSONs found: {file_counts['review']}")
print(f"Total Content JSONs found: {file_counts['content']}")
print("")
print("-" * 30)
print("")


total_papers = 0
papers_missing_content = 0
papers_missing_reviews = 0
decision_counts = Counter()

for conf_year in tqdm(conference_years, desc="Processing Papers"):
    conf_year_path = os.path.join(dataset_root, conf_year)
    for root, _, files in os.walk(conf_year_path):
        for filename in files:
            if filename.endswith('_paper.json'):
                total_papers += 1
                filepath = os.path.join(root, filename)
                data = safe_load_json(filepath)
                if data:
                    paper_metadata.append(data)
                    decision = data.get('decision', 'Unknown')
                    decision_counts[decision] += 1
                    if not data.get('hasContent', True): # Default to True if key missing
                        papers_missing_content += 1
                    if not data.get('hasReview', True):
                        papers_missing_reviews += 1

print(f"Total papers analyzed: {total_papers}")
print("Decision Distribution:")
for decision, count in decision_counts.most_common():
    print(f"  - {decision}: {count} ({count/total_papers:.1%})")
print(f"Papers flagged as missing content: {papers_missing_content}")
print(f"Papers flagged as missing reviews: {papers_missing_reviews}")
print("")
print("-" * 30)
print("")


total_reviews = 0
review_ratings = []
review_confidences = []
review_lengths_chars = []
review_lengths_words = []
reviews_missing_rating = 0
reviews_missing_confidence = 0
reviews_missing_text = 0

for conf_year in tqdm(conference_years, desc="Processing Reviews"):
    conf_year_path = os.path.join(dataset_root, conf_year)
    for root, _, files in os.walk(conf_year_path):
        for filename in files:
            if filename.endswith('_review.json'):
                filepath = os.path.join(root, filename)
                paper_review_data = safe_load_json(filepath)
                if paper_review_data:
                    paper_id = paper_review_data.get('id')
                    reviews_list = paper_review_data.get('reviews', [])
                    for idx, review in enumerate(reviews_list):
                        total_reviews += 1
                        review_id = f"{paper_id}_review_{idx}" # unique ID construct
                        all_review_ids.add(review_id)

                        rating_str = review.get('rating')
                        confidence_str = review.get('confidence')
                        review_text = review.get('review')

                        rating_score = parse_score(rating_str)
                        confidence_score = parse_score(confidence_str)

                        if rating_score is not None:
                            review_ratings.append(rating_score)
                        else:
                            reviews_missing_rating += 1

                        if confidence_score is not None:
                            review_confidences.append(confidence_score)
                        else:
                            reviews_missing_confidence += 1

                        if review_text and isinstance(review_text, str):
                            review_lengths_chars.append(len(review_text))
                            review_lengths_words.append(len(review_text.split()))
                        else:
                            reviews_missing_text += 1

                        review_data.append({
                            'review_id': review_id,
                            'paper_id': paper_id,
                            'rating_str': rating_str,
                            'confidence_str': confidence_str,
                            'rating_score': rating_score,
                            'confidence_score': confidence_score,
                            'text_length_chars': len(review_text) if review_text else 0,
                            'text_length_words': len(review_text.split()) if review_text else 0,
                            'review_text_present': bool(review_text)
                        })


print(f"Total reviews analyzed: {total_reviews}")
print(f"Reviews missing rating field or failed parse: {reviews_missing_rating}")
print(f"Reviews missing confidence field or failed parse: {reviews_missing_confidence}")
print(f"Reviews missing text field: {reviews_missing_text}")

review_df = pd.DataFrame(review_data)
print("\nReview Rating Score Statistics:")
print(review_df['rating_score'].describe())
print("\nReview Confidence Score Statistics:")
print(review_df['confidence_score'].describe())
print("\nReview Length (Chars) Statistics:")
print(review_df['text_length_chars'].describe())
print("\nReview Length (Words) Statistics:")
print(review_df['text_length_words'].describe())
print("")
print("-" * 30)
print("")


content_files_count = 0
content_missing_abstract = 0
content_missing_sections = 0

for conf_year in tqdm(conference_years, desc="Processing Content"):
    conf_year_path = os.path.join(dataset_root, conf_year)
    for root, _, files in os.walk(conf_year_path):
        for filename in files:
            if filename.endswith('_content.json'):
                content_files_count += 1
                filepath = os.path.join(root, filename)
                data = safe_load_json(filepath)
                if data:
                    content_data.append({'id': data.get('id', 'Unknown'), 'path': filepath})
                    if not data.get('abstractText'):
                        content_missing_abstract += 1
                    if not data.get('sections'):
                        content_missing_sections += 1


print(f"Total content files analyzed: {content_files_count}")
print(f"Content files missing 'abstractText': {content_missing_abstract}")
print(f"Content files missing 'sections': {content_missing_sections}")
print(f"(Quality Check) Paper count with hasContent=True: {total_papers - papers_missing_content}")
print(f"Match between paper flag and content file count: {content_files_count == (total_papers - papers_missing_content)}")
print("")
print("-" * 30)
print("")


aspect_file_path = os.path.join(dataset_root, 'aspect_data', 'review_with_aspect.jsonl')
total_aspect_reviews = 0
aspect_class_counts = Counter()
aspects_per_review = []
aspect_span_lengths = []
aspect_annotation_errors = 0
reviews_with_aspect_text_mismatch = 0
reviews_with_overlapping_spans = 0

if os.path.exists(aspect_file_path):
    with open(aspect_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Aspects"):
            try:
                data = json.loads(line)
                total_aspect_reviews += 1
                review_id = data.get('id')
                review_text = data.get('text')
                labels = data.get('labels', [])

                if review_id and review_text:
                    aspect_review_ids.add(review_id)
                    aspect_review_texts[review_id] = review_text
                    aspect_reviews_data.append(data)
                    aspects_per_review.append(len(labels))

                    sorted_labels = sorted(labels, key=lambda x: x[0])
                    overlap_found = False
                    for i in range(len(sorted_labels) - 1):
                        if sorted_labels[i][1] > sorted_labels[i+1][0]:
                             reviews_with_overlapping_spans += 1
                             overlap_found = True
                             break # Count only once per review

                    for start, end, label in labels:
                        aspect_class_counts[label] += 1
                        span_len = end - start
                        if span_len < 0:
                            aspect_annotation_errors += 1
                        else:
                             aspect_span_lengths.append(span_len)
                else:
                    aspect_annotation_errors += 1
            except json.JSONDecodeError:
                aspect_annotation_errors += 1

    print(f"Total reviews with aspect labels found: {total_aspect_reviews}")
    print(f"Reviews with annotation errors (e.g., missing ID/text, negative span): {aspect_annotation_errors}")
    print(f"Reviews found with overlapping aspect spans: {reviews_with_overlapping_spans}")

    print("\nAspect Class Distribution:")
    for label, count in aspect_class_counts.most_common():
        print(f"  - {label}: {count} ({count/sum(aspect_class_counts.values()):.1%})")

    print("\nAspects per Review Statistics:")
    if aspects_per_review:
        print(f"  Min: {min(aspects_per_review)}, Max: {max(aspects_per_review)}")
        print(f"  Mean: {statistics.mean(aspects_per_review):.2f}, Median: {statistics.median(aspects_per_review):.2f}")
        print(f"  StdDev: {statistics.stdev(aspects_per_review):.2f}")
    else:
        print("  No valid aspect counts found.")


    print("\nAspect Span Length (Chars) Statistics:")
    if aspect_span_lengths:
        print(f"  Min: {min(aspect_span_lengths)}, Max: {max(aspect_span_lengths)}")
        print(f"  Mean: {statistics.mean(aspect_span_lengths):.2f}, Median: {statistics.median(aspect_span_lengths):.2f}")
        print(f"  StdDev: {statistics.stdev(aspect_span_lengths):.2f}")
    else:
         print("  No valid span lengths found.")

else:
    print("Aspect data file not found.")
print("")
print("-" * 30)
print("")


if total_reviews > 0 and total_aspect_reviews > 0:
    coverage = total_aspect_reviews / total_reviews
    print(f"Aspect Label Coverage: {total_aspect_reviews} / {total_reviews} = {coverage:.1%} of all reviews have aspect labels.")
    unmatched_aspect_ids = aspect_review_ids - all_review_ids
    print(f"Number of aspect review IDs NOT found in the main review list: {len(unmatched_aspect_ids)}")
    if unmatched_aspect_ids:
        print(f"  Examples: {list(unmatched_aspect_ids)[:5]}")
else:
    print("Aspect coverage calculation skipped (no reviews or no aspect data).")

# Text Consistency Check (Sample)
print("\nChecking text consistency between aspect file and review file (sampling)")
sample_size = min(5, len(aspect_reviews_data)) # Check up to 5 reviews
if sample_size > 0:
    sample_indices = random.sample(range(len(aspect_reviews_data)), sample_size)
    mismatches_found = 0
    for i in sample_indices:
        aspect_entry = aspect_reviews_data[i]
        aspect_review_id = aspect_entry['id']
        aspect_text = aspect_entry['text']

        original_review_text = None
        review_match = review_df[review_df['review_id'] == aspect_review_id]

        if not review_match.empty:
            paper_id = review_match.iloc[0]['paper_id']
            review_idx_str = aspect_review_id.split('_review_')[-1]
            review_idx = int(review_idx_str)
            parts = paper_id.split('_')
            conf, year = parts[0], parts[1]
            review_file_path = os.path.join(dataset_root, f"{conf}_{year}", f"{paper_id}_review.json")
            original_review_json = safe_load_json(review_file_path)
            if original_review_json and 'reviews' in original_review_json and len(original_review_json['reviews']) > review_idx:
                    original_review_text = original_review_json['reviews'][review_idx].get('review')

        print(f"  Checking Review ID: {aspect_review_id}")
        if original_review_text is None:
            print("    - Could not find original review text for comparison.")
            mismatches_found += 1
        elif aspect_text == original_review_text:
            print("    - Text matches.")
            if aspect_entry['labels']:
                start, end, label = aspect_entry['labels'][0]
                try:
                     span_text = aspect_text[start:end]
                     print(f"      Sample Span [{start}:{end}] Label '{label}': '{span_text[:100]}...'")
                except IndexError:
                     print(f"      Error extracting span [{start}:{end}] for label '{label}'")
        else:
            print("    - !!! TEXT MISMATCH DETECTED !!!")
            mismatches_found += 1
            print(f"      Aspect Text Start: '{aspect_text[:80]}...'")
            print(f"      Original Text Start: '{original_review_text[:80]}...'")

    print(f"Text consistency check complete. Mismatches found in {mismatches_found}/{sample_size} samples.")
else:
    print("Skipping text consistency check (no aspect data).")

