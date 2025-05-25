import os
import json
from collections import Counter, defaultdict
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

def analyze_papers(conference_years_list, data_root, category_name):
    """Analyzes paper metadata for a list of conference-years."""
    print(f"\n--- 2. Analyzing Paper Metadata ({category_name}) ---")
    paper_metadata = []
    total_papers = 0
    decision_counts = Counter()
    papers_missing_content = 0
    papers_missing_reviews = 0
    error_logs = defaultdict(list)

    for conf_year in tqdm(conference_years_list, desc=f"Processing Papers ({category_name})"):
        conf_year_path = os.path.join(data_root, conf_year)
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
                        if not data.get('hasContent', True):
                            papers_missing_content += 1
                        if not data.get('hasReview', True):
                            papers_missing_reviews += 1
                    else:
                        error_logs['paper_load_errors'].append(filepath)

    print(f"Total {category_name} papers analyzed: {total_papers}")
    if total_papers > 0:
        print("Decision Distribution:")
        for decision, count in decision_counts.most_common():
            print(f"  - {decision}: {count} ({count/total_papers:.1%})")
        print(f"Papers flagged as missing content: {papers_missing_content}")
        print(f"Papers flagged as missing reviews: {papers_missing_reviews}")
    else:
        print("No papers found for this category.")

    return paper_metadata, total_papers, decision_counts, papers_missing_content, papers_missing_reviews, error_logs

def analyze_reviews(conference_years_list, data_root, category_name):
    """Analyzes review metadata for a list of conference-years."""
    print(f"\n--- 3. Analyzing Review Metadata ({category_name}) ---")
    review_data = []
    all_review_ids = set()
    total_reviews = 0
    review_ratings = []
    review_confidences = []
    review_lengths_chars = []
    review_lengths_words = []
    reviews_missing_rating = 0
    reviews_missing_confidence = 0
    reviews_missing_text = 0
    error_logs = defaultdict(list)
    review_texts_map = {} # Store text for later comparison {review_id: text}

    for conf_year in tqdm(conference_years_list, desc=f"Processing Reviews ({category_name})"):
        conf_year_path = os.path.join(data_root, conf_year)
        for root, _, files in os.walk(conf_year_path):
            for filename in files:
                if filename.endswith('_review.json'):
                    filepath = os.path.join(root, filename)
                    paper_review_data = safe_load_json(filepath)
                    if paper_review_data:
                        paper_id = paper_review_data.get('id')
                        if not paper_id or not paper_id.startswith(category_name):
                            error_logs['review_id_mismatch'].append(f"Expected {category_name} but got {paper_id} in {filepath}")
                            continue # Skip reviews not belonging to the current category

                        reviews_list = paper_review_data.get('reviews', [])
                        for idx, review in enumerate(reviews_list):
                            total_reviews += 1
                            # Use a consistent format that includes category implicitly via paper_id
                            review_id = f"{paper_id}_review_{idx}"
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
                                if rating_str: error_logs['rating_parse_fail'].append(f"{review_id}: {rating_str}")

                            if confidence_score is not None:
                                review_confidences.append(confidence_score)
                            else:
                                reviews_missing_confidence += 1
                                if confidence_str: error_logs['confidence_parse_fail'].append(f"{review_id}: {confidence_str}")

                            if review_text and isinstance(review_text, str):
                                review_lengths_chars.append(len(review_text))
                                review_lengths_words.append(len(review_text.split()))
                                review_texts_map[review_id] = review_text # Store text
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
                    else:
                        error_logs['review_load_errors'].append(filepath)

    print(f"Total {category_name} reviews analyzed: {total_reviews}")
    if total_reviews > 0:
        print(f"Reviews missing rating field or failed parse: {reviews_missing_rating}")
        print(f"Reviews missing confidence field or failed parse: {reviews_missing_confidence}")
        print(f"Reviews missing text field: {reviews_missing_text}")

        # Convert to DataFrame for easier stats
        review_df = pd.DataFrame(review_data)

        print("\nReview Rating Score Statistics:")
        print(review_df['rating_score'].describe().to_string())
        print("\nReview Confidence Score Statistics:")
        print(review_df['confidence_score'].describe().to_string())
        print("\nReview Length (Chars) Statistics:")
        print(review_df['text_length_chars'].describe().to_string())
        print("\nReview Length (Words) Statistics:")
        print(review_df['text_length_words'].describe().to_string())
    else:
        print("No reviews found for this category.")
        review_df = pd.DataFrame() # Return empty DataFrame

    return review_df, all_review_ids, total_reviews, review_texts_map, error_logs

def analyze_content(conference_years_list, data_root, category_name, total_papers_category, papers_missing_content_category):
    """Analyzes content files for a list of conference-years."""
    print(f"\n--- 4. Analyzing Content Files ({category_name}) ---")
    content_files_count = 0
    content_missing_abstract = 0
    content_missing_sections = 0
    error_logs = defaultdict(list)

    for conf_year in tqdm(conference_years_list, desc=f"Processing Content ({category_name})"):
        conf_year_path = os.path.join(data_root, conf_year)
        for root, _, files in os.walk(conf_year_path):
            for filename in files:
                if filename.endswith('_content.json'):
                    paper_id = filename.replace('_content.json', '')
                    if not paper_id.startswith(category_name):
                         continue # Skip content not belonging to the current category

                    content_files_count += 1
                    filepath = os.path.join(root, filename)
                    data = safe_load_json(filepath)
                    if data:
                        if not data.get('abstractText'):
                            content_missing_abstract += 1
                        if not data.get('sections'):
                            content_missing_sections += 1
                    else:
                        error_logs['content_load_errors'].append(filepath)

    print(f"Total {category_name} content files analyzed: {content_files_count}")
    if content_files_count > 0:
        print(f"Content files missing 'abstractText': {content_missing_abstract}")
        print(f"Content files missing 'sections': {content_missing_sections}")
    else:
        print("No content files found for this category.")

    expected_content_files = total_papers_category - papers_missing_content_category
    print(f"(Quality Check) Expected content files based on paper flags: {expected_content_files}")
    if total_papers_category > 0: # Avoid division by zero if no papers
        match = content_files_count == expected_content_files
        print(f"Match between paper flag and content file count: {match}")
        if not match:
             error_logs['content_file_count_mismatch'].append(f"Expected {expected_content_files}, Found {content_files_count}")
    else:
        print("Match check skipped (no papers in category).")


    return content_files_count, error_logs

def analyze_aspects(aspect_reviews_list, category_name):
    """Analyzes a list of aspect review data for a specific category."""
    print(f"\n--- 5. Analyzing Aspect Data ({category_name}) ---")
    total_aspect_reviews = len(aspect_reviews_list)
    aspect_class_counts = Counter()
    aspects_per_review = []
    aspect_span_lengths = []
    aspect_annotation_errors = 0
    reviews_with_overlapping_spans = 0
    aspect_review_ids_category = set()
    error_logs = defaultdict(list)

    if not aspect_reviews_list:
        print(f"No aspect data found for {category_name}.")
        return (total_aspect_reviews, aspect_review_ids_category,
                aspect_class_counts, aspects_per_review, aspect_span_lengths,
                reviews_with_overlapping_spans, error_logs)

    for data in tqdm(aspect_reviews_list, desc=f"Analyzing Aspects ({category_name})"):
        review_id = data.get('id')
        review_text = data.get('text')
        labels = data.get('labels', [])

        if review_id and review_text:
            aspect_review_ids_category.add(review_id)
            aspects_per_review.append(len(labels))

            # Check for overlapping spans
            sorted_labels = sorted(labels, key=lambda x: x[0] if isinstance(x, list) and len(x)>1 else 999999) # Handle potential malformed labels
            overlap_found = False
            for i in range(len(sorted_labels) - 1):
                 # Check if labels are valid lists with at least start and end
                 if (isinstance(sorted_labels[i], list) and len(sorted_labels[i]) >= 2 and
                     isinstance(sorted_labels[i+1], list) and len(sorted_labels[i+1]) >= 1 and
                     isinstance(sorted_labels[i][1], (int, float)) and isinstance(sorted_labels[i+1][0], (int, float))):

                    if sorted_labels[i][1] > sorted_labels[i+1][0]:
                        reviews_with_overlapping_spans += 1
                        overlap_found = True
                        error_logs['overlapping_spans'].append(f"{review_id} - Span {i} ends at {sorted_labels[i][1]}, Span {i+1} starts at {sorted_labels[i+1][0]}")
                        break # Count only once per review
                 else:
                     aspect_annotation_errors += 1
                     error_logs['malformed_label_structure'].append(f"{review_id} - Label pair at index {i}")


            for label_data in labels:
                # Check label format before unpacking
                if isinstance(label_data, list) and len(label_data) == 3:
                    start, end, label = label_data
                    # Further type checks if needed
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and isinstance(label, str):
                        aspect_class_counts[label] += 1
                        span_len = end - start
                        if span_len < 0:
                            aspect_annotation_errors += 1
                            error_logs['negative_span_length'].append(f"{review_id} - Label: {label}, Start: {start}, End: {end}")
                        else:
                            aspect_span_lengths.append(span_len)
                    else:
                         aspect_annotation_errors += 1
                         error_logs['malformed_label_content'].append(f"{review_id} - Label: {label_data}")
                else:
                     aspect_annotation_errors += 1
                     error_logs['malformed_label_structure'].append(f"{review_id} - Label: {label_data}")

        else:
            aspect_annotation_errors += 1
            error_logs['missing_id_or_text_aspect'].append(f"Data: {str(data)[:100]}") # Log part of the data

    print(f"Total {category_name} reviews with aspect labels: {total_aspect_reviews}")
    print(f"Reviews with annotation errors (format, negative span, etc.): {aspect_annotation_errors}")
    print(f"Reviews found with overlapping aspect spans: {reviews_with_overlapping_spans}")

    if total_aspect_reviews > 0:
        print("\nAspect Class Distribution:")
        total_aspect_labels = sum(aspect_class_counts.values())
        if total_aspect_labels > 0:
             for label, count in aspect_class_counts.most_common():
                 print(f"  - {label}: {count} ({count/total_aspect_labels:.1%})")
        else:
             print("  No valid aspect labels found.")


        print("\nAspects per Review Statistics:")
        if aspects_per_review:
            print(f"  Min: {min(aspects_per_review)}, Max: {max(aspects_per_review)}")
            print(f"  Mean: {statistics.mean(aspects_per_review):.2f}, Median: {statistics.median(aspects_per_review):.2f}")
            if len(aspects_per_review) > 1:
                 print(f"  StdDev: {statistics.stdev(aspects_per_review):.2f}")
        else:
            print("  No valid aspect counts found.")


        print("\nAspect Span Length (Chars) Statistics:")
        if aspect_span_lengths:
            print(f"  Min: {min(aspect_span_lengths)}, Max: {max(aspect_span_lengths)}")
            print(f"  Mean: {statistics.mean(aspect_span_lengths):.2f}, Median: {statistics.median(aspect_span_lengths):.2f}")
            if len(aspect_span_lengths) > 1:
                print(f"  StdDev: {statistics.stdev(aspect_span_lengths):.2f}")
        else:
            print("  No valid span lengths found.")

    return (total_aspect_reviews, aspect_review_ids_category,
            aspect_class_counts, aspects_per_review, aspect_span_lengths,
            reviews_with_overlapping_spans, error_logs)


def perform_cross_checks(category_name, total_reviews_category, aspect_review_ids_category,
                         all_review_ids_category, aspect_reviews_list_category, review_texts_map_category):
    """Performs cross-data checks for a specific category."""
    print(f"\n--- 6. Cross-Data Quality Checks ({category_name}) ---")
    error_logs = defaultdict(list)
    total_aspect_reviews_category = len(aspect_reviews_list_category)

    # Aspect Coverage
    if total_reviews_category > 0 and total_aspect_reviews_category > 0:
        coverage = total_aspect_reviews_category / total_reviews_category
        print(f"Aspect Label Coverage: {total_aspect_reviews_category} / {total_reviews_category} = {coverage:.1%} of {category_name} reviews have aspect labels.")
        # Check ID matching precisely
        unmatched_aspect_ids = aspect_review_ids_category - all_review_ids_category
        print(f"Number of aspect review IDs NOT found in the {category_name} review list: {len(unmatched_aspect_ids)}")
        if unmatched_aspect_ids:
            print(f"  Examples: {list(unmatched_aspect_ids)[:5]}") # Show a few examples
            error_logs['unmatched_aspect_ids'].extend(list(unmatched_aspect_ids))
    elif total_reviews_category == 0:
         print(f"Aspect coverage check skipped ({category_name}: No reviews found).")
    else:
         print(f"Aspect coverage check skipped ({category_name}: No aspect reviews found).")


    # Text Consistency Check (Sample)
    print(f"\nChecking text consistency between aspect file and review file ({category_name} - sampling)...")
    sample_size = min(5, len(aspect_reviews_list_category)) # Check up to 5 reviews
    if sample_size > 0:
        sample_indices = random.sample(range(len(aspect_reviews_list_category)), sample_size)
        mismatches_found = 0
        for i in sample_indices:
            aspect_entry = aspect_reviews_list_category[i]
            aspect_review_id = aspect_entry['id']
            aspect_text = aspect_entry['text']

            # Find the corresponding original review text using the pre-loaded map
            original_review_text = review_texts_map_category.get(aspect_review_id)

            print(f"  Checking Review ID: {aspect_review_id}")
            if original_review_text is None:
                print(f"    - Could not find original review text in {category_name} map for comparison.")
                mismatches_found += 1
                error_logs['original_review_lookup_fail'].append(aspect_review_id)
            elif aspect_text == original_review_text:
                print("    - Text matches.")
                # Optional: Check a span boundary
                labels = aspect_entry.get('labels', [])
                if labels and isinstance(labels[0], list) and len(labels[0]) == 3:
                    start, end, label = labels[0] # Check first label
                    try:
                        if isinstance(start, int) and isinstance(end, int) and isinstance(aspect_text, str):
                            span_text = aspect_text[start:end]
                            print(f"      Sample Span [{start}:{end}] Label '{label}': '{span_text[:100]}...'") # Print start of span
                        else:
                             print(f"      Invalid types for span extraction: start={type(start)}, end={type(end)}")
                             error_logs['span_extraction_error'].append(f"{aspect_review_id} Label: {label} InvalidTypes Start:{start} End:{end}")

                    except IndexError:
                        print(f"      Error extracting span [{start}:{end}] for label '{label}' - Likely index out of bounds")
                        error_logs['span_extraction_error'].append(f"{aspect_review_id} Label: {label} IndexError Start:{start} End:{end} TextLen:{len(aspect_text)}")
                    except Exception as e:
                         print(f"      Error extracting span [{start}:{end}] for label '{label}': {e}")
                         error_logs['span_extraction_error'].append(f"{aspect_review_id} Label: {label} Exception:{e} Start:{start} End:{end}")

            else:
                print("    - !!! TEXT MISMATCH DETECTED !!!")
                mismatches_found += 1
                print(f"      Aspect Text Start: '{aspect_text[:80]}...'")
                print(f"      Original Text Start: '{original_review_text[:80]}...'")
                error_logs['text_mismatch'].append(aspect_review_id)

        print(f"Text consistency check complete. Mismatches found in {mismatches_found}/{sample_size} samples.")

    else:
        print(f"Skipping text consistency check ({category_name}: No aspect data).")

    return error_logs


def print_final_report(category_name, all_errors_dict):
    """Prints a summary of errors for the category."""
    print(f"\n--- 7. Final Summary & Error Report ({category_name}) ---")
    has_errors = any(all_errors_dict.values())

    if has_errors:
        print("\nPotential Issues / Errors Logged:")
        for category, messages in all_errors_dict.items():
            if messages: # Only print categories with messages
                print(f"  - {category}: {len(messages)} occurrences")
                # Print first few examples for context
                for i, msg in enumerate(messages[:3]):
                    # Truncate long messages like file paths if needed
                    msg_str = str(msg)
                    print(f"    Example {i+1}: {msg_str[:200]}{'...' if len(msg_str) > 200 else ''}")
    else:
        print("\nNo major errors logged during processing for this category.")

# --- Main Execution ---

print("--- Starting Dataset Analysis (Separated by ICLR/NIPS) ---")
print(f"Root directory: {dataset_root}\n")

# 1. Identify Conference Categories and Years
print("--- 1. Identifying Conference Categories and Years ---")
iclr_conf_years = []
nips_conf_years = []
other_items = []

if not os.path.isdir(dataset_root):
    print(f"ERROR: Dataset root directory not found at '{dataset_root}'")
    exit()

for item in os.listdir(dataset_root):
    item_path = os.path.join(dataset_root, item)
    if os.path.isdir(item_path):
        if item.startswith('ICLR_') and item.split('_')[1].isdigit():
            iclr_conf_years.append(item)
        elif item.startswith('NIPS_') and item.split('_')[1].isdigit():
            nips_conf_years.append(item)
        elif item != 'aspect_data': # Ignore aspect_data here
            other_items.append(item)

iclr_conf_years.sort()
nips_conf_years.sort()

print(f"Found ICLR Conference-Years: {', '.join(iclr_conf_years)}")
print(f"Found NIPS Conference-Years: {', '.join(nips_conf_years)}")
if other_items:
    print(f"Found other top-level directories: {', '.join(other_items)}")
print("-" * 50)


# --- Load and Partition Aspect Data FIRST ---
print("--- Pre-loading and Partitioning Aspect Data ---")
aspect_file_path = os.path.join(dataset_root, 'aspect_data', 'review_with_aspect.jsonl')
all_aspect_reviews_list = []
iclr_aspect_reviews_list = []
nips_aspect_reviews_list = []
other_aspect_reviews_list = []
aspect_load_errors = 0

if os.path.exists(aspect_file_path):
    print(f"Loading aspect data from: {aspect_file_path}")
    with open(aspect_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                all_aspect_reviews_list.append(data)
                review_id = data.get('id')
                if review_id:
                    if review_id.startswith('ICLR_'):
                        iclr_aspect_reviews_list.append(data)
                    elif review_id.startswith('NIPS_'):
                        nips_aspect_reviews_list.append(data)
                    else:
                        other_aspect_reviews_list.append(data)
                else:
                    # Handle cases where ID might be missing in aspect file
                    other_aspect_reviews_list.append(data)
            except json.JSONDecodeError:
                print(f"Warning: JSON decode error in aspect file, line {line_num + 1}")
                aspect_load_errors += 1

    print(f"Total aspect entries loaded: {len(all_aspect_reviews_list)}")
    print(f" - Assigned to ICLR: {len(iclr_aspect_reviews_list)}")
    print(f" - Assigned to NIPS: {len(nips_aspect_reviews_list)}")
    print(f" - Assigned to Other/Unknown: {len(other_aspect_reviews_list)}")
    if aspect_load_errors > 0:
         print(f" - Aspect file load/parse errors: {aspect_load_errors}")

else:
    print(f"Aspect data file not found at: {aspect_file_path}")
    print("Skipping aspect analysis and related cross-checks.")
print("-" * 50)


# --- Analyze ICLR ---
print("\n" + "=" * 30 + " ICLR Analysis " + "=" * 30)
iclr_errors = defaultdict(list)

(iclr_paper_metadata, iclr_total_papers, iclr_decision_counts,
 iclr_papers_missing_content, iclr_papers_missing_reviews, paper_err) = analyze_papers(iclr_conf_years, dataset_root, "ICLR")
iclr_errors.update(paper_err)

(iclr_review_df, iclr_all_review_ids, iclr_total_reviews,
 iclr_review_texts_map, review_err) = analyze_reviews(iclr_conf_years, dataset_root, "ICLR")
iclr_errors.update(review_err)

iclr_content_files_count, content_err = analyze_content(iclr_conf_years, dataset_root, "ICLR", iclr_total_papers, iclr_papers_missing_content)
iclr_errors.update(content_err)

if all_aspect_reviews_list: # Only run if aspect file was loaded
    (iclr_total_aspect_reviews, iclr_aspect_review_ids, _, _, _, _, aspect_err) = analyze_aspects(iclr_aspect_reviews_list, "ICLR")
    iclr_errors.update(aspect_err)

    cross_check_err = perform_cross_checks("ICLR", iclr_total_reviews, iclr_aspect_review_ids,
                                         iclr_all_review_ids, iclr_aspect_reviews_list, iclr_review_texts_map)
    iclr_errors.update(cross_check_err)

print_final_report("ICLR", iclr_errors)
print("=" * (75))


# --- Analyze NIPS ---
print("\n" + "=" * 30 + " NIPS Analysis " + "=" * 30)
nips_errors = defaultdict(list)

(nips_paper_metadata, nips_total_papers, nips_decision_counts,
 nips_papers_missing_content, nips_papers_missing_reviews, paper_err) = analyze_papers(nips_conf_years, dataset_root, "NIPS")
nips_errors.update(paper_err)

(nips_review_df, nips_all_review_ids, nips_total_reviews,
 nips_review_texts_map, review_err) = analyze_reviews(nips_conf_years, dataset_root, "NIPS")
nips_errors.update(review_err)

nips_content_files_count, content_err = analyze_content(nips_conf_years, dataset_root, "NIPS", nips_total_papers, nips_papers_missing_content)
nips_errors.update(content_err)

if all_aspect_reviews_list: # Only run if aspect file was loaded
    (nips_total_aspect_reviews, nips_aspect_review_ids, _, _, _, _, aspect_err) = analyze_aspects(nips_aspect_reviews_list, "NIPS")
    nips_errors.update(aspect_err)

    cross_check_err = perform_cross_checks("NIPS", nips_total_reviews, nips_aspect_review_ids,
                                         nips_all_review_ids, nips_aspect_reviews_list, nips_review_texts_map)
    nips_errors.update(cross_check_err)


print_final_report("NIPS", nips_errors)
print("=" * (75))

print("\n--- Full Dataset Analysis Complete ---")