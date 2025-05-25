from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
import pandas as pd
from ragas.metrics import ContextRelevance
from datasets import Dataset
from metric_calculators.check_openai_key import check_openai_api_key

if not check_openai_api_key():
    print("a valid OPENAI_API_KEY is not set. Aborting")
    exit()

dummy_data = {
    "question": ["What is the capital of France?"],
    "contexts": [["Paris is a city in France.", "The Eiffel Tower is a famous landmark."]],
    "answer": ["Paris."],
    "ground_truth": ["The capital of France is Paris."] 
}
dummy_dataset = Dataset.from_dict(dummy_data)


try:
    result = evaluate(dummy_dataset, metrics=[ContextRelevance()], raise_exceptions=True)
    print("Minimal RAGAS test successful:", result)
except Exception as e:
    print("Minimal RAGAS test failed:", e)