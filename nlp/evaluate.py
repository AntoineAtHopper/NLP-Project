from searchable_index import search_contexts 
from questions_answering import answer_question

from datasets import load_dataset
from datasets import load_metric

# Load dataset & metrics
metric = load_metric("squad_v2")
dataset = load_dataset("squad_v2")

SIZE = 10 # Number of rows to test on.

predictions = []
references = []
for i, batch in enumerate(dataset['validation']):
    if i >= SIZE:
        break
    contexts = search_contexts(batch["question"])
    results = answer_question(batch["question"], " ".join(contexts))
    predictions.append({'prediction_text': results["answer"], 'id': batch["id"], 'no_answer_probability': results["score"]})
    references.append({'answers': batch["answers"], 'id': batch["id"]})

# Compute metrics
score = metric.compute(predictions=predictions, references=references)
print(score)
