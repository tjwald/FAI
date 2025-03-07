import time
from itertools import batched

import pandas as pd
from transformers import DistilBertTokenizer, \
    DistilBertForSequenceClassification, pipeline


def process_batch(pipeline, sentences, expected_labels):
    start_time = time.time()

    results = []
    for chunk in batched(sentences, 20):
        results.extend(pipeline(list(chunk), padding=True, truncation=True, batch_size=20))

    end_time = time.time()

    # Extract predicted labels
    predicted_labels = [1 if result['label'] == 'POSITIVE' else 0 for result in results]

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_sentence = (total_time / len(sentences)) * 1000
    accuracy = sum([1 for i in range(len(sentences)) if predicted_labels[i] == expected_labels[i]]) / len(sentences)

    # Print results
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per sentence: {avg_time_per_sentence:.2f} ms")
    print(f"Accuracy: {accuracy:.2%}")


data = pd.read_parquet('distilbert-base-uncased-finetuned-sst-2-english/train-00000-of-00001.parquet')
sentences = data['sentence'].tolist()
labels = data['label'].tolist()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipeline = pipeline('text-classification', tokenizer=tokenizer, model=model, device='cuda:0')
process_batch(pipeline, sentences, labels)
