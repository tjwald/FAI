import time
from itertools import batched

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, \
    DistilBertForSequenceClassification, pipeline
from transformers.pipelines.base import KeyDataset


def process_batch(pipeline, sentences, expected_labels):
    print("started: ")
    start_time = time.time()

    results = []
    for chunk in batched(sentences, 20):
        results.extend(pipeline(list(chunk), padding=True, truncation=True, batch_size=20))

    predicted_labels = [1 if result['label'] == 'POSITIVE' else 0 for result in results]

    end_time = time.time()

    print_run_statistics(end_time, expected_labels, predicted_labels, sentences, start_time)


def process_datasets(pipeline, sentences, expected_labels):
    print("started: ")
    start_time = time.time()

    dataset = Dataset.from_dict({"text": sentences})
    results = pipeline(KeyDataset(dataset, key='text'), padding=True, truncation=True, batch_size=20)
    predicted_labels = [1 if result['label'] == 'POSITIVE' else 0 for result in results]

    end_time = time.time()
    print("ended")

    print_run_statistics(end_time, expected_labels, predicted_labels, sentences, start_time)


def print_run_statistics(end_time, expected_labels, predicted_labels, sentences, start_time):
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_sentence = (total_time / len(sentences)) * 1000
    accuracy = sum([1 for i in range(len(sentences)) if predicted_labels[i] == expected_labels[i]]) / len(sentences)
    # Print results
    print(f"Total time taken: {total_time:.8f} seconds")
    if avg_time_per_sentence < 1:
        avg_time_per_sentence *= 1000
        print(f"Average time: {avg_time_per_sentence:.2f} µs/it")
    else:
        print(f"Average time: {avg_time_per_sentence:.2f} ms/it")
    print(f"Accuracy: {accuracy:.2%}")


data = pd.read_parquet('distilbert-base-uncased-finetuned-sst-2-english/train-00000-of-00001.parquet')
# sentences = data['sentence'].tolist()
# labels = data['label'].tolist()

sentences = data['sentence'].tolist()[:10000]
labels = data['label'].tolist()[:10000]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipeline = pipeline('text-classification', tokenizer=tokenizer, model=model, device='cuda:0')
process_batch(pipeline, sentences, labels)

# process_datasets(pipeline, sentences, labels)