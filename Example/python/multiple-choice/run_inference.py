import time

import numpy as np
import pandas as pd
import torch
from more_itertools import chunked, flatten
from tqdm import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Load the model and tokenizer from disk
model_directory = "./multiple_choice_model"
model = AutoModelForMultipleChoice.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ensure the model runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load data from the Parquet file
parquet_file = "./multiple_choice_model/swag_train.parquet"
data_df = pd.read_parquet(parquet_file)


start = time.time()

# Preprocess the data
# Assuming your Parquet file contains the input texts as "input_texts" column (adjust as needed)
option_count = 4
contexts = flatten([context] * option_count for context in data_df["sent1"].tolist())
options = (
    (data_df["sent2"] + " " + data_df["ending0"]).tolist()
    + (data_df["sent2"] + " " + data_df["ending1"]).tolist()
    + (data_df["sent2"] + " " + data_df["ending2"]).tolist()
    + (data_df["sent2"] + " " + data_df["ending3"]).tolist()
)

inputs = [(context, option) for context, option in zip(contexts, options)]

# Tokenize the inputs in batches
# batch_size = option_count * 5  # 5375it / 30s
batch_size = option_count * 10  # 5410it / 30s
# batch_size = option_count * 15  # 5265it / 30s
# batch_size = option_count * 20  # 5100it / 30s
# batch_size = option_count * 50  # 4350it / 30s

# Prepare for inference
results = []
model.eval()  # Set the model to evaluation mode

with torch.no_grad(), tqdm(total=len(inputs) // option_count, desc="Processing batches") as progress_bar:
    for batch in chunked(inputs, batch_size):
        # Tokenize the batch
        actual_batch_size = len(batch)
        tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        attention_mask = tokenized.attention_mask.view(actual_batch_size // option_count, option_count, -1).to(device)
        input_ids = tokenized.input_ids.view(actual_batch_size // option_count, option_count, -1).to(device)

        # Perform inference
        outputs = model(attention_mask=attention_mask, input_ids=input_ids)
        logits = outputs.logits  # Extract logits (raw predictions)
        predictions = torch.argmax(logits, dim=-1)  # Get predicted labels

        # Store results
        results.extend(predictions.cpu().tolist())
        progress_bar.update(actual_batch_size // option_count)

end = time.time()

accuracy_score = (data_df["label"].array == np.array(results)).sum() / len(results)

print(f"Took: {end - start} seconds")
print(f"Accuracy: {accuracy_score}")
