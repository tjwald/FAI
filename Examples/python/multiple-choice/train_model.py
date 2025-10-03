# Copied from: https://huggingface.co/docs/transformers/en/tasks/multiple_choice

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer, DataCollatorForMultipleChoice, Trainer, TrainingArguments

print("load dataset")
swag = load_dataset("swag", "regular")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
ending_names = ["ending0", "ending1", "ending2", "ending3"]


def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


print("preprocess")
tokenized_swag = swag.map(preprocess_function, batched=True)
collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)


print("load accuracy")
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
model = model.to("cuda:0")

training_args = TrainingArguments(
    output_dir="multiple_choice_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    use_cpu=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["validation"],
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

print("training")
trainer.train()
