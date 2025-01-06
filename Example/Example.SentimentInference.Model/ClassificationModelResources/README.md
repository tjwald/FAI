---
language: en
license: apache-2.0
datasets:
- sst2
- glue
model-index:
- name: distilbert-base-uncased-finetuned-sst-2-english
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: glue
      type: glue
      config: sst2
      split: validation
    metrics:
    - type: accuracy
      value: 0.9105504587155964
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2YyOGMxYjY2Y2JhMjkxNjIzN2FmMjNiNmM2ZWViNGY3MTNmNWI2YzhiYjYxZTY0ZGUyN2M1NGIxZjRiMjQwZiIsInZlcnNpb24iOjF9.uui0srxV5ZHRhxbYN6082EZdwpnBgubPJ5R2-Wk8HTWqmxYE3QHidevR9LLAhidqGw6Ih93fK0goAXncld_gBg
    - type: precision
      value: 0.8978260869565218
      name: Precision
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzgwYTYwYjA2MmM0ZTYwNDk0M2NmNTBkZmM2NGNhYzQ1OGEyN2NkNDQ3Mzc2NTQyMmZiNDJiNzBhNGVhZGUyOSIsInZlcnNpb24iOjF9.eHjLmw3K02OU69R2Au8eyuSqT3aBDHgZCn8jSzE3_urD6EUSSsLxUpiAYR4BGLD_U6-ZKcdxVo_A2rdXqvUJDA
    - type: recall
      value: 0.9301801801801802
      name: Recall
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMGIzM2E3MTI2Mzc2MDYwNmU3ZTVjYmZmZDBkNjY4ZTc5MGY0Y2FkNDU3NjY1MmVkNmE3Y2QzMzAwZDZhOWY1NiIsInZlcnNpb24iOjF9.PUZlqmct13-rJWBXdHm5tdkXgETL9F82GNbbSR4hI8MB-v39KrK59cqzFC2Ac7kJe_DtOeUyosj34O_mFt_1DQ
    - type: auc
      value: 0.9716626673402374
      name: AUC
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDM0YWIwZmQ4YjUwOGZmMWU2MjI1YjIxZGQ2MzNjMzRmZmYxMzZkNGFjODhlMDcyZDM1Y2RkMWZlOWQ0MWYwNSIsInZlcnNpb24iOjF9.E7GRlAXmmpEkTHlXheVkuL1W4WNjv4JO3qY_WCVsTVKiO7bUu0UVjPIyQ6g-J1OxsfqZmW3Leli1wY8vPBNNCQ
    - type: f1
      value: 0.9137168141592922
      name: F1
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMGU4MjNmOGYwZjZjMDQ1ZTkyZTA4YTc1MWYwOTM0NDM4ZWY1ZGVkNDY5MzNhYTQyZGFlNzIyZmUwMDg3NDU0NyIsInZlcnNpb24iOjF9.mW5ftkq50Se58M-jm6a2Pu93QeKa3MfV7xcBwvG3PSB_KNJxZWTCpfMQp-Cmx_EMlmI2siKOyd8akYjJUrzJCA
    - type: loss
      value: 0.39013850688934326
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTZiNzAyZDc0MzUzMmE1MGJiN2JlYzFiODE5ZTNlNGE4MmI4YzRiMTc2ODEzMTUwZmEzOTgxNzc4YjJjZTRmNiIsInZlcnNpb24iOjF9.VqIC7uYC-ZZ8ss9zQOlRV39YVOOLc5R36sIzCcVz8lolh61ux_5djm2XjpP6ARc6KqEnXC4ZtfNXsX2HZfrtCQ
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: sst2
      type: sst2
      config: default
      split: train
    metrics:
    - type: accuracy
      value: 0.9885521685548412
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2I3NzU3YzhmMDkxZTViY2M3OTY1NmI0ZTdmMDQxNjNjYzJiZmQxNzczM2E4YmExYTY5ODY0NDBkY2I4ZjNkOCIsInZlcnNpb24iOjF9.4Gtk3FeVc9sPWSqZIaeUXJ9oVlPzm-NmujnWpK2y5s1Vhp1l6Y1pK5_78wW0-NxSvQqV6qd5KQf_OAEpVAkQDA
    - type: precision
      value: 0.9881965062029833
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZDdlZDMzY2I3MTAwYTljNmM4MGMyMzU2YjAzZDg1NDYwN2ZmM2Y5OWZhMjUyMGJiNjY1YmZiMzFhMDI2ODFhNyIsInZlcnNpb24iOjF9.cqmv6yBxu4St2mykRWrZ07tDsiSLdtLTz2hbqQ7Gm1rMzq9tdlkZ8MyJRxtME_Y8UaOG9rs68pV-gKVUs8wABw
    - type: precision
      value: 0.9885521685548412
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjFlYzAzNmE1YjljNjUwNzBjZjEzZDY0ZDQyMmY5ZWM2OTBhNzNjYjYzYTk1YWE1NjU3YTMxZDQwOTE1Y2FkNyIsInZlcnNpb24iOjF9.jnCHOkUHuAOZZ_ZMVOnetx__OVJCS6LOno4caWECAmfrUaIPnPNV9iJ6izRO3sqkHRmxYpWBb-27GJ4N3LU-BQ
    - type: precision
      value: 0.9885639626373408
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZGUyODFjNjBlNTE2MTY3ZDAxOGU1N2U0YjUyY2NiZjhkOGVmYThjYjBkNGU3NTRkYzkzNDQ2MmMwMjkwMWNiMyIsInZlcnNpb24iOjF9.zTNabMwApiZyXdr76QUn7WgGB7D7lP-iqS3bn35piqVTNsv3wnKjZOaKFVLIUvtBXq4gKw7N2oWxvWc4OcSNDg
    - type: recall
      value: 0.9886145346602994
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNTU1YjlhODU3YTkyNTdiZDcwZGFlZDBiYjY0N2NjMGM2NTRiNjQ3MDNjNGMxOWY2ZGQ4NWU1YmMzY2UwZTI3YSIsInZlcnNpb24iOjF9.xaLPY7U-wHsJ3DDui1yyyM-xWjL0Jz5puRThy7fczal9x05eKEQ9s0a_WD-iLmapvJs0caXpV70hDe2NLcs-DA
    - type: recall
      value: 0.9885521685548412
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODE0YTU0MDBlOGY4YzU0MjY5MzA3OTk2OGNhOGVkMmU5OGRjZmFiZWI2ZjY5ODEzZTQzMTI0N2NiOTVkNDliYiIsInZlcnNpb24iOjF9.SOt1baTBbuZRrsvGcak2sUwoTrQzmNCbyV2m1_yjGsU48SBH0NcKXicidNBSnJ6ihM5jf_Lv_B5_eOBkLfNWDQ
    - type: recall
      value: 0.9885521685548412
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWNkNmM0ZGRlNmYxYzIwNDk4OTI5MzIwZWU1NzZjZDVhMDcyNDFlMjBhNDQxODU5OWMwMWNhNGEzNjY3ZGUyOSIsInZlcnNpb24iOjF9.b15Fh70GwtlG3cSqPW-8VEZT2oy0CtgvgEOtWiYonOovjkIQ4RSLFVzVG-YfslaIyfg9RzMWzjhLnMY7Bpn2Aw
    - type: f1
      value: 0.9884019815052447
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmM4NjQ5Yjk5ODRhYTU1MTY3MmRhZDBmODM1NTg3OTFiNWM4NDRmYjI0MzZkNmQ1MzE3MzcxODZlYzBkYTMyYSIsInZlcnNpb24iOjF9.74RaDK8nBVuGRl2Se_-hwQvP6c4lvVxGHpcCWB4uZUCf2_HoC9NT9u7P3pMJfH_tK2cpV7U3VWGgSDhQDi-UBQ
    - type: f1
      value: 0.9885521685548412
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZDRmYWRmMmQ0YjViZmQxMzhhYTUyOTE1MTc0ZDU1ZjQyZjFhMDYzYzMzZDE0NzZlYzQyOTBhMTBhNmM5NTlkMiIsInZlcnNpb24iOjF9.VMn_psdAHIZTlW6GbjERZDe8MHhwzJ0rbjV_VJyuMrsdOh5QDmko-wEvaBWNEdT0cEKsbggm-6jd3Gh81PfHAQ
    - type: f1
      value: 0.9885546181087554
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjUyZWFhZDZhMGQ3MzBmYmRiNDVmN2FkZDBjMjk3ODk0OTAxNGZkMWE0NzU5ZjI0NzE0NGZiNzM0N2Y2NDYyOSIsInZlcnNpb24iOjF9.YsXBhnzEEFEW6jw3mQlFUuIrW7Gabad2Ils-iunYJr-myg0heF8NEnEWABKFE1SnvCWt-69jkLza6SupeyLVCA
    - type: loss
      value: 0.040652573108673096
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTc3YjU3MjdjMzkxODA5MjU5NGUyY2NkMGVhZDg3ZWEzMmU1YWVjMmI0NmU2OWEyZTkzMTVjNDZiYTc0YjIyNCIsInZlcnNpb24iOjF9.lA90qXZVYiILHMFlr6t6H81Oe8a-4KmeX-vyCC1BDia2ofudegv6Vb46-4RzmbtuKeV6yy6YNNXxXxqVak1pAg
---

# DistilBERT base uncased finetuned SST-2

## Table of Contents
- [Model Details](#model-details)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)

## Model Details
**Model Description:** This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).
- **Developed by:** Hugging Face
- **Model Type:** Text Classification
- **Language(s):** English
- **License:** Apache-2.0
- **Parent Model:** For more details about DistilBERT, we encourage users to check out [this model card](https://huggingface.co/distilbert-base-uncased).
- **Resources for more information:**
    - [Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification)
    - [DistilBERT paper](https://arxiv.org/abs/1910.01108)

## How to Get Started With the Model

Example of single-label classification:
​​
```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

```

## Uses

#### Direct Use

This model can be used for  topic classification. You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

#### Misuse and Out-of-scope Use
The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.


## Risks, Limitations and Biases

Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift. In this [colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb), [Aurélien Géron](https://twitter.com/aureliengeron) made an interesting map plotting these probabilities for each country.

<img src="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/map.jpeg" alt="Map of positive probabilities per country." width="500"/>

We strongly advise users to thoroughly probe these aspects on their use-cases in order to evaluate the risks of this model. We recommend looking at the following bias evaluation datasets as a place to start: [WinoBias](https://huggingface.co/datasets/wino_bias), [WinoGender](https://huggingface.co/datasets/super_glue), [Stereoset](https://huggingface.co/datasets/stereoset).



# Training


#### Training Data


The authors use the following Stanford Sentiment Treebank([sst2](https://huggingface.co/datasets/sst2)) corpora for the model.

#### Training Procedure

###### Fine-tuning hyper-parameters


- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 3.0


