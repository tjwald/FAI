# Load model directly
from torch.onnx import export
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from transformers.utils import PaddingStrategy

# Define the model name
model_dir = "./multiple_choice_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir)

tokenizer.model_max_length = 512

contexts = ["Members of the procession walk down the street holding small horn brass instruments."] * 4
sentence = "A drum line "
endings = [
    "passes by walking down the street playing their instruments.",
    "has heard approaching them.",
    "arrives and they're outside dancing and asleep.",
    "turns the lead singer watches the performance.",
]

choices = [sentence + ending for ending in endings]

tokenized = dict(tokenizer(contexts, choices, return_tensors="pt", padding=PaddingStrategy.LONGEST, truncation=True))
tokenized.pop("token_type_ids")

inputs = {k: v.view(1, 4, -1) for k, v in tokenized.items()}


export(
    model,
    f="model.onnx",
    kwargs=inputs,
    input_names=list(inputs.keys()),
    output_names=["logits"],
    opset_version=20,
    dynamic_axes={**{key: {0: "batch_size", 1: "sequence_length", 2: "sequence_length"} for key in list(inputs.keys())}, "logits": {0: "batch_size"}},
)
