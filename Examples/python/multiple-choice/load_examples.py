from datasets import load_dataset

swag = load_dataset("swag", "regular", keep_in_memory=False)
swag["train"].to_parquet("swag_train.parquet")
