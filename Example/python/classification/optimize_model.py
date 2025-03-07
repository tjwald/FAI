from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# Define the optimization configuration
optimization_config = OptimizationConfig(optimization_level=2, fp16=True, disable_shape_inference=True)

# Create the optimizer
optimizer = ORTOptimizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english/onnx/")

# Optimize the model
optimizer.optimize(save_dir="distilbert-base-uncased-finetuned-sst-2-english/onnx/optimized",
                   optimization_config=optimization_config)

print("Model optimized and saved to 'optimized_model' directory.")
