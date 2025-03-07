from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

model_dir = './multiple_choice_model/onnx/'

# Define the optimization configuration
optimization_config = OptimizationConfig(optimization_level=99, fp16=True, disable_shape_inference=True)

# Create the optimizer
optimizer = ORTOptimizer.from_pretrained(model_dir)

# Optimize the model
optimizer.optimize(save_dir=model_dir, file_suffix='optimized.onnx',
                   optimization_config=optimization_config)

print("Model optimized and saved to 'optimized_model' directory.")
