using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;

namespace FAI.Core.InferenceTasks.Classification;

public class ClassificationTask<TInput, TPreprocessContainer, TPreprocess, TClassification, TScore>
    : InferenceSteps<
        TInput,
        TPreprocessContainer,
        ClassificationResult<TClassification, TScore>[],
        ClassificationResult<TClassification, TScore>>
    where TScore : unmanaged, IFloatingPointIeee754<TScore>
    where TPreprocessContainer : IEnumerable<Tensor<TPreprocess>>
{
    private readonly IPreprocessor<TInput, TPreprocessContainer, TPreprocess> _preprocessor;
    private readonly IModelExecutor<TPreprocess, TScore> _modelExecutor;
    private readonly ClassificationOptions<TClassification> _pipelineOptions;

    protected ClassificationTask(IPreprocessor<TInput, TPreprocessContainer, TPreprocess> preprocessor, IModelExecutor<TPreprocess, TScore> modelExecutor,
        ClassificationOptions<TClassification> pipelineOptions)
    {
        _preprocessor = preprocessor;
        _modelExecutor = modelExecutor;
        _pipelineOptions = pipelineOptions;
    }

    public override TPreprocessContainer Preprocess(ReadOnlySpan<TInput> input)
    {
        return _preprocessor.Preprocess(input);
    }

    public override async Task<ClassificationResult<TClassification, TScore>[]> RunModel(ReadOnlyMemory<TInput> input, TPreprocessContainer preprocesses)
    {
        var outputs = new ClassificationResult<TClassification, TScore>[input.Length];
        await _modelExecutor.RunAsync(preprocesses.ToArray(), (logits, _) =>
        {
            int indexInBatch = 0;
            foreach (ReadOnlyTensorSpan<TScore> rowLogits in logits.GetDimensionSpan(0))
            {
                outputs[indexInBatch] = _pipelineOptions.GetClassificationResult(rowLogits.AsSpan());
                indexInBatch++;
            }
        });
        return outputs;
    }

    public override void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocessContainer preprocesses,
        ClassificationResult<TClassification, TScore>[] modelOutput, Span<ClassificationResult<TClassification, TScore>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }
}