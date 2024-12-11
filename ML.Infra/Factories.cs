using ML.Infra.Abstractions;
using ML.Infra.ModelExecutors.Onnx;
using ML.Infra.Pipelines;
using ML.Infra.Tokenization;

namespace ML.Infra;

public static class Factories
{
    public static async Task<TextClassificationPipeline<TClassification>> TextClassificationPipelineFromPretrained<TClassification>(string modelDir,
        TextClassificationOptions<TClassification> textClassificationOptions, PretrainedTokenizerOptions tokenizerOptions,
        OnnxModelExecutorOptions modelRunnerOptions,
        IPipelineBatchExecutor<string, ClassificationResult<TClassification>> executor)
    {
        Task<PretrainedTokenizer> tokenizer = TokenizationUtils.BpeTokenizerFromPretrained(modelDir, tokenizerOptions);
        Task<OnnxModelExecutor> modelRunner = OnnxModelExecutor.FromPretrained(modelDir, modelRunnerOptions);
        await Task.WhenAll(tokenizer, modelRunner);
        return new TextClassificationPipeline<TClassification>(tokenizer.Result, modelRunner.Result, textClassificationOptions, executor);
    }
}