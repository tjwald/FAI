using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Factories;
using ML.NLP.Configuration;
using ML.NLP.InferenceTasks;
using ML.NLP.PipelineBatchExecutors;
using ML.NLP.Tokenization;
using ML.Onnx.Factories;
using ML.Infra.ResultTypes;

namespace Example.SentimentInference.Model;

public static class SentimentInferenceFactory
{
    public static async Task<IInference<string, bool>> CreateSentimentInference(SentimentInferenceOptions options)
    {
        Console.WriteLine($"Model: {options.ModelDir}");
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);

        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.ModelExecutorOptions);

        return CreateSentimentInference(options, tokenizer, modelExecutor);
    }

    private static SentimentInference CreateSentimentInference(SentimentInferenceOptions options, PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor)
    {
        IInferenceSteps<TokenizedText, ClassificationResult<bool>> textClassificationTask =
            new TextClassification<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]));

        IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>> executor;
        if (options.PipeBatchExecutorOptions is TokenBasedBatchExecutorOptions opt)
        {
            executor =
                PipelineBatchExecutorFactory.CreatePipelineBatchExecutor<TokenizedText, BatchTokenizedResult, Tensor<float>[], ClassificationResult<bool>>(opt.InternalExecutorOptions,
                    textClassificationTask);

            if (opt.MaxTokensCount.HasValue)
            {
                Console.WriteLine("Using TokenBatchSize chunking");
                executor = new TokenBatchSizeBatchExecutor<ClassificationResult<bool>>(executor, opt.MaxTokensCount.Value);
            }

            if (opt.SortTokens)
            {
                Console.WriteLine("Using Sort by token count execution");
                executor = new TokenCountSortingBatchExecutor<ClassificationResult<bool>>(tokenizer, executor);
            }
        }
        else
        {
            executor = PipelineBatchExecutorFactory.CreatePipelineBatchExecutor<TokenizedText, BatchTokenizedResult, Tensor<float>[], ClassificationResult<bool>>(
                options.PipeBatchExecutorOptions, textClassificationTask);
        }

        var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(executor);
        return new SentimentInference(pipeline);
    }
}