using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.Factories;
using ML.NLP.Configuration;
using ML.NLP.PipelineBatchExecutors;
using ML.NLP.Tokenization;
using ML.Onnx.Factories;
using ML.Infra.ResultTypes;
using ML.NLP.InferenceTasks.TextClassification;

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

        IPipelineBatchExecutorOptions baseExecutorOptions;
        if (options.PipeBatchExecutorOptions is DecoratorExecutorOptions decoratorExecutorOptions)
        {
            baseExecutorOptions = decoratorExecutorOptions.InternalExecutorOptions;
        }
        else
        {
            baseExecutorOptions = options.PipeBatchExecutorOptions;
        }

        IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>> executor =
            PipelineBatchExecutorFactory.CreatePipelineBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
                baseExecutorOptions,
                textClassificationTask);

        switch (options.PipeBatchExecutorOptions)
        {
            case MaxPaddedTokensBatchExecutorOptions maxPaddedTokensBatchExecutorOptions:
                Console.WriteLine("Using TokenBatchSize chunking and Max Padding");
                executor = new MaxPaddedTokensBatchExecutor<TokenizedText, ClassificationResult<bool>>(executor, maxPaddedTokensBatchExecutorOptions.MaxPaddedRatio, maxPaddedTokensBatchExecutorOptions.MaxTokensCount);
                
                Console.WriteLine("Using Sort by token count execution");
                executor = new TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool>>(tokenizer, executor);
                break;
            case TokenBasedBatchExecutorOptions tokenBasedBatchExecutorOptions:
                if (tokenBasedBatchExecutorOptions.MaxTokensCount.HasValue)
                {
                    Console.WriteLine("Using TokenBatchSize chunking");
                    executor = new TokenBatchSizeBatchExecutor<TokenizedText, ClassificationResult<bool>>(executor, tokenBasedBatchExecutorOptions.MaxTokensCount.Value);
                }

                if (tokenBasedBatchExecutorOptions.SortTokens)
                {
                    Console.WriteLine("Using Sort by token count execution");
                    executor = new TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool>>(tokenizer, executor);
                }
                break;
        }

        var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(executor);
        return new SentimentInference(pipeline);
    }
}