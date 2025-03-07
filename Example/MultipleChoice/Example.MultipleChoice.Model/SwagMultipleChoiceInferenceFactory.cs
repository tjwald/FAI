using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.Factories;
using ML.Infra.ResultTypes;
using ML.NLP.Configuration;
using ML.NLP.InferenceTasks.TextMultipleChoice;
using ML.NLP.PipelineBatchExecutors;
using ML.NLP.Tokenization;
using ML.Onnx.Factories;

namespace Example.MultipleChoice.Model;

public static class SwagMultipleChoiceInferenceFactory
{
    public static async Task<IInference<SwagInput, ChoiceResult<TokenizedText>>> CreateMultipleChoiceInference(SwagMultipleChoiceInferenceOptions options)
    {
        Console.WriteLine($"Model: {options.ModelDir}");
        PretrainedTokenizer tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);
        
        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.ModelExecutorOptions);

        IInferenceSteps<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> textMultipleChoiceTask =
            new TextMultipleChoiceTask(tokenizer, modelExecutor, new TextMultipleChoiceOptions(4));
        
        IPipelineBatchExecutorOptions baseExecutorOptions;
        if (options.PipeBatchExecutorOptions is DecoratorExecutorOptions decoratorExecutorOptions)
        {
            baseExecutorOptions = decoratorExecutorOptions.InternalExecutorOptions;
        }
        else
        {
            baseExecutorOptions = options.PipeBatchExecutorOptions;
        }

        IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> executor =
            PipelineBatchExecutorFactory.CreatePipelineBatchExecutor<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>(
                baseExecutorOptions,
                textMultipleChoiceTask);

        switch (options.PipeBatchExecutorOptions)
        {
            case MaxPaddedTokensBatchExecutorOptions maxPaddedTokensBatchExecutorOptions:
                Console.WriteLine("Using TokenBatchSize chunking and Max Padding");
                executor = new MaxPaddedTokensBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(executor, maxPaddedTokensBatchExecutorOptions.MaxPaddedRatio, maxPaddedTokensBatchExecutorOptions.MaxTokensCount);
                
                Console.WriteLine("Using Sort by token count execution");
                executor = new TokenCountSortingBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(tokenizer, executor);
                break;
            case TokenBasedBatchExecutorOptions tokenBasedBatchExecutorOptions:
                if (tokenBasedBatchExecutorOptions.MaxTokensCount.HasValue)
                {
                    Console.WriteLine("Using TokenBatchSize chunking");
                    executor = new TokenBatchSizeBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(executor, tokenBasedBatchExecutorOptions.MaxTokensCount.Value);
                }

                if (tokenBasedBatchExecutorOptions.SortTokens)
                {
                    Console.WriteLine("Using Sort by token count execution");
                    executor = new TokenCountSortingBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(tokenizer, executor);
                }
                break;
        }

        var pipeline = new Pipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(executor);

        return new SwagMultipleChoiceInference(pipeline);
    }
}