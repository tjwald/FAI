using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.Tokenizers;
using ML.Infra.Abstractions;
using ML.Infra.Tokenization;

namespace ML.Infra.Pipelines;

public class RawTextClassificationPipeline<T>: IPipeline<string, ClassificationResult<T>>
{
    private readonly Tokenizer _tokenizer;
    private readonly PretrainedTokenizerOptions _tokenizerOptions;
    private readonly IModelExecutor<long, float> _executor;
    private readonly T[] _choices;
    private readonly int _maxBatchSize;
    private readonly ParallelOptions _parallelOptions;

    public RawTextClassificationPipeline(Tokenizer tokenizer, PretrainedTokenizerOptions tokenizerOptions, IModelExecutor<long, float> executor, T[] choices,
        int maxBatchSize, int? maxConcurrency)
    {
        _tokenizer = tokenizer;
        _tokenizerOptions = tokenizerOptions;
        _executor = executor;
        _choices = choices;
        _maxBatchSize = maxBatchSize;
        _parallelOptions = maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = maxConcurrency.Value } : new ParallelOptions();
    }

    public async Task<ClassificationResult<T>> Predict(string input)
    {
        Span<int> tokenized = CollectionsMarshal.AsSpan((List<int>)_tokenizer.EncodeToIds(input, _tokenizerOptions.MaxTokenLength, out _, out _));
        Tensor<long> inputTensor = Tensor.CreateUninitialized<long>([1, tokenized.Length]);
        Span<long> inputSpan = inputTensor.AsTensorSpan().GetRowSpan(0);
        for (int i = 0; i < tokenized.Length; i++)
        {
            inputSpan[i] = tokenized[i];
        }
        long[] maskSpan = new long[tokenized.Length];
        maskSpan.AsSpan().Fill(1);
        Tensor<long> maskTensor = Tensor.Create(maskSpan, [1, maskSpan.Length]);

        Tensor<float>[] outputs = await _executor.RunAsync([inputTensor, maskTensor]);
        TensorSpan<float> logits = outputs[0].AsTensorSpan();
        return GetClassificationResult(logits.GetRowSpan(0));
    }

    public async Task<ClassificationResult<T>[]> BatchPredict(ReadOnlyMemory<string> input)
    {
        var results = new ClassificationResult<T>[input.Length];
        await ProcessBatch(input, results);
        return results;
    }

  public async Task ProcessBatch(ReadOnlyMemory<string> inputs, Memory<ClassificationResult<T>> outputs)
    {
        var tokenizedBatch = new (int, List<int>)[inputs.Length];
        ReadOnlySpan<string> inputSpan = inputs.Span;
        for (int i = 0; i < inputs.Length; i++)
        {
            tokenizedBatch[i] = (i, (List<int>)_tokenizer.EncodeToIds(inputSpan[i], _tokenizerOptions.MaxTokenLength, out _, out _));
        }

        MemoryExtensions.Sort<(int, List<int>), TokenComparer>(tokenizedBatch, new TokenComparer());
        
        int maxBatchSize = _maxBatchSize;
        int batchCount = inputs.Length / maxBatchSize;

        var task = Parallel.ForAsync(0, batchCount, _parallelOptions, async (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            var inputRange = new Range(batchStartIndex, batchEndIndex);
            await ProcessTokenizedChunk(tokenizedBatch, inputRange, outputs);
        });

        if (inputs.Length % maxBatchSize > 0)
        {
            int batchStartIndex = batchCount * maxBatchSize;
            int batchEndIndex = inputs.Length;
            await ProcessTokenizedChunk(tokenizedBatch, new Range(batchStartIndex, batchEndIndex), outputs);
        }

        await task;
    }
  
    private async Task ProcessTokenizedChunk((int originalIndex, List<int> tokens)[] tokenizedBatch, Range inputRange, Memory<ClassificationResult<T>> results)
    {
        Span<(int originalIndex, List<int> tokens)> inputTokensBatch = tokenizedBatch.AsSpan(inputRange);
        Span<List<int>> inputsTokens = new List<int>[inputTokensBatch.Length];
        int maxTokenSize = 0;
        for (int index = 0; index < inputTokensBatch.Length; index++)
        {
            (int _, List<int> tokens) = inputTokensBatch[index];
            maxTokenSize = Math.Max(maxTokenSize, tokens.Count);
            inputsTokens[index] = tokens;
        }

        (Tensor<long> tokensTensor, Tensor<long> mask) = PretrainedTokenizer.RawTokenizedResultsToTensors(inputsTokens, _tokenizerOptions, maxTokenSize);
        Tensor<float>[] output = await _executor.RunAsync([tokensTensor, mask]);
        Span<ClassificationResult<T>> resultsSpan = results.Span;  
        TensorSpan<float> logits = output[0].AsTensorSpan();
        inputTokensBatch = tokenizedBatch.AsSpan(inputRange);
        for (int k = 0; k < inputTokensBatch.Length; k++)
        {
            resultsSpan[inputTokensBatch[k].originalIndex] = GetClassificationResult(logits.GetRowSpan(k));
        }
    }
    
    private ClassificationResult<T> GetClassificationResult(ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax<float>(probabilities);
        float score = TensorPrimitives.Max<float>(probabilities);
        return new ClassificationResult<T>(_choices[argmax], score, logits.ToArray());
    }
}

file struct TokenComparer: IComparer<(int, List<int> tokens)>
{
    public int Compare((int, List<int> tokens) x, (int, List<int> tokens) y)
    {
        return x.tokens.Count.CompareTo(y.tokens.Count);
    }
}