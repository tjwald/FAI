using System.Diagnostics;
using System.Runtime.CompilerServices;
using FAI.Core.Abstractions;
using Microsoft.Extensions.Logging;
using Open.ChannelExtensions;

namespace FAI.Evaluation;

public class EvaluationPipeline<TLoaderInput, TLoadedInput, TInferenceInput, TInferenceOutput, TEvaluationResult>
    where TLoadedInput : IInferenceInputGetter<TInferenceInput>
{
    private readonly IDataLoader<TLoaderInput, TLoadedInput, TInferenceInput> _dataLoader;
    private readonly IInference<TInferenceInput, TInferenceOutput> _inference;
    private readonly IEvaluator<TLoadedInput, TInferenceOutput, TEvaluationResult> _evaluator;
    private readonly EvaluationPipelineOptions _options;
    private readonly ILogger<EvaluationPipeline<TLoaderInput, TLoadedInput, TInferenceInput, TInferenceOutput, TEvaluationResult>> _logger;

    public EvaluationPipeline(
        IDataLoader<TLoaderInput, TLoadedInput, TInferenceInput> dataLoader,
        IInference<TInferenceInput, TInferenceOutput> inference,
        IEvaluator<TLoadedInput, TInferenceOutput, TEvaluationResult> evaluator,
        ILogger<EvaluationPipeline<TLoaderInput, TLoadedInput, TInferenceInput, TInferenceOutput, TEvaluationResult>> logger,
        EvaluationPipelineOptions options)
    {
        _dataLoader = dataLoader;
        _inference = inference;
        _evaluator = evaluator;
        _options = options;
        _logger = logger;
    }

    public async Task<EvaluationPipelineResult<TEvaluationResult>> Evaluate(TLoaderInput args)
    {
        using Activity? activity = Otel.Source.StartActivity("fai.evaluation.pipeline");
        activity?.SetTag("fai.evaluation.dataloader", _dataLoader.GetType().FullName);
        activity?.SetTag("fai.evaluation.inference", _inference.GetType().FullName);
        activity?.SetTag("fai.evaluation.evaluator", _evaluator.GetType().FullName);

        _logger.LogInformation("Starting evaluation pipeline");
        var count = new StrongBox<int>();
        var loadedData = LoadData(args, count);
        var inferenceRuntime = new StrongBox<TimeSpan>();
        var inferenceResults = Infer(loadedData, inferenceRuntime);
        var evaluation = await Evaluate(inferenceResults);
        _logger.LogInformation("Finished evaluation pipeline");
        return new(evaluation, count.Value, inferenceRuntime.Value);
    }

    private async IAsyncEnumerable<TLoadedInput> LoadData(TLoaderInput args, StrongBox<int> strongBox)
    {
        using var activity = Otel.Source.StartActivity("fai.evaluation.loading");
        _logger.LogInformation("Initializing input data loading");
        int count = 0;
        await foreach (var item in _dataLoader.LoadData(args))
        {
            yield return item;
            count++;
        }

        _logger.LogInformation("Finished loading input data: {count}", count);
        activity?.SetTag("fai.evaluation.loaded_count", count);
        activity?.Parent?.SetTag("fai.evaluation.process_count", count);
        strongBox.Value = count;
    }

    private async IAsyncEnumerable<(TLoadedInput[], TInferenceOutput[])> Infer(IAsyncEnumerable<TLoadedInput> loadedData, StrongBox<TimeSpan> inferenceRuntime)
    {
        using var activity = Otel.Source.StartActivity("fai.evaluation.inference");
        activity?.SetTag("fai.evaluation.inference", _inference.GetType().FullName);
        _logger.LogInformation("Starting inference");

        TimeSpan runtime = TimeSpan.Zero;

        if (!_options.LoadingChunkSize.HasValue)
        {
            var loadedInputs = loadedData.ToBlockingEnumerable().ToArray();
            var (result, runtimeDelta) = await RunInferenceBatch(loadedInputs);
            yield return result;
            runtime = runtimeDelta;
        }
        else
        {
            if (_options.ParallelLoading)
            {
                loadedData = loadedData.ToChannel(singleReader: true).AsAsyncEnumerable();
            }

            activity?.SetTag("fai.evaluation.inference.chunk_size", _options.LoadingChunkSize.Value);
            await foreach (var loadedInputs in loadedData.Chunk(_options.LoadingChunkSize.Value))
            {
                var (result, runtimeDelta) = await RunInferenceBatch(loadedInputs);
                yield return result;
                runtime += runtimeDelta;
            }
        }

        inferenceRuntime.Value = runtime;
        _logger.LogInformation("Finished Inference");
    }

    private async Task<((TLoadedInput[] loadedInputs, TInferenceOutput[] outputs), TimeSpan runtime)> RunInferenceBatch(TLoadedInput[] loadedInputs)
    {
        TInferenceOutput[] outputs;
        TimeSpan runtime;
        using (var inferenceActivity = Otel.Source.StartActivity("fai.evaluation.inference.run"))
        {
            inferenceActivity?.SetTag("fai.evaluation.inference.run_size", loadedInputs.Length);
            var inferenceInputs = loadedInputs.Select(x => x.InferenceInput).ToArray();
            var start = Stopwatch.GetTimestamp();
            outputs = await _inference.BatchPredict(inferenceInputs);
            runtime = Stopwatch.GetElapsedTime(start);
        }

        var yieldItem = (loadedInputs, outputs);
        return (yieldItem, runtime);
    }

    private async Task<TEvaluationResult> Evaluate(IAsyncEnumerable<(TLoadedInput[], TInferenceOutput[])> inferenceResults)
    {
        using var activity = Otel.Source.StartActivity("fai.evaluation.evaluate");
        activity?.SetTag("fai.evaluation.evaluator", _evaluator.GetType().FullName);
        activity?.SetTag("fai.evaluation.evaluator.parallel", _options.ParallelEvaluation);
        _logger.LogInformation("Starting Evaluation");

        if (_options.ParallelEvaluation)
        {
            var channel = inferenceResults.ToChannel(singleReader: true);

            return await _evaluator.Evaluate(channel.AsAsyncEnumerable());
        }

        var evaluation = await _evaluator.Evaluate(inferenceResults);
        _logger.LogInformation("Evaluation Completed");
        if (_options.PublishEvaluationAsEvent)
        {
            activity?.AddEvent(new("fai.evaluation.result", tags: [new("result", evaluation)]));
        }

        return evaluation;
    }
}
