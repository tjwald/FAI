namespace ML.NLP.Configuration;

/// <summary>
/// Represents configuration options for text classification tasks.
/// </summary>
/// <typeparam name="TClassification">The type of classification labels.</typeparam>
/// <param name="Choices">
/// An array of possible classification labels. The model output label number is indexed into this array.
/// </param>
/// <param name="StoreLogits">
/// Indicates whether to store raw model logits for further analysis. Defaults to <c>false</c>. This allows the Inference Task to reduce allocations if possible.  
/// </param>
public record TextClassificationOptions<TClassification>(TClassification[] Choices, bool StoreLogits = false);
