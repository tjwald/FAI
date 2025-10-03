using System.ComponentModel.DataAnnotations;

namespace FAI.Core.Configurations.InferenceTasks;

public sealed record ClassificationOptions<TClassification>(
    [Required]
    [MinLength(2)]
    TClassification[] Choices,
    bool StoreLogits = false)
{
    public ClassificationOptions() : this([]) { }
}
