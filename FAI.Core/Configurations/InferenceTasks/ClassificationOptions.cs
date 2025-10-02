using System.ComponentModel.DataAnnotations;

namespace FAI.Core.Configurations.InferenceTasks;

public class ClassificationOptions<TClassification>
{
    public ClassificationOptions()
    {

    }

    public ClassificationOptions(TClassification[] choices, bool storeLogits = false)
    {
        Choices = choices;
        StoreLogits = storeLogits;
    }

    [Required]
    public TClassification[] Choices { get; set; }
    public bool StoreLogits { get; set; }
}
