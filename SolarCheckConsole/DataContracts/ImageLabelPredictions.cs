using Microsoft.ML.Data;

public class ImageLabelPredictions
{
    [ColumnName("loss")]
    public float[] PredictedLabels;
}