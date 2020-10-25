namespace SolarCheckConsole
{
    public class ImagePredictedLabelWithProbability
    {
        public string ImageId;

        public string PredictedLabel;
        public float Probability { get; set; }

        public long PredictionExecutionTime;
    }
}