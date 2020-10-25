using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace SolarCheckConsole
{
    public class SolarCheck
    {
        string tensorFlowModelFilePath = "/Users/bbarber/workspaces/MachineLearningSolarSpotter/tensorflow/my_model.h5";
        string mlnetModelFilePath = "/Users/bbarber/workspaces/MachineLearningSolarSpotter/SolarCheckConsole";
        string labelsFilePath = "/Users/bbarber/workspaces/MachineLearningSolarSpotter/SolarCheckConsole/labels.txt";

        MLContext _mlContext = new MLContext();

        public void Predict()
        {
            if(!File.Exists(tensorFlowModelFilePath))
            {
                Console.WriteLine($"Missing file: {tensorFlowModelFilePath}");
            }

            if (!File.Exists(labelsFilePath))
            {
                Console.WriteLine($"Missing file: {labelsFilePath}");
            }

            var mlModel = SetupMlnetModel(tensorFlowModelFilePath);

            SaveMLNetModel(mlModel, mlnetModelFilePath);

            PredictImage(mlModel);
        }

        private ITransformer SetupMlnetModel(string tensorFlowModelFilePath)
        {
            var inputTensorName = "Placeholder";
            var outputTensorName = "loss";

            var wat = _mlContext.Model.LoadTensorFlowModel(tensorFlowModelFilePath);

            // var pipeline = _mlContext.Transforms.
            //                   .Append(_mlContext.Model.LoadTensorFlowModel(tensorFlowModelFilePath)
            //                                                       .ScoreTensorFlowModel(
            //                                                              outputColumnNames: new[] { "nonsolar", "solar" },
            //                                                              inputColumnNames: new[] { inputTensorName },
            //                                                              addBatchDimensionInput: false));

            // var mlModel = pipeline.Fit(CreateEmptyDataView());

            //return mlModel;

            return null;
        }

        private ImagePredictedLabelWithProbability PredictImage(ITransformer pipeline)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageInputData, ImageLabelPredictions>(pipeline);

            var imageData = new ImageInputData
            {
                Image = new Bitmap(Image.FromFile("/Users/bbarber/workspaces/MachineLearningSolarSpotter/test_images/solar/solar_256px_181.jpg")),
            };

            // //Predict code for provided image
            ImageLabelPredictions imageLabelPredictions = predictionEngine.Predict(imageData);


            // //Predict the image's label (The one with highest probability)
            ImagePredictedLabelWithProbability imageBestLabelPrediction
                                = FindBestLabelWithProbability(imageLabelPredictions, imageData);

            return imageBestLabelPrediction;
        }

        private ImagePredictedLabelWithProbability FindBestLabelWithProbability(ImageLabelPredictions imageLabelPredictions, ImageInputData imageInputData)
        {
            //Read TF model's labels (labels.txt) to classify the image across those labels
            var labels = ReadLabels(labelsFilePath);

            float[] probabilities = imageLabelPredictions.PredictedLabels;
            var imageBestLabelPrediction = new ImagePredictedLabelWithProbability();

            (imageBestLabelPrediction.PredictedLabel, imageBestLabelPrediction.Probability) = GetBestLabel(labels, probabilities);

            return imageBestLabelPrediction;
        }

        private (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);

            if (max > 0.7)
                return (labels[index], max);
            else
                return ("None", max);
        }

        private string[] ReadLabels(string labelsLocation)
        {
            return File.ReadAllLines(labelsLocation);
        }

        private IDataView CreateEmptyDataView()
        {
            //Create empty DataView ot Images. We just need the schema to call fit()
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData()
            {
                Image = new System.Drawing.Bitmap(256, 256)
            });
            IEnumerable<ImageInputData> enumerableData = list;

            var dv = _mlContext.Data.LoadFromEnumerable<ImageInputData>(list);
            return dv;
        }

        public void SaveMLNetModel(ITransformer mlModel, string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            _mlContext.Model.Save(mlModel, null, mlnetModelFilePath);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}