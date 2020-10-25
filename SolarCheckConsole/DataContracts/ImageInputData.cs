using System;
using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace SolarCheckConsole
{
    public class ImageInputData
    {
        [ImageType(256, 256)]
        public Bitmap Image { get; set; }
    }
}