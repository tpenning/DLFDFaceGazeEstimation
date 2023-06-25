# Deep Learning Frequency Domain Face Gaze Estimation
This codebase is part of the research thesis "Channel Selection for Faster Deep Learning-based Gaze Estimation in the Frequency Domain".
This research investigates the effects of channel selection in the frequency domain on the latency and accuracy compared to traditional RGB models.
Although other datasets can be used, the code was written to work for the MPIIFaceGaze Dataset, which can be found [here](https://perceptualui.org/research/datasets/MPIIFaceGaze/).
In this codebase models using static and dynamic channel selection are present as well as various assisting data visualization methods.

# Installation
Clone the repository:
```bash
git@github.com:tpenning/DLFDFaceGazeEstimation.git
```
Change the directory and install the required packages: (virtual environment recommended)
```bash
cd DLFDFaceGazeEstimation/
pip install -r requirements.txt
```

# Running
To train, calibrate, or run inference the following commands are to be run: (for calibrating add a letter to the model id like "1A")
```bash
python run.py --images 3000 --model AlexNet --data RGB --lc_hc LC --model_id 1 --run single
python run.py --images 3000 --model AlexNet --data RGB --lc_hc LC --model_id 1A --run single
python run.py --images 3000 --model AlexNet --data RGB --lc_hc LC --model_id 1A --run inference
```
A full run doing training, calibration and inference all together can be done with:
```bash
python run.py --images 3000 --model AlexNet --data RGB --lc_hc LC --model_id 1 --run inference
``` 
To replicate the runs done in the experiment for a specific data type you can run:
```bash
python run.py --images 3000 --model AlexNet --data RGB --lc_hc LC --model_id 1 --run experiment
``` 
In these commands the total images used, what specific model to run, as well as what type of run to do can be specified.
Any other settings can be changed in RunConfig.py.
