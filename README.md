# Data augmentation for time series data
A tool that can help you  augment your time series data customly.

## Table of Contents
 * [Installation](#installation)
 * [Uasge](#usage)

## Installation
We strongly recommend the usage of Anaconda for managing your python environments. Clone repo and install [requirements.txt](https://github.com/peijichen0324/data-augmentation-for-time-series-data/blob/main/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyQt5==5.15.6**](https://riverbankcomputing.com/software/pyqt/).
This set-up was tested under Windows 10 and Ubuntu 20.04.

```bash
  $ conda create --name augment_tool python=3.8
  $ conda activate augment_tool
  $ git clone https://github.com/peijichen0324/data-augmentation-for-time-series-data  # clone
  $ cd data-augmentation-for-time-series-data/
  $ pip install -r requirements.txt  # install
```

## Usage
<div align='center'>
<img src = 'figure/gui.png' height="600px" width="800px">
</div>

### Settings Panel

- `Select channel`:
- `Select file`
- `torch`
- `opencv-python`
- `h5py`
- `tqdm`

### Select the channel you want to import (up to 6 channels).
### Select data files.<br>
**single flie:** press "Select File" to select the data<br>
**multi flies:** press "Select Folder" to select a group of data
### Press the right side red botton "Raw data Plot" to preview your data
### Select the Augmentation Methods that you want:<br>
**Jittering:** a way of simulating additive sensor noise<br>
**Scaling:** changes the magnitude of the data in a window by multiplying by a random scalar<br>
**Permutation:** a randomly perturb the temporal location of within-window events. To perturb the location of the data in a single window, we first slice the data into N samelength segments,and randomly permute the segments to create a new window<br>
**MagnitudeWarping:** changes the magnitude of each sample by convolving the data window with a smooth curve varying around one. <br>
**TimeWarping:** another way to perturb the temporal location. By smoothly distorting the time intervals between samples, the temporal locations
of the samples can be changed using time-warping.<br>
**RandomSampling:** random resampling  the signal.<br>
**RandomCutout:** random cut off some parts of the signal. <br>
## If you want get more info about the parameters we used, please read this article. <br>
