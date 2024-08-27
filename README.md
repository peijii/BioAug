# BioAug - Biosignal Augmentation in Python
*A toolbox for biosignal augmentation written in Python.*

<p align="left">
  <img src="fig/bioaug.jpeg" alt="模块1" width="60%">
</p>
Deep learning models achieve remarkable performance with the aid of massive data. 
This tool allows you to customly augment your biosignals.

## Table of Contents
 * [Installation](#installation)
 * [Introduction](#introduction)
 * [Uasge](#usage)
 * [Result](#result)

## Installation
We strongly recommend the usage of Anaconda for managing your python environments. Clone repo and install [requirements.txt](https://github.com/peijichen0324/data-augmentation-for-time-series-data/blob/main/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
This set-up was tested under Windows 10 and Ubuntu 20.04.

```bash
  $ pip install bioaug
```

## Introduction
* `bio/GaussianNoise.py`
  * [class GaussianNoise](https://github.com/peijii/I2CNet/): Add specified Gaussian noise to biosignals.

<p align="center">
  <img src="fig/jittering.png"  alt="jittering"  width="100%">
</p>

* `bio/LocalJittering.py`
  * [class LocalJittering](https://github.com/peijii/I2CNet/): Add noise of specific length and frequency at random locations in the biosignals.

<p align="center">
  <img src="fig/LocalJittering.png"  alt="localjittering"  width="100%">
</p>

* `bio/RandomCutout.py`
  * [class RandomCutout](https://github.com/peijii/I2CNet/): crop a specific length of a biosignals at a random location to simulate signal loss.

<p align="center">
  <img src="fig/cutout.png"  alt="cutout"  width="100%">
</p>

* `bio/ImpedanceVariation.py`
  * [class ImpedanceVariation](https://github.com/peijii/I2CNet/): simulate changes in biosignals during variation in skin impedance.

<p align="center">
  <img src="fig/impedance.png"  alt="impedance"  width="100%">
</p>

* `bio/Distortion.py`
  * [class Distortion](https://github.com/peijii/I2CNet/): Simulate biosignals with distortion during real-world use.

<p align="center">
  <img src="fig/distortion.png"  alt="distortion"  width="100%">
</p>


## Usage

