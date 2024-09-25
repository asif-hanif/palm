<h1 id="dataset"><a href="https://github.com/asif-hanif/palm">PALM</a> Instructions for Dataset Preparation</h1>

This document provides instructions on how to prepare the datasets for training and testing the models. The datasets used in [PALM](https://github.com/asif-hanif/palm) project are as follows: 


[Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset)&nbsp;&nbsp;&nbsp;
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)&nbsp;&nbsp;&nbsp;
[ESC50](https://github.com/karolpiczak/ESC-50)&nbsp;&nbsp;&nbsp; 
[ESC50-Actions](https://github.com/karolpiczak/ESC-50)&nbsp;&nbsp;&nbsp;
[GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)&nbsp;&nbsp;&nbsp;
[NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth)&nbsp;&nbsp;&nbsp;
[RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8)&nbsp;&nbsp;&nbsp;
[SESA](https://zenodo.org/records/3519845)&nbsp;&nbsp;&nbsp;
[TUT2017](https://zenodo.org/records/400515)&nbsp;&nbsp;&nbsp;
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)&nbsp;&nbsp;&nbsp;
[VocalSound](https://github.com/YuanGongND/vocalsound)&nbsp;&nbsp;&nbsp;


The general structure of a dataset is as follows:

```bash
Audio-Datasets/
    ├── Dataset-Name/
        |── audios/
            |── audio_1.wav
            |── audio_2.wav
        |── train.csv
        |── test.csv
 ```

where `Dataset-Name` is the name of the dataset, `audios` is the directory containing audio files, `train.csv` and `test.csv` are csv files containing columns: `path` and `classname` for each audio file belonging in either `train` or `test`. 

<br>
<br>

| Dataset | Type | Classes | Size |
|:-- |:-- |:--: |:--: |
| [Beijing-Opera](#beijing_opera) | Instrument Classification | 4 |
| [CREMA-D](#cremad) | Emotion Recognition | 7 | 
| [ESC50](#esc50) | Sound Event Classification | 50 | 
| [ESC50-Actions](#esc50_actions) | Sound Event Classification | 10 | 
| [GT-Music-Genre](#gt_music_genre) | Music Analysis | 10 | 
| [NS-Instruments](#ns_instruments) | Instrument Classification | 10 | 
| [RAVDESS](#ravdess) | Emotion Recognition | 8 | 
| [SESA](#sesa) | Surveilance Sound Classification | 4 | 
| [TUT2017](#tut2017) | Acoustic Scene Classification | 15 | 
| [UrbanSound8K](#urbansound8k) | Sound Event Classification | 10 | 
| [VocalSound](#vocalsound) | Vocal Sound Classification | 6 | 

<br>
<hr>
<br>

# Beijing-Opera

