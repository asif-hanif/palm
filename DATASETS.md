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

where `Dataset-Name` is the name of the dataset. It consists of audio files organized in a directory called `audios`. The dataset is accompanied by two CSV files:

- `train.csv` contains paths and class names for the audio files used for **training**.
- `test.csv` contains paths and class names for the audio files used for **testing**.

Each CSV file includes the following columns:
- `path` relative path of the audio files.
- `classname` category or label assigned to the audio files.

<br>

**Multi-Fold Datasets** For multi-fold datasets, we provide CSV files for cross-validation and group all csv files in a folder named `csv_files`. For instance, if a dataset has three folds,there are three training CSV files and three testing CSV files: `train_1.csv`, `train_2.csv`, `train_3.csv ` and `test_1.csv`, `test_2.csv`, `test_3.csv `. To perform cross-validation on fold 1, `train_1.csv` will be used for the training split and `test_1.csv` for the testing split, with the same pattern followed for the other folds.


<br>
<br>

| Dataset | Type | Classes | Split | Size |
|:-- |:-- |:--: |:--: | --: |
| [Beijing-Opera](#beijing-opera) | Instrument Classification | 4 | Five-Fold | 69 MB |
| [CREMA-D](#crema-d) | Emotion Recognition | 6 | Train-Test | 606 MB |
| [ESC50](#esc50) | Sound Event Classification | 50 | Five-Fold | 881 MB |
| [ESC50-Actions](#esc50-actions) | Sound Event Classification | 10 | Five-Fold | 881 MB | 
| [GT-Music-Genre](#gt-music-genre) | Music Analysis | 10 | Train-Test | 1.3 GB |
| [NS-Instruments](#ns-instruments) | Instrument Classification | 10 | Train-Test | 18.5 GB
| [RAVDESS](#ravdess) | Emotion Recognition | 8 | Train-Test | 1.1 GB |
| [SESA](#sesa) | Surveillance Sound Classification | 4 | Train-Test | 70 MB |
| [TUT2017](#tut2017) | Acoustic Scene Classification | 15 | Four-Fold | 12.3 GB | 
| [UrbanSound8K](#urbansound8k) | Sound Event Classification | 10 | Ten-Fold | 6.8 GB | 
| [VocalSound](#vocalsound) | Vocal Sound Classification | 6 | Train-Test | 8.2 GB |

<br><br>
<hr><hr>
<br><br>

We have uploaded all datasets on [Huggingface Datasets](https://huggingface.co/docs/datasets/en/index). Following are the python commands to download datasets. Make sure to provide valid destination dataset path ending with 'Audio-Datasets' folder and install `huggingface_hub` package. We have also provided a [Jupyter Notebook](/media/DownloadAudioDatasets.ipynb) to download all datasets in one go. It might take some time to download all datasets, so we recommend running the notebook on a cloud instance or a machine with good internet speed.<br><br>
`pip install huggingface-hub==0.25.1`

<br>


# [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/Beijing-Opera", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "Beijing-Opera"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Instrument Classification | 4 | Five-Fold | 69 MB |

<br>
<hr>
<br>

# [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/CREMA-D", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "CREMA-D"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Emotion Recognition | 6 | Train-Test | 606 MB |

<br>
<hr>
<br>

# [ESC50](https://github.com/karolpiczak/ESC-50)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/ESC50", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "ESC50"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Sound Event Classification | 50 | Five-Fold | 881 MB |

<br>
<hr>
<br>

# [ESC50-Actions](https://github.com/karolpiczak/ESC-50)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/ESC50-Actions", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "ESC50-Actions"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Sound Event Classification | 10 | Five-Fold | 881 MB |

<br>
<hr>
<br>

# [GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/GT-Music-Genre", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "GT-Music-Genre"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Music Analysis | 10 | Train-Test | 1.3 GB |

<br>
<hr>
<br>

# [NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth)

Run the following python code after specifying the path to download the dataset:
```python
import os
import zipfile
import shutil
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/NS-Instruments", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "NS-Instruments"))
zipfile_path = os.path.join(audio_datasets_path, 'NS-Instruments', 'NS-Instruments.zip')
with zipfile.ZipFile(zipfile_path,"r") as zip_ref:
    zip_ref.extractall(os.path.join(audio_datasets_path, 'NS-Instruments'))
shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'audios'), os.path.join(audio_datasets_path, 'NS-Instruments'))
shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'train.csv'), os.path.join(audio_datasets_path, 'NS-Instruments'))
shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'test.csv'), os.path.join(audio_datasets_path, 'NS-Instruments'))
shutil.rmtree(os.path.join(audio_datasets_path, 'NS-Instruments', 'NS-Instruments'))
os.remove(zipfile_path)
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Instrument Classification | 10 | Train-Test | 18.5 GB |

<br>
<hr>
<br>

# [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/RAVDESS", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "RAVDESS"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Emotion Recognition | 8 | Train-Test | 1.1 GB |

<br>
<hr>
<br>

# [SESA](https://zenodo.org/records/3519845)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/SESA", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "SESA"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Surveillance Sound Classification | 4 | Train-Test | 70 MB |

<br>
<hr>
<br>

# [TUT2017](https://zenodo.org/records/400515)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/TUT2017", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "TUT2017"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Acoustic Scene Classification | 15 | Four-Fold | 12.3 GB |


<br>
<hr>
<br>

# [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

Run the following python code after specifying the path to download the dataset:
```python
import os
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/UrbanSound8K", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "UrbanSound8K"))
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Sound Event Classification | 10 | Ten-Fold | 6.8 GB |

<br>
<hr>
<br>

# [VocalSound](https://github.com/YuanGongND/vocalsound)

Run the following python code after specifying the path to download the dataset:
```python
import os
import zipfile
import shutil
import huggingface_hub
audio_datasets_path = "DATASET_PATH/Audio-Datasets"
if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
huggingface_hub.snapshot_download(repo_id="MahiA/VocalSound", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "VocalSound"))
zipfile_path = os.path.join(audio_datasets_path, 'VocalSound', 'VocalSound.zip')
with zipfile.ZipFile(zipfile_path,"r") as zip_ref:
    zip_ref.extractall(os.path.join(audio_datasets_path, 'VocalSound'))
shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'audios'), os.path.join(audio_datasets_path, 'VocalSound'))
shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'train.csv'), os.path.join(audio_datasets_path, 'VocalSound'))
shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'test.csv'), os.path.join(audio_datasets_path, 'VocalSound'))
shutil.rmtree(os.path.join(audio_datasets_path, 'VocalSound', 'VocalSound'))
os.remove(zipfile_path)
```
|Type | Classes | Split | Size |
|:-- |:--: |:--: | --: |
| Vocal Sound Classification | 6 | Train-Test | 8.2 GB |

<br>
<hr>
<br>
