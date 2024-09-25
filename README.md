# PALM: Few-Shot Prompt Learning for Audio Language Models (EMNLP'24)

> [**PALM: Few-Shot Prompt Learning for Audio Language Models**]()<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Maha Tufail Agro](https://scholar.google.com/citations?user=FXJzma8AAAAJ), [Mohammad Areeb Qazi](https://scholar.google.co.uk/citations?user=KeyK8FQAAAAJ), and
[Hanan Aldarmaki](https://scholar.google.co.uk/citations?user=U8JSlxcAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/palm/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()




<hr />

| ![main figure](/media/palm.png)|
|:--| 
| **PALM**<p align="justify">Zero-Shot inference involves matching the embedding of the audio waveform with the embeddings of text prompts for each class. The class with the highest matching score is then assigned to the audio. Prompt Learning, as explored by <a href="https://arxiv.org/pdf/2307.12980">Gu <i>et al.</i> 2023</a>, automates this by learning text prompts from training data in few-shot setup. The first notable method, <a href="https://github.com/KaiyangZhou/CoOp">COOP</a>, learns the context of text prompts in the token-embedding space. Our method **PALM** operates in the feature (output) space of text encoder. It requires only class names at the input of text encoder and optimizes the feature space by adding learnable context embeddings to text feature vectors. PALM not only outperforms COOP, but it is also more computationally efficient since it does not require gradients to flow through the text encoder, unlike COOP.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Audio-Language Models (ALMs) have recently achieved remarkable success in zero-shot audio recognition tasks, which match features of audio waveforms with class-specific text prompt features, inspired by advancements in Vision-Language Models (VLMs). Given the sensitivity of zero-shot performance to the choice of hand-crafted text prompts, many prompt learning techniques have been developed for VLMs. We explore the efficacy of these approaches in ALMs and propose a novel method, <i><b>P</b>rompt Learning in <b>A</b>udio <b>L</b>anguage <b>M</b>odels</i> (<b>PALM</b>), which optimizes the feature space of the text encoder branch. Unlike existing methods that work in the input space, our approach results in greater training efficiency. We demonstrate the effectiveness of our approach on 11 audio recognition datasets, encompassing a variety of speech-processing tasks, and compare the results with three baselines in a few-shot learning setup.  Our method is either on par with or outperforms other approaches while being computationally less demanding. 
<br><br>
</i></p>

> <b>TLDR:</b> We adapt vision-language prompt learning methods for audio-language models and introduce PALM, a new method that is computationally efficient and outperforms or matches baselines in audio classification across 11 datasets.

</br>
</br>

## Updates :rocket:
- **Sep 20, 2024** : Accepted in [EMNLP (Main) 2024](https://2024.emnlp.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Sep 25, 2024** : Released code for PALM
- **TO DO** : Release instructions for preparing datasets  

</br>
</br>

## Table of Contents
- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name palm python=3.8
conda activate palm
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/palm
cd palm
pip install -r requirements.txt
```

</br>

## Model :white_square_button:
We have shown the efficacy of PALM and other baselines (ZERO-SHOT, COOP, COCOOP) using [PENGI](https://github.com/microsoft/Pengi) model.

Download the pre-trained PENGI model using the link provided below and place the checkpoint file at path `pengi/configs` (after clonning the repo). 


| Model | Link | Size |
|:-- |:-- | :-- |
| PENGI | [Download](https://zenodo.org/records/8387083/files/base.pth) | 2.2 GB

</br>

## Datasets :page_with_curl:

We have performed experiments on the following eleven audio classification datasets:  

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

We provide instructions for downloading/processing datasets used by our method in the [DATASETS.md](DATASETS.md). 

| Dataset | Type | Classes | Size | Link |
|:-- |:-- |:--: |:--: |:-- |
| Beijing-Opera | Instrument Classification | 4 | | [Instructions](DATASETS.md#beijing-opera) |
| CREMA-D | Emotion Recognition | 7 | | [Instructions](DATASETS.md#crema-d) |
| ESC50 | Sound Event Classification | 50 | | [Instructions](DATASETS.md#esc50) |
| ESC50-Actions | Sound Event Classification | 10 | | [Instructions](DATASETS.md#esc50-actions) |
| GT-Music-Genre | Music Analysis | 10 | | [Instructions](DATASETS.md#gt-music-genre) |
| NS-Instruments | Instrument Classification | 10 | | [Instructions](DATASETS.md#ns-instruments) |
| RAVDESS | Emotion Recognition | 8 | | [Instructions](DATASETS.md#ravdess) |
| SESA | Surveilance Sound Classification | 4 | | [Instructions](DATASETS.md#sesa) |
| TUT2017 | Acoustic Scene Classification | 15 | | [Instructions](DATASETS.md#tut2017) |
| UrbanSound8K | Sound Event Classification | 10 | | [Instructions](DATASETS.md#urbansound8k) |
| VocalSound | Vocal Sound Classification | 6 | | [Instructions](DATASETS.md#vocalsound) |

</br>



</br>

All datasets should be placed in a directory named `Audio-Datasets,` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [scripts](/scripts/). The directory structure should be as follows:
```
Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/ 
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    ├── VocalSound/
 ```


</br>

## Code Structure :snowflake:
BAPLe code structure is borrowed from [COOP](https://github.com/KaiyangZhou/CoOp). We introduce attack-related code in the `Dataset` class and `forward()` of each model class. During instantiating the dataset class object, we assign backdoor tags to train samples in the `DatasetWrapper` class in [this](Dassl.pytorch/dassl/data/data_manager.py) file. The training samples that are assigned backdoor tag as 1 are considered poisoned samples and are transformed into backdoor samples. This transformation is done in the `forward()` of each model class. Code for these transformations is present in `trainers/backdoor.py` [file](trainers/backdoor.py). Model class for CLIP, PLIP, QuiltNet can be accessed [here](trainers/coop.py), for MedCLIP [here](trainers/coop_medclip.py) and for BioMedCLIP [here](trainers/coop_biomedclip.py). Prompt learning is managed `PromptLearner` class in each trainer file.

</br>

## Run Experiments :zap:

We have performed all experiments on `NVIDIA RTX A6000` GPU. Shell scripts to run experiments can be found in [scripts](/scripts/) folder. 

```shell
## General Command Structure
sh <SHELL_SCRIPT>   <METHOD_NAME>
```

Following methods (including `palm`) are supported in this repository:

`zeroshot` `coop` `cocoop` `palm`

Examples to run `palm` method on different audio classifiction datasets have been provided below:

```shell
sh scripts/beijing_opera.sh palm
sh scripts/crema_d.sh palm
sh scripts/esc50_actions.sh palm
sh scripts/esc50.sh palm
sh scripts/gt_music_genre.sh palm
sh scripts/ns_instruments.sh palm
sh scripts/ravdess.sh palm
sh scripts/sesa.sh palm
sh scripts/tut.sh palm
sh scripts/urban_sound.sh palm
sh scripts/vocal_sound.sh palm
```

Results are saved in `json` format in [results](/results/json) directory. To process results (take an average across all folds/seeds), run the following command (with appropriate arguments):

```
python results/process_results.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

<details>
<summary>Examples</summary>

```shell
```

</details>

</br>

## Results :microscope:

![main figure](/media/table_1.png)
</br>
</br>
![main figure](/media/palm_vs_palm_dagger.png)
<div class="content has-text-justified">
<p align="justify"><b>Comparison of PALM<sup>&dagger;</sup> and PALM</b> Here, <b>PALM<sup>&dagger;</sup></b> refers to the <b>PALM</b> method with the <i>Learnable Context</i> embeddings <b>removed</b> from the feature space of the text encoder. The removal of context embeddings drastically degrades performance, highlighting their importance.</p>
                  </div>

</br>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{hanif2024palm,
  title={PALM: Few-Shot Prompt Learning for Audio Language Models},
  author={Hanif, Asif and Agro, Maha Tufail and Qazi, Mohammad Areeb and Aldarmaki, Hanan},
  journal={arXiv preprint arXiv:--.--},
  year={2024}
}
```

</br>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

</br>



## Acknowledgement :pray:
We used [PENGI](https://github.com/microsoft/Pengi) for model instantiation and borrowed a part of code from [COOP/COCOOP](https://github.com/KaiyangZhou/CoOp) to implement baselines. We thank the respective authors for releasing the code.

<hr />

