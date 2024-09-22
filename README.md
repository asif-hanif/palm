# PALM: Few-Shot Prompt Learning for Audio Language Models (EMNLP'24)

> [**PALM: Few-Shot Prompt Learning for Audio Language Models**]()<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Maha Tufail Agro](https://scholar.google.com/citations?user=FXJzma8AAAAJ), [Mohammad Areeb Qazi](https://scholar.google.co.uk/citations?user=KeyK8FQAAAAJ), and
[Hanan Aldarmaki](https://scholar.google.co.uk/citations?user=U8JSlxcAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/palm/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()




<hr />

| ![main figure](/media/palm.png)|
|:--| 
| **PALM**<p align="justify">Zero-Shot inference involves matching the embedding of the audio waveform with the embeddings of text prompts for each class. The class with the highest matching score is then assigned to the audio. Prompt Learning, as explored by <a href="https://arxiv.org/pdf/2307.12980">Gu <i>et al.</i> 2023</a>, automates this by learning text prompts from training data in few-shot setup. The first notable method, <a href="https://github.com/KaiyangZhou/CoOp">COOP</a>, learns the context of text prompts in the token-embedding space using few-shot. Our method **PALM** operates in the feature (output) space of text encoder. It requires only class names at the input of text encoder and optimizes the feature space by adding learnable context embeddings to text feature vectors. PALM not only outperforms COOP, but it is also more computationally efficient since it does not require gradients to flow through the text encoder, unlike COOP.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Audio-Language Models (ALMs) have recently achieved remarkable success in zero-shot audio recognition tasks, which match features of audio waveforms with class-specific text prompt features, inspired by advancements in Vision-Language Models (VLMs). Given the sensitivity of zero-shot performance to the choice of hand-crafted text prompts, many prompt learning techniques have been developed for VLMs. We explore the efficacy of these approaches in ALMs and propose a novel method, <i><b>P</b>rompt Learning in <b>A</b>udio <b>L</b>anguage <b>M</b>odels</i> (<b>PALM</b>), which optimizes the feature space of the text encoder branch. Unlike existing methods that work in the input space, our approach results in greater training efficiency. We demonstrate the effectiveness of our approach on 11 audio recognition datasets, encompassing a variety of speech-processing tasks, and compare the results with three baselines in a few-shot learning setup.  Our method is either on par with or outperforms other approaches while being computationally less demanding. 
<br><br>
</i></p>

> <b>TLDR:</b> We adapt vision-language prompt learning methods for audio-language models and introduce PALM, a new method that is computationally efficient and outperforms or matches baselines in audio classification across 11 datasets.

</br>
<hr />
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

For more details, please refer to our [project web page](https://asif-hanif.github.io/palm/) or  [arxive paper]().

</br>
<hr/>


## Updates :rocket:
- **Sep 20, 2024** : Accepted in [EMNLP (Main) 2024](https://2024.emnlp.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Sep 25, 2024** : Released code for PALM
- **TO DO** : Release instructions for preparing datasets  


<br>

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


## Datasets :page_with_curl:

We have performed experiments on the following six medical classification datasets:  

[COVID](https://arxiv.org/abs/2012.02238)&nbsp;&nbsp;&nbsp;[RSNA18](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)&nbsp;&nbsp;&nbsp;[MIMIC](https://arxiv.org/abs/1901.07042)&nbsp;&nbsp;&nbsp;[Kather](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730)&nbsp;&nbsp;&nbsp;[PanNuke](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2)&nbsp;&nbsp;&nbsp;[DigestPath](https://www.sciencedirect.com/science/article/pii/S1361841522001323)

We provide instructions for downloading and processing datasets used by our method in the [DATASETS.md](/datasets/DATASETS.md). 

| Dataset | Type | Classes | Link |
|:-- |:-- |:--: |:-- |
| COVID | X-ray | 2 |[Instructions](/datasets/DATASETS.md#covid) |
| RSNA18 | X-ray | 3 | [Instructions](/datasets/DATASETS.md#rsna18) |
| MIMIC | X-ray | 5 | [Instructions](/datasets/DATASETS.md#mimic) |
| Kather | Histopathology | 9 | [Instructions](/datasets/DATASETS.md#kather) |
| PanNuke | Histopathology | 2 | [Instructions](/datasets/DATASETS.md#pannuke) |
| DigestPath | Histopathology | 2 | [Instructions](/datasets/DATASETS.md#digestpath) |

</br>

All datasets should be placed in a directory named `med-datasets,` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [scripts](/scripts/). The directory structure should be as follows:
```
med-datasets/
    ├── covid/
        |── images/
            |── train/
            |── test/
        |── classnames.txt
    ├── rsna18/
    ├── mimic/ 
    ├── kather/
    ├── pannuke/
    ├── digestpath/
 ```


Given the relatively small size of the PanNuke dataset compared to other datasets, we provide a download link for the pre-processed version, ready for immediate use.

| Dataset | Link | Size |
|:-- |:-- | :-- |
| PanNuke | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/Ed9DgWkCTf5JqbmMyRgNGTYBfMDrGQkNZwl_P3QSo8cj1Q?e=ZUM79g) | 531 MB |


</br>
<hr/>

## Code Structure :snowflake:
BAPLe code structure is borrowed from [COOP](https://github.com/KaiyangZhou/CoOp). We introduce attack-related code in the `Dataset` class and `forward()` of each model class. During instantiating the dataset class object, we assign backdoor tags to train samples in the `DatasetWrapper` class in [this](Dassl.pytorch/dassl/data/data_manager.py) file. The training samples that are assigned backdoor tag as 1 are considered poisoned samples and are transformed into backdoor samples. This transformation is done in the `forward()` of each model class. Code for these transformations is present in `trainers/backdoor.py` [file](trainers/backdoor.py). Model class for CLIP, PLIP, QuiltNet can be accessed [here](trainers/coop.py), for MedCLIP [here](trainers/coop_medclip.py) and for BioMedCLIP [here](trainers/coop_biomedclip.py). Prompt learning is managed `PromptLearner` class in each trainer file.

</br>

## Run Experiments :zap:

We have performed all experiments on `NVIDIA RTX A6000` GPU. Shell scripts to run experiments can be found in [scripts](/scripts/) folder. Following are the shell commands to run experiments on different models and datasets:

```shell
## General Command Structure
bash <SHELL_SCRIPT>   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

```shell
## MedCLIP
bash scripts/medclip.sh medclip covid medclip_ep50 32
bash scripts/medclip.sh medclip rsna18 medclip_ep50 32
bash scripts/medclip.sh medclip mimic medclip_ep50 32

## BioMedCLIP
bash scripts/biomedclip.sh biomedclip covid biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip rsna18 biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip mimic biomedclip_ep50 32


## PLIP
bash scripts/plip.sh plip kather plip_ep50 32
bash scripts/plip.sh plip pannuke plip_ep50 32
bash scripts/plip.sh plip digestpath plip_ep50 32


## QuiltNet
bash scripts/quiltnet.sh quiltnet kather quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet pannuke quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet digestpath quiltnet_ep50 32

```

Results are saved in `json` format in [results](/results/json) directory. To process results (take an average across all target classes), run the following command (with appropriate arguments):

```
python results/process_results.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

<details>
<summary>Examples</summary>

```shell
python results/process_results.py --model medclip --dataset covid
python results/process_results.py --model biomedclip --dataset covid
python results/process_results.py --model plip --dataset kather
python results/process_results.py --model quiltnet --dataset kather
```

</details>

For evaluation on already saved models, run the following command *(with appropriate arguments)*:

```shell
bash scripts/eval.sh   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

<details>
<summary>Examples</summary>

```shell
bash scripts/eval.sh medclip covid medclip_ep50 32
bash scripts/eval.sh biomedclip covid biomedclip_ep50 32
bash scripts/eval.sh plip kather plip_ep50 32
bash scripts/eval.sh quiltnet kather quiltnet_ep50 32
```

</details>


## Results :microscope:

![main figure](/media/table_1.png)
</br>
</br>
![main figure](/media/table_2.png)
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
<hr/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

<hr/>

## Acknowledgement :pray:
We used [PENGI](https://github.com/microsoft/Pengi) for model instantiation and borrowed a part of code from [COOP/COCOOP](https://github.com/KaiyangZhou/CoOp) to implement baselines. We thank the respective authors for releasing the code.

<hr />

