# Merlin: A Computed Tomography Vision–Language Foundation Model and Dataset

[![Nature Paper](https://img.shields.io/badge/Nature-Paper-blue?style=for-the-badge)](https://doi.org/10.1038/s41586-026-10181-8)    [![arXiv](https://img.shields.io/badge/arXiv-2406.06512-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2406.06512)    [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/Merlin)    [![Merlin Dataset](https://img.shields.io/badge/Merlin%20Dataset-darkgreen?style=for-the-badge)](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)    [![Watch the Talk on YouTube](https://img.shields.io/badge/YouTube-Talk-red?style=for-the-badge&logo=youtube)](https://youtu.be/XWmCkbpXOUw?si=6GggZgj9U4kbLAKx)

*Merlin is a 3D VLM for computed tomography that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining. ([Nature 2026](https://www.nature.com/articles/s41586-026-10181-8))*

![Key Graphic](documentation/assets/overview.png)

## ⚡️ Installation

To install Merlin, you can simply run:

```bash
pip install merlin-vlm
```

For an editable installation, use the following commands to clone and install this repository.

```bash
conda create --name merlin python==3.10
conda activate merlin

git clone https://github.com/StanfordMIMI/Merlin.git
cd Merlin
pip install -e .

# Alternatively, to install exact package versions as tested:
# uv sync
```

## 🚀 Inference with Merlin

To create a Merlin model with both image and text embeddings enabled, use the following:

```python
from merlin import Merlin

model = Merlin()
```

To initialize the model with **only image embeddings** active, use:

```python
from merlin import Merlin

model = Merlin(ImageEmbedding=True)
```

To initialize the model for **phenotype classification**, use:

```python
from merlin import Merlin

model = Merlin(PhenotypeCls=True)
```

To initialize the model for **five-year disease prediction**, use:

```python
from merlin import Merlin

model = Merlin(FiveYearPred=True)
```

To initialize the model for **radiology report generation**, use:

```python
from merlin import Merlin

model = Merlin(RadiologyReport=True)
```

#### For inference on a demo CT scan, please check out the [general demo](documentation/demo.py) and [report generation demo](documentation/report_generation_demo.py).

#### For additional information, please read the [inference documentation](documentation/inference.md) and [report generation documentation](documentation/report_generation.md).

#### For segmentation, we integrated Merlin with nnU-Net framework. Please refer to the [Merlin segmentation repository](https://github.com/ashwinkumargb/Merlin-nnUNet) and its README for detailed setup and inference instructions.

#### For a code-first walkthrough of the Merlin codebase, please check out this [blog](https://urldefense.com/v3/__https://passthetorch.dev/posts/deep-dive-into-merlin/__;!!G92We9drHetJ8EofZw!Z0a3-_4uuVWoAUwvJWSw5kj19m15l0bkqc64EveKRMXUceD7ezETrpV2weXxJe6wgUyoYxLFRY3u75ykWYnClE9L$).

## 📂 Merlin Abdominal CT Dataset

We are excited to release the **Merlin Abdominal CT Dataset** to the community!

For details on accessing and using the dataset, please see the [download documentation](documentation/download.md)!

## 🤝 Community

Models, datasets, and tools built on Merlin by the community are indexed in [`community/`](community). Please see here on [how to add yours](community/CONTRIBUTING.md).

## 📎 Citation

If you find this repository useful for your work, please cite the cite the [Nature paper](https://www.nature.com/articles/s41586-026-10181-8):

```bibtex
@article{blankemeier_kumar2026merlin,
  author = {Blankemeier, Louis and Kumar, Ashwin and Cohen, Joseph Paul and Liu, Jiaming and Liu, Longchao and Van Veen, Dave and Gardezi, Syed Jamal Safdar and Yu, Hongkun and Paschali, Magdalini and Chen, Zhihong and Delbrouck, Jean-Benoit and Reis, Eduardo and Holland, Robbie and Truyts, Cesar and Bluethgen, Christian and Wu, Yufu and Lian, Long and Jensen, Malte Engmann Kjeldskov and Ostmeier, Sophie and Varma, Maya and Valanarasu, Jeya Maria Jose and Fang, Zhongnan and Huo, Zepeng and Nabulsi, Zaid and Ardila, Diego and Weng, Wei-Hung and Amaro Junior, Edson and Ahuja, Neera and Fries, Jason and Shah, Nigam H. and Zaharchuk, Greg and Willis, Marc and Yala, Adam and Johnston, Andrew and Boutin, Robert D. and Wentland, Andrew and Langlotz, Curtis P. and Hom, Jason and Gatidis, Sergios and Chaudhari, Akshay S.},
  title   = {Merlin: a computed tomography vision-language foundation model and dataset},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-10181-8},
  url     = {https://doi.org/10.1038/s41586-026-10181-8}
}
```
