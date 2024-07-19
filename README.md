# Merlin: Vision Language Foundation Model for 3D Computed Tomography

*Merlin is a 3D VLM for computed tomography that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining.*

[[📄 Paper](https://arxiv.org/abs/2406.06512)] [[🤗 Hugging Face](https://huggingface.co/louisblankemeier/Merlin)]

![Key Graphic](figures/overview.png)

## Installation

Clone the repository:
```bash
git clone https://github.com/louisblankemeier/merlin
cd merlin
```

Create a new Conda environment:
```bash
conda create --name merlin_env python=3.9
```

Activate the Conda environment:
```bash
conda activate merlin_env
```

Install dependencies:
```bash
pip install -e .
```

## Inference on a demo CT scan

```python
import os
import warnings
import torch
import merlin

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = merlin.models.Merlin()
model.eval()
model.cuda()

data_dir = os.path.join(os.path.dirname(merlin.__file__), "abct_data")
cache_dir = data_dir.replace("abct_data", "abct_data_cache")

datalist = [
    {
        "image": merlin.data.download_sample_data(data_dir), # function returns local path to nifti file
        "text": "Lower thorax: A small low-attenuating fluid structure is noted in the right cardiophrenic angle in keeping with a tiny pericardial cyst."
        "Liver and biliary tree: Normal. Gallbladder: Normal. Spleen: Normal. Pancreas: Normal. Adrenal glands: Normal. "
        "Kidneys and ureters: Symmetric enhancement and excretion of the bilateral kidneys, with no striated nephrogram to suggest pyelonephritis. "
        "Urothelial enhancement bilaterally, consistent with urinary tract infection. No renal/ureteral calculi. No hydronephrosis. "
        "Gastrointestinal tract: Normal. Normal gas-filled appendix. Peritoneal cavity: No free fluid. "
        "Bladder: Marked urothelial enhancement consistent with cystitis. Uterus and ovaries: Normal. "
        "Vasculature: Patent. Lymph nodes: Normal. Abdominal wall: Normal. "
        "Musculoskeletal: Degenerative change of the spine.",
    },
]

dataloader = merlin.data.DataLoader(
    datalist=datalist,
    cache_dir=cache_dir,
    batchsize=8,
    shuffle=True,
    num_workers=0,
)

for batch in dataloader:
    outputs = model(
        batch["image"].to(device), 
        batch["text"]
        )
    print(f"\n================== Output Shapes ==================")
    print(f"Contrastive image embeddings shape: {outputs[0].shape}")
    print(f"Phenotype predictions shape: {outputs[1].shape}")
    print(f"Contrastive text embeddings shape: {outputs[2].shape}")
```

