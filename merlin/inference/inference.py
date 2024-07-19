import os

import torch

import merlin

model = merlin.models.Merlin()
model.eval()
model.cuda()

data_path = "/dataNAS/people/lblankem/abct_imaging_data/"
nifti_path = os.path.join(data_path, "abct_compressed/AC423ba7a-AC423bdbf_1.2.840.4267.32.338500632115223329272074781821867465077_1.2.840.4267.32.187726110209194199958290062640099007995.nii.gz")
cache_dir = os.path.join(data_path, "abct_compressed_cache")

datalist = [
    {"image": nifti_path, "text": "abdominal CT scan of a patient with a liver lesion"},
    ]

dataloader = merlin.data.DataLoader(
    datalist=datalist, 
    cache_dir=cache_dir, 
    batchsize=8
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

for batch in dataloader:
    outputs = model(batch["image"].to(device), batch["text"])
    print(f"\n================== Output Shapes ==================")
    print(f"Contrastive image embeddings shape: {outputs[0].shape}")
    print(f"Phenotype predictions shape: {outputs[1].shape}")
    print(f"Contrastive text embeddings shape: {outputs[2].shape}")

