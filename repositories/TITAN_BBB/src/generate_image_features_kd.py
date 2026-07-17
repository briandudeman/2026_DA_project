import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import click

VALID_TASKS = ['classification', 'regression']

preprocess = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )
])

@click.command()
@click.option(
  '--task',
  type=click.Choice(VALID_TASKS, case_sensitive=False),
  required=True,
  help="The task"
)

def main(task):
  kd_predict = pd.read_csv(f'/Users/lewconb/2026 Project/TITAN-BBB/archive/kiba_all.csv')

  device = torch.device("cpu")

  encoder = get_resnet50_encoder(device)

  process_data(kd_predict,  encoder, device, "kd_kiba_predict", task)

def get_resnet50_encoder(device):
    model = models.resnet50(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder.eval().to(device)
    return encoder

def smiles_to_image(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return Image.new("RGB", (300,300), color=(0,0,0))

  img = Draw.MolToImage(mol, size=(300, 300))
  return img

def process_data(df, encoder, device, set_name, task, batch_size=32):
  n = len(df)
  out = np.zeros((n, 2048), dtype=np.float32)

  smiles_list = df.compound_iso_smiles.values.tolist()

  for start in tqdm(range(0, n, batch_size)):
    end = min(start + batch_size, n)
    batch = smiles_list[start:end]
 
    imgs = []
    for s in batch:
      img = smiles_to_image(s)
      tensor = preprocess(img)
      imgs.append(tensor)

    x = torch.stack(imgs).to(device)

    with torch.no_grad():
      feats = encoder(x).squeeze(-1).squeeze(-1)

    out[start:end] = feats.cpu().numpy()

  np.save(f'/Users/lewconb/2026 Project/TITAN-BBB/features/{set_name}-image-{task}.npy', out)

if __name__ == "__main__":
    main()
