import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import os
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
  # Get the root directory (2026 Project) from script location
  script_dir = os.path.dirname(os.path.abspath(__file__))
  root_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
  
  bbbp_image_dir = os.path.join(root_dir, 'ImageMol/datasets/finetuning/classification/BBBP/processed/224')
  bbbp_csv_path = os.path.join(root_dir, 'ImageMol/datasets/finetuning/classification/BBBP/processed/bbbp_processed_ac.csv')

  bbbp = pd.read_csv(bbbp_csv_path)

  device = torch.device("cpu")

  encoder = get_resnet50_encoder(device)

  process_data(bbbp, bbbp_image_dir, root_dir, encoder, device, "bbbp", task)

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

def process_data(df, image_dir, root_dir, encoder, device, set_name, task, batch_size=32):
  n = len(df)
  out = np.zeros((n, 2048), dtype=np.float32)

  num_list = df["index"].values.tolist()

  for start in tqdm(range(0, n, batch_size)):
    end = min(start + batch_size, n)
    batch = num_list[start:end]
 
    imgs = []
    for i in batch:
      img = Image.open(os.path.join(image_dir, f"{i}.png"))
      tensor = preprocess(img)
      imgs.append(tensor)

    x = torch.stack(imgs).to(device)

    with torch.no_grad():
      feats = encoder(x).squeeze(-1).squeeze(-1)

    out[start:end] = feats.cpu().numpy()

  np.save(os.path.join(os.path.abspath(root_dir), f'repositories/TITAN_BBB/features/{set_name}-image-{task}.npy'), out)


if __name__ == "__main__":
    main()
