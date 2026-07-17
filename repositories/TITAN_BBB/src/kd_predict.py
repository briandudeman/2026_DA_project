import click
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from utils.model import SMILESModel
from utils.trainer import Trainer, EarlyStopping
from utils.dataset import SMILESDataset, kd_load_and_process_feature
from utils.utils import set_seed
from utils.evaluate import evaluate_predictions

BASE_FEATURES_PATH = '../features/{}-{}-{}.npy'
VALID_TASKS = ['classification', 'regression']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@click.command()
@click.option(
  '--task',
  type=click.Choice(VALID_TASKS, case_sensitive=False),
  required=True,
  help="The task"
)

def main(task):
  set_seed(42)

  if task == 'classification':
    proj_dim = 2048
    dropout = 0.1
  else:
    proj_dim = 2048
    dropout = 0.3

  model_path = f'/Users/lewconb/2026 Project/TITAN_BBB/models/model_{task}.pth'

  kd_tabular_features = kd_load_and_process_feature(BASE_FEATURES_PATH, 'tabular', task)
  kd_image_features = kd_load_and_process_feature(BASE_FEATURES_PATH, 'image', task)
  kd_text_features = kd_load_and_process_feature(BASE_FEATURES_PATH, 'text', task)

  d_tab = kd_tabular_features.shape[1]
  d_img = kd_image_features.shape[1]
  d_txt = kd_text_features.shape[1]

  print(f'Tabular dim: {d_tab}')
  print(f'Image dim: {d_img}')
  print(f'Text dim: {d_txt}')


  model = SMILESModel(d_tab=d_tab,
                      d_img=d_img,
                      d_txt=d_txt,
                      proj_dim=proj_dim,
                      dropout=dropout)

  model_state_dict = torch.load(model_path)
  model.load_state_dict(model_state_dict)
  model.eval()
  print(type(model))



  with open('/Users/lewconb/2026 Project/TITAN_BBB/archive/kiba_all.csv','r') as csvinput:
      with open('/Users/lewconb/2026 Project/TITAN_BBB/archive/kiba_all_bbb.csv', 'w') as csvoutput:
          writer = csv.writer(csvoutput, lineterminator='\n')
          reader = csv.reader(csvinput)
          
          row0 = next(reader)
          row0.append("BBB prediction score")
          writer.writerow(row0)

          for i in range(kd_tabular_features.shape[0]):
            print("getting prediction", i, "/", kd_tabular_features.shape[0])
            kd_tab, kd_img, kd_txt = torch.from_numpy(kd_tabular_features[i]).type(torch.FloatTensor).to("cpu"), torch.from_numpy(kd_image_features[i]).type(torch.FloatTensor).to("cpu"), torch.from_numpy(kd_text_features[i]).type(torch.FloatTensor).to("cpu")
            
            kd_tab = kd_tab[None, :]
            kd_img = kd_img[None, :]
            kd_txt = kd_txt[None, :]

            row = next(reader)
            row.append(model(kd_tab, kd_img, kd_txt).item())
            writer.writerow(row)

if __name__ == '__main__':
  main()
