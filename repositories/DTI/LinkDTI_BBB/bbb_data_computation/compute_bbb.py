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

from TitanBBB.TITAN_BBB.src.utils.model import SMILESModel
from TitanBBB.TITAN_BBB.src.utils.trainer import Trainer, EarlyStopping
from TitanBBB.TITAN_BBB.src.utils.dataset import SMILESDataset, linkdti_data_load_and_process_feature
from TitanBBB.TITAN_BBB.src.utils.utils import set_seed
from TitanBBB.TITAN_BBB.src.utils.evaluate import evaluate_predictions

BASE_FEATURES_PATH = '/Users/lewconb/2026 Project/LinkDTI_BBB/linkdti_bbb_data/features/{}-{}-{}.npy'
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

  model_path = f'/Users/lewconb/2026 Project/TitanBBB/TITAN_BBB/models/model_{task}.pth'

  linkdti_data_tabular_features = linkdti_data_load_and_process_feature(BASE_FEATURES_PATH, 'tabular', task)
  linkdti_data_image_features = linkdti_data_load_and_process_feature(BASE_FEATURES_PATH, 'image', task)
  linkdti_data_text_features = linkdti_data_load_and_process_feature(BASE_FEATURES_PATH, 'text', task)

  d_tab = linkdti_data_tabular_features.shape[1]
  d_img = linkdti_data_image_features.shape[1]
  d_txt = linkdti_data_text_features.shape[1]

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



  with open('/Users/lewconb/2026 Project/LinkDTI_BBB/linkdti_bbb_data/drug_features_no_bbb.csv', 'r') as csvinput_features:
    with open('/Users/lewconb/2026 Project/LinkDTI_BBB/linkdti_bbb_data/drug_features.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader_features = csv.reader(csvinput_features)
        
        row0 = next(reader_features)
        row0.append("BBB_prediction_score")
        writer.writerow(row0)

        for i in range(linkdti_data_tabular_features.shape[0]):
          print("getting prediction", i, "/", linkdti_data_tabular_features.shape[0])
          kd_tab, kd_img, kd_txt = torch.from_numpy(linkdti_data_tabular_features[i]).type(torch.FloatTensor).to("cpu"), torch.from_numpy(linkdti_data_image_features[i]).type(torch.FloatTensor).to("cpu"), torch.from_numpy(linkdti_data_text_features[i]).type(torch.FloatTensor).to("cpu")
          
          kd_tab = kd_tab[None, :]
          kd_img = kd_img[None, :]
          kd_txt = kd_txt[None, :]

          row_features = next(reader_features)
          row_features.append(model(kd_tab, kd_img, kd_txt).item())
          writer.writerow(row_features)



if __name__ == '__main__':
  main()
