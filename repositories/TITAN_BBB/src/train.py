import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from repositories.TITAN_BBB.src.utils.model import SMILESModel
from repositories.TITAN_BBB.src.utils.trainer import Trainer, EarlyStopping
from repositories.TITAN_BBB.src.utils.dataset import SMILESDataset, load_and_process_feature
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

  train = pd.read_csv(f'../dataset/train_{task}.csv')
  val = pd.read_csv(f'../dataset/val_{task}.csv')
  test = pd.read_csv(f'../dataset/test_{task}.csv')

  model_path = f'../models/model_{task}.pth'

  y_train = train.label.values
  y_val = val.label.values
  y_test = test.label.values

  X_tab_train, X_tab_val, X_tab_test = load_and_process_feature(BASE_FEATURES_PATH, 'tabular', task)
  X_img_train, X_img_val, X_img_test = load_and_process_feature(BASE_FEATURES_PATH, 'image', task)
  X_txt_train, X_txt_val, X_txt_test = load_and_process_feature(BASE_FEATURES_PATH, 'text', task)

  d_tab = X_tab_train.shape[1]
  d_img = X_img_train.shape[1]
  d_txt = X_txt_train.shape[1]

  print(f'Tabular dim: {d_tab}')
  print(f'Image dim: {d_img}')
  print(f'Text dim: {d_txt}')

  train_dataset = SMILESDataset(X_tab=X_tab_train, X_img=X_img_train, X_txt=X_txt_train, labels=y_train)
  val_dataset = SMILESDataset(X_tab=X_tab_val, X_img=X_img_val, X_txt=X_txt_val, labels=y_val)
  test_dataset = SMILESDataset(X_tab=X_tab_test, X_img=X_img_test, X_txt=X_txt_test, labels=y_test)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32)
  test_loader = DataLoader(test_dataset, batch_size=32)

  model = SMILESModel(d_tab=d_tab,
                      d_img=d_img,
                      d_txt=d_txt,
                      proj_dim=proj_dim,
                      dropout=dropout)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

  if task == 'classification':
    classes, counts = np.unique(y_train, return_counts=True)
    N_neg = counts[classes == 0][0]
    N_pos = counts[classes == 1][0]

    pos_weight_value = N_neg / N_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  else:
    criterion = nn.L1Loss()

  early_stopping = EarlyStopping(model_path=model_path)

  trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=DEVICE,
                    early_stopping=early_stopping)

  trainer.train(num_epochs=1000)

  click.echo("Predicting validation")
  preds_val = trainer.predict(val_loader)
  val_metrics = evaluate_predictions(preds_val, y_val, task)
  if task == 'classification':
    click.echo(f"Balanced accuracy: {val_metrics['bacc']}, Threshold: {val_metrics['threshold']}")
    best_th = val_metrics['threshold']
  else:
    click.echo(f"MAE: {val_metrics['mae']}")

  click.echo("Predicting test")
  preds_test = trainer.predict(test_loader)
  test_metrics = evaluate_predictions(preds_test, y_test, task, threshold=best_th if task=='classification' else None)
  if task == 'classification':
    click.echo(f"Balanced accuracy: {test_metrics['bacc']}, Threshold: {best_th}")
  else:
    click.echo(f"MAE: {test_metrics['mae']}")
  
  if task == 'classification':
    preds_val = torch.sigmoid(torch.from_numpy(preds_val)).numpy()
    preds_test = torch.sigmoid(torch.from_numpy(preds_test)).numpy()

  torch.save(model.state_dict(), "../models/model_classification.pth")

  np.save(f'../predictions/val-{task}.npy', preds_val)
  np.save(f'../predictions/test-{task}.npy', preds_test)

if __name__ == '__main__':
  main()
