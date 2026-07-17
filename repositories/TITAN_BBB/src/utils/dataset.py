import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class SMILESDataset(Dataset):
  def __init__(self, X_tab, X_img, X_txt, labels):
    self.X_tab = X_tab
    self.X_img = X_img
    self.X_txt = X_txt
    self.labels = labels

  def __len__(self):
    return len(self.X_tab)

  def __getitem__(self, idx):
    X_tab = torch.tensor(self.X_tab[idx], dtype=torch.float)
    X_img = torch.tensor(self.X_img[idx], dtype=torch.float)
    X_txt = torch.tensor(self.X_txt[idx], dtype=torch.float)
    y = torch.tensor(self.labels[idx], dtype=torch.float)

    return X_tab, X_img, X_txt, y


def load_and_process_feature(BASE_FEATURES_PATH, dataset, feature_name, task):
  X_test = np.load(BASE_FEATURES_PATH.format(dataset, feature_name, task))

  scaler = StandardScaler()
  X_test = scaler.fit_transform(X_test)

  return X_test

def kd_load_and_process_feature(BASE_FEATURES_PATH, feature_name, task):
  kd_features = np.load(BASE_FEATURES_PATH.format("kd_kiba_predict", feature_name, task))

  scaler = StandardScaler()
  kd_features = scaler.fit_transform(kd_features)

  return kd_features

def bbbp_load_and_process_feature(BASE_FEATURES_PATH, feature_name, task):
  bbbp_features = np.load(BASE_FEATURES_PATH.format("bbbp", feature_name, task))

  scaler = StandardScaler()
  bbbp_features = scaler.fit_transform(bbbp_features)

  return bbbp_features

def linkdti_data_load_and_process_feature(BASE_FEATURES_PATH, feature_name, task):
  linkdti_data_features = np.load(BASE_FEATURES_PATH.format("linkdti_data_predict", feature_name, task))

  scaler = StandardScaler()
  linkdti_data_features = scaler.fit_transform(linkdti_data_features)

  return linkdti_data_features
