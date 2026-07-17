
import numpy as np
import pandas as pd
import os, sys
import torch
from torch import nn

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../../../'))
sys.path.insert(0, root_dir)


from utils.model import SMILESModel
from utils.dataset import bbbp_load_and_process_feature, SMILESDataset

from ImageMol.utils.public_utils import setup_device
sys.path.insert(0, os.path.join(root_dir, "ImageMol"))
from model.cnn_model_utils import evaluate_titanbbb_on_multitask



bbbp = pd.read_csv("TITAN_BBB/archive/BBBP.csv")

x = bbbp["smiles"]
y = bbbp["p_np"]

task = "classification"
proj_dim = 2048
dropout = 0.1

bbbp_image_dir = os.path.join(root_dir, 'ImageMol/datasets/finetuning/classification/BBBP/processed/224')
bbbp_csv_path = os.path.join(root_dir, 'ImageMol/datasets/finetuning/classification/BBBP/processed/bbbp_processed_ac.csv')


BASE_FEATURES_PATH = os.path.join(root_dir, 'TITAN_BBB/features/{}-{}-{}.npy')
model_path = os.path.join(root_dir, f'TITAN_BBB/models/model_{task}.pth')

bbbp_tabular_features = bbbp_load_and_process_feature(BASE_FEATURES_PATH, 'tabular', task)
bbbp_image_features = bbbp_load_and_process_feature(BASE_FEATURES_PATH, 'image', task)
bbbp_text_features = bbbp_load_and_process_feature(BASE_FEATURES_PATH, 'text', task)

d_tab = bbbp_tabular_features.shape[1]
d_img = bbbp_image_features.shape[1]
d_txt = bbbp_text_features.shape[1]

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


test_dataloader = SMILESDataset(X_tab=bbbp_tabular_features, X_img=bbbp_image_features, X_txt=bbbp_text_features, labels=y)
criterion = nn.BCEWithLogitsLoss(reduction="none")


device, device_ids = setup_device(1)

test_loss, test_results, test_data_dict = evaluate_titanbbb_on_multitask(model=model, data_loader=test_dataloader,
                                                                criterion=criterion, device=device, epoch=-1,
                                                                task_type="classification", return_data_dict=True)

test_result = test_results["rocauc".upper()]

print("[test] {}: {:.1f}%".format("rocauc", test_result * 100))