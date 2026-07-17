import os, sys
import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../'))
sys.path.insert(0, root_dir)

train_classification = pd.read_csv(os.path.join(root_dir, "repositories/ImageMol/datasets/tb_data/tb_data.csv"))
val_classification = pd.read_csv(os.path.join(root_dir, "repositories/ImageMol/datasets/tb_data/val_classification.csv"))
test_classification = pd.read_csv(os.path.join(root_dir, "repositories/ImageMol/datasets/tb_data/test_classification.csv"))

print("train val overlap: ", sum(train_classification["InChI_Key"].isin(val_classification["InChI_Key"])))
print("val test overlap: ", sum(val_classification["InChI_Key"].isin(test_classification["InChI_Key"])))
print("test train overlap: ", sum(test_classification["InChI_Key"].isin(train_classification["InChI_Key"])), "\n")


print("val train overlap: ", sum(val_classification["InChI_Key"].isin(train_classification["InChI_Key"])))
print("test val overlap: ", sum(test_classification["InChI_Key"].isin(val_classification["InChI_Key"])))
print("train test overlap: ", sum(train_classification["InChI_Key"].isin(test_classification["InChI_Key"])))
