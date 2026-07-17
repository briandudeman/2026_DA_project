import pandas as pd
import numpy as np
import csv, argparse, os


from smiles2img_pretrain import loadSmilesAndSave

def main():

    parser = argparse.ArgumentParser(description='Pretraining Data Generation for ImageMol')
    parser.add_argument('--dataroot', type=str, default="./datasets/finetuning/classification", help='data root')
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. data')
    args = parser.parse_args()

    raw_file_path = os.path.join(args.dataroot, args.dataset, "{}.csv".format(args.dataset))
    img_save_root = os.path.join(args.dataroot, args.dataset, "processed/224")
    csv_save_path = os.path.join(args.dataroot, args.dataset, "processed/{}_processed_ac.csv".format(args.dataset))
    error_save_path = os.path.join(args.dataroot, args.dataset, "error_smiles.csv")

    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path, sep=",")

    index, compound_name, p_np, smiles = df["num"].values, df["name"].values, df["p_np"].values, df["smiles"].values

    processed_ac_data = []
    error_smiles = []
    for i, (index, name, p_np, smile) in enumerate(zip(index, compound_name, p_np, smiles)):
        filename = "{}.png".format(index)
        img_save_path = os.path.join(img_save_root, filename)
        try:
            loadSmilesAndSave(smile, img_save_path)
            processed_ac_data.append([index, filename, name, smile, p_np])
        except:
            print("passing, not writing data")
            pass

    
    pd_processed_ac = pd.DataFrame(processed_ac_data, columns=["index", "filename", "drug_name", "smiles", "label"])

    pd_processed_ac.to_csv(csv_save_path, index=False)


    if len(error_smiles) > 0:
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)


if __name__ == '__main__':
    main()

