from rdkit import Chem
from rdkit.Chem import AllChem
import traceback
import argparse
import pandas as pd
from pathlib import Path
import sys, os


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../'))
sys.path.insert(0, root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='script to compare ImageMol and TitanBBB on a specific dataset')

    parser.add_argument('--data_root', type=Path,
                    help='The relative path to the folder containing different datasets from "2026 Project"')

    parser.add_argument('--dataset', type=Path,
                    help='The name of the data folder in data_root, should be the same as the csv file')

    parser.add_argument('--smiles_name', type=str, default="smiles",
                    help='the name of the smiles column in the dataset')
    

    return parser.parse_args()


def mol2sdf(mol: Chem.rdchem.Mol, sdf_save_path: Path):
    if mol is not None:


        writer = Chem.SDWriter(sdf_save_path)

        writer.write(mol)
        writer.close()
        print("wrote ", sdf_save_path)

    else:
        print("molecule does not exist, not saving")


def generate_3d_comformer(smiles, sdf_save_path, mmffVariant="MMFF94", randomSeed=0, maxIters=500, increment=2, optim_count=10, save_force=False):
    count = 0
    while count < optim_count:
        try:
            m = Chem.MolFromSmiles(smiles)
            m3d = Chem.AddHs(m)
            if save_force:
                try:
                    AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                    res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                    m3d = Chem.RemoveHs(m3d)
                except:
                    m3d = Chem.RemoveHs(m3d)
                    print("forcing saving molecule which can't be optimized ...")
                    mol2sdf(m3d, sdf_save_path)
            else:
                if AllChem.EmbedMolecule(m3d, randomSeed=randomSeed, useRandomCoords=True, maxAttempts=500) != -1:
                    res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                    print("res", res)
                    m3d = Chem.RemoveHs(m3d)
                else:
                    # EmbedMolecule failed
                    res = -1
        except Exception as e:
            traceback.print_exc()
        
        if res == 1:
            maxIters = maxIters * increment
            count += 1
            continue
        elif res == 0:
            print(sdf_save_path, "has converged\n")
            mol2sdf(m3d, sdf_save_path)
            break
        else:
            print(sdf_save_path, "has not converged, still writing")
            mol2sdf(m3d, sdf_save_path)
            break
    if save_force:
        print("forcing saving molecule without convergence ...")
        mol2sdf(m3d, sdf_save_path)



def main(args):
    
    processed_data = pd.read_csv(os.path.join(root_dir, args.data_root, args.dataset, f"processed/{args.dataset}_processed_ac.csv"))

    output_directory = os.path.join(root_dir, f"repositories/VideoMol/datasets/fine-tuning/{args.dataset}/sdfs")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for _, row in processed_data.iterrows():
        generate_3d_comformer(row[args.smiles_name], os.path.join(output_directory, f"{row['index']}.sdf"))


if __name__ == "__main__":
    args = parse_args()
    main(args)