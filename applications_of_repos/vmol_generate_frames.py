import argparse, os, sys
from pathlib import Path
import pandas as pd
from typing import List
from multiprocessing import Pool, freeze_support


root_dir = os.path.abspath("/Users/lewconb/2026 Project")

from applications_of_repos.vmol_worker import rotate_and_save, init_worker

def parse_args():
    parser = argparse.ArgumentParser(description='script to compare ImageMol and TitanBBB on a specific dataset')

    parser.add_argument('--sdf_directory', type=Path,
                    help='The relative path to the folder containing the sdf files from "2026 Project"')

    parser.add_argument('--data_root', type=Path,
                    help='The relative path to the folder containing the features of the compounds in the sdf files, from "2026 Project"')
    
    parser.add_argument('--dataset', type=Path,
                    help='The name of the dataset with the features')

    parser.add_argument('--output_directory', type=Path,
                    help='The relative path to the folder where the frames should go from "2026 Project')
    
    return parser.parse_args()    

            

def main(args):
    sdf_directory = os.fsencode(os.path.join(root_dir, args.sdf_directory))
    
    sdf_features = pd.read_csv(os.path.join(root_dir, args.data_root, f"{args.dataset}/processed/{args.dataset}_processed_ac.csv"))

    sdf_dir_list = os.listdir(sdf_directory)
    
    
    pool_args = []

    for i, sdf in enumerate(sdf_dir_list):

        #print(str(i) + " out of " + str(len(sdf_dir_list)))

        sdf_index = os.fsdecode(sdf).replace(".sdf", "")
        #print("sdf_index", sdf_index)
        sdf_filepath = os.path.join(root_dir, args.sdf_directory, f"{sdf_index}.sdf")
        

        save_frames_path = os.path.join(root_dir, args.output_directory, f"{sdf_index}")

        pool_args.append((sdf_filepath, save_frames_path, sdf_index, sdf_features, os.path.join(root_dir, args.output_directory), len(sdf_dir_list)))

    with Pool(processes=10, initializer=init_worker) as pool:
        results = pool.starmap(rotate_and_save, pool_args)
        print("results", results)




args = parse_args()
main(args)
