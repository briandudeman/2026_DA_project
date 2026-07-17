import regex as re
import csv
import itertools
from pathlib import Path
import pandas as pd
from typing import List



data_directory = "linkdti_bbb_data/" 
output_file_name = "drug_features"
drug_card_file = "../luo_2017_dataset/drugbank/drugbank.txt"
drug_list_file = "../LinkDTI_heterogenous_KG_DA_project/data/drug.txt"
drug_sdf_file = "../luo_2017_dataset/drugbank/all.sdf"

# for the drugs in linkdti's list, gets specific features from drugbanks drug card files
def collate_linkdti_drugbank_cards(data_directory: str, output_file_name: str, drug_card_file: str, drug_list_file: str, features: List[str]):
    drug_list_len = 0
    with open(drug_list_file, "r") as drug_list_len_finder:
        for i in drug_list_len_finder:
            drug_list_len += 1


    output_file = Path(data_directory + output_file_name + ".csv")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(drug_list_file, "r") as drug_list:
        with open(drug_card_file, "r") as drug_card_list:
            with open(data_directory + output_file_name + ".csv", "w+") as output_file:
                writer = csv.writer(output_file, delimiter="\t")
                writer.writerow(["DrugBank_ID"] + features)
                drug_card_list = list(drug_card_list) # not the best memory wise but otherwise next and enumerate is tricky
                for i, drug_id in enumerate(drug_list):
                    print("drug", i + 1, "out of", drug_list_len)
                    found = False
                    drug_features = ["N/A"] * len(features)
                    for j in range(len(drug_card_list) - 1):
                        drug_card_line = drug_card_list[j]
                        if re.search(f"#BEGIN_DRUGCARD {drug_id}", drug_card_line):
                            found = True
                        
                        if found:
                            for feature_index in range(len(drug_features)):
                                if re.search(f"# {features[feature_index]}:", drug_card_line):
                                    drug_features[feature_index] = drug_card_list[j+1].strip()
                            
                        if re.search(f"#END_DRUGCARD {drug_id}", drug_card_line): # terminating if, can be changed
                            writer.writerow([drug_id.strip()] + drug_features)
                            break


# same as collate_linkdti_drugbank_cards, but with drugbanks sdf file of all the drugs, useful for smiles and other points
def collate_linkdti_drugbank_sdf(data_directory: str, output_file_name: str, drug_sdf_file: str, drug_list_file: str, features: List[str]):
    drug_list_len = 0
    with open(drug_list_file, "r") as drug_list_len_finder:
        for i in drug_list_len_finder:
            drug_list_len += 1
    
    output_file = Path(data_directory + output_file_name + ".csv")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(drug_list_file, "r") as drug_list:
        with open(drug_sdf_file, "r") as drug_sdf_list:
            with open(data_directory + output_file_name + ".csv", "w+") as output_file:
                writer = csv.writer(output_file, delimiter="\t")
                writer.writerow(["DrugBank_ID"] + features)
                with open("log.txt", "a") as log:
                    drug_sdf_list = list(drug_sdf_list)

                    for i, drug_id in enumerate(drug_list):
                        print("drug", i + 1, "out of", drug_list_len)
                        found = False
                        drug_features = ["N/A"] * len(features)
                        for j in range(len(drug_sdf_list) - 1):
                            drug_sdf_line = drug_sdf_list[j]
                            next_line = drug_sdf_list[j+1]
                            if re.search(f"> <DATABASE_ID>", drug_sdf_line) and re.search(drug_id, next_line):
                                found = True
                            if found:
                                for feature_index in range(len(drug_features)):
                                    if re.search(f"> <{features[feature_index]}>", drug_sdf_line):
                                        drug_features[feature_index] = next_line.strip()
                            log.write("\n")
                            if re.search(r"\$\$\$\$", drug_sdf_line) and found: # terminating if, can be changed
                                writer.writerow([drug_id.strip()] + drug_features)
                                break

def combine_sdf_data_and_drugbank_data(data_directory: str, output_file_name: str, drug_sdf_feature_file: str, drug_card_feature_file: str):
        output_file = Path(data_directory + output_file_name + ".csv")
        output_file.parent.mkdir(exist_ok=True, parents=True)

        drug_card_list = pd.read_csv(drug_card_feature_file, delimiter="\t")
        drug_sdf_list = pd.read_csv(drug_sdf_feature_file, delimiter="\t")

        all_drug_features = pd.concat([drug_card_list, drug_sdf_list.iloc[:, 1:]], axis=1)

        all_drug_features.to_csv(output_file, sep="\t", index=False)
        


combine_sdf_data_and_drugbank_data(data_directory=data_directory, output_file_name=output_file_name, drug_card_feature_file="linkdti_bbb_data/drug_features.csv", drug_sdf_feature_file="linkdti_bbb_data/drug_smiles_collation.csv")


#collate_linkdti_drugbank_cards(data_directory=data_directory, output_file_name=output_file_name, drug_card_file=drug_card_file, drug_list_file=drug_list_file, features=["Molecular_Weight_Avg", "Predicted_LogP_Hydrophobicity", "Predicted_LogS", "PubChem_Compound_ID"])

#collate_linkdti_drugbank_sdf(data_directory=data_directory, output_file_name="drug_smiles_collation", drug_sdf_file=drug_sdf_file, drug_list_file=drug_list_file, features=["SMILES", "JCHEM_AVERAGE_POLARIZABILITY", "JCHEM_PKA_STRONGEST_ACIDIC", "JCHEM_PKA_STRONGEST_BASIC", "JCHEM_POLAR_SURFACE_AREA", "JCHEM_REFRACTIVITY"])               