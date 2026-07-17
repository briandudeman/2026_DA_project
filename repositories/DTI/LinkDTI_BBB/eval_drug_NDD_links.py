from linkdti_bbb import HeteroGCN, create_samples_and_labels, set_seed
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import normalize
import os
from typing import List

in_feats = 17
hidden_feats = 256
out_feats = 128
model = HeteroGCN(in_feats, hidden_feats, out_feats)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()
device = "cpu"
model_file_directory = "LinkDTI_BBB/models/"
model_file_name = "1_to_1_neg_9_rand_features"

NDDs = ["alzheimer disease", "alzheimer disease, familial", "alzheimer disease type 2", "alzheimer disease, familial, 3, with spastic paraparesis and apraxia", "alzheimer disease type 4",
        "parkinson disease", "parkinson disease, secondary", "parkinson disease, late-onset", "parkinson disease, juvenile, autosomal recessive", "parkinson disease 9", "parkinson disease 15, autosomal recessive early-onset", "parkinson disease 11, autosomal dominant", "parkinson disease 13, autosomal dominant, susceptibility to", "parkinson disease 8, autosomal dominant", "parkinson disease 4, autosomal dominant lewy body", "parkinson disease 7, autosomal recessive early-onset", "parkinson disease 6, autosomal recessive early-onset", "parkinson disease 14, autosomal recessive", "parkinson disease 1, autosomal dominant", "amyotrophic lateral sclerosis-parkinsonism/dementia complex 1", "parkinson disease 5, autosomal dominant", ]

def create_ndd_subgraph_and_features(ndds: List[str], drug_disease: np.ndarray, disease_list: pd.DataFrame):
    ndd_ids = []
    for ndd in ndds:
        ndd_ids.append(disease_list[disease_list["Disease_Name"] == ndd].index[0] + 1)
    
    print(ndd_ids)

    num_drug = len(drug_disease)

    # testing all the nonexistent edges to see most plausible association
    list_drug_disease = [(row, col) for row in range(num_drug) for col in ndd_ids if drug_disease[row, col] == 0]
    list_disease_drug = [(col, row) for row in range(num_drug) for col in ndd_ids if drug_disease[row, col] == 0]

    cols = []
    for col, row in list_disease_drug:
        cols.append(col)

    disease_count = len(set(cols))

    print("disease_count", disease_count)


    g_HIN = dgl.heterograph({
        ('drug', 'drug_disease association', 'disease'): list_drug_disease,
        ('disease', 'disease_drug association', 'drug'): list_disease_drug,
    })

    g = g_HIN.edge_type_subgraph([
        'drug_disease association', 'disease_drug association'
    ])

    features = get_features(g, drug_disease_list=list_drug_disease)
    return g, features

def get_features(g, drug_disease_list: List[tuple], extra_dim_embedding=8, seed=42):

    print("Reading actual features...")
    drug_features = pd.read_csv("LinkDTI_BBB/linkdti_bbb_data/drug_features.csv", delimiter="\t")

    # cleaning non numeric features for now
    drug_features = drug_features.select_dtypes(include="number")

    # normalizes each column
    drug_features = (drug_features - drug_features.min()) / (drug_features.max() - drug_features.min())

    # avging nan values
    for col in drug_features.columns:
        rng = np.random.default_rng(0)
        if drug_features[col].isna().sum() > 0:
            mu = drug_features[col].mean()
            sd = drug_features[col].std()
            filler = pd.Series(rng.normal(loc=mu, scale=sd, size=len(drug_features[col])))
            drug_features[col] = drug_features[col].fillna(filler)
   

    print("Collating real and random features...")
    set_seed(seed)

    ndd_drug_features = drug_features.iloc[[pair[0] for pair in drug_disease_list]].drop_duplicates()
    features = {}
    for ntype in g.ntypes:
        num_nodes = g.number_of_nodes(ntype)
        print(ntype, num_nodes)
        feat = nn.Parameter(torch.FloatTensor(num_nodes, extra_dim_embedding + drug_features.shape[1]).to(device))
        torch.nn.init.normal_(feat, mean=0, std=0.1)
        if ntype == "drug":
            feat = torch.cat((torch.tensor(ndd_drug_features.values), feat[:, ndd_drug_features.shape[1]:]), dim=1).float().to(device)
        features[ntype] = feat
    return features


# drug and disease list are indexes used in interaction matrix
def create_pairs(drug_list, disease_list, interaction_matrix, src_type, dst_type, num_neg_samples_per_pos=1, undirected=True, seed=42):
    print(f"Creating pairs for drug-target associations")
    set_seed(seed)
    
    possible_interactions = []

    for drug in drug_list:
        for disease in disease_list:
            if interaction_matrix[drug, disease] == 0:
                if undirected:
                    possible_interactions.append([min(drug, disease), max(drug, disease)])
                else:
                    possible_interactions.append([drug, disease])      

    return np.array(possible_interactions)

model.load_state_dict(torch.load(model_file_directory + model_file_name + ".pth", weights_only=True))

disease_list = pd.read_csv('LinkDTI_heterogenous_KG_DA_project/data/disease.txt', sep="\t", header=None)
disease_list.columns = ["Disease_Name"]

drug_list = pd.read_csv('LinkDTI_heterogenous_KG_DA_project/data/drug.txt', sep="\t", header=None)
drug_list.columns = ["Drug_Name"]

drug_disease = np.loadtxt('LinkDTI_heterogenous_KG_DA_project/data/mat_drug_disease.txt')

g, features = create_ndd_subgraph_and_features(ndds=NDDs, drug_disease=drug_disease, disease_list=disease_list)

ndd_ids = []
for ndd in NDDs:
    ndd_ids.append(disease_list[disease_list["Disease_Name"] == ndd].index[0] + 1)
    
edge_pairs = torch.from_numpy(create_pairs([i for i in range(0, 708)], ndd_ids, drug_disease, 'drug', 'disease', undirected=False))
print("edge pairs", edge_pairs)
print("edge pairs shape", edge_pairs.shape)

canonical_etype = ('drug', 'drug_disease association', 'disease')

predictions = model(g, features, edge_pairs, canonical_etype)

# filtering out the edges that already exist, easiest to do at this step
edge_pairs_cleaned = []
predictions_cleaned = []
for i, pair in enumerate(edge_pairs):
    if drug_disease[pair[0], pair[1]] == 0:
        edge_pairs_cleaned.append(pair.tolist())
        predictions_cleaned.append(predictions[i].item())

edge_pairs_preds = torch.cat((torch.tensor(edge_pairs_cleaned), torch.tensor(predictions_cleaned).unsqueeze(1)), dim=1)
edge_pairs_preds = pd.DataFrame(edge_pairs_preds, columns=["Drug_Index", "Disease_Index", "Pred_Score"])
edge_pairs_preds["DrugBank_ID"] = drug_list.iloc[edge_pairs_preds["Drug_Index"].values - 1]["Drug_Name"].values

#print(edge_pairs_preds.sort_values("Pred_Score", ascending=False).head(15))


akguller_founds = pd.read_csv("pre_identified_drugs/akguller_et_al_2025/akguller_et_al_2025_a1.csv", sep=r"\s+")

found_edge_preds = edge_pairs_preds[edge_pairs_preds["DrugBank_ID"].isin(akguller_founds["DrugBank_ID"])]
print(found_edge_preds.sort_values("Pred_Score", ascending=False))
print("Number unique drugs found that are in akguller a1", found_edge_preds["Drug_Index"].nunique())
