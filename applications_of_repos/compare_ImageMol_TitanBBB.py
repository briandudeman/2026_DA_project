import numpy as np
import argparse
from pathlib import Path
import os, sys
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../'))
sys.path.insert(0, root_dir)

from repositories.TitanBBB.TITAN_BBB.src.utils.model import SMILESModel
from repositories.TitanBBB.TITAN_BBB.src.utils.dataset import load_and_process_feature, SMILESDataset
from repositories.TitanBBB.TITAN_BBB.src.bbbp.gather_image_features_bbbp import process_data as process_image_data, get_resnet50_encoder
from repositories.TitanBBB.TITAN_BBB.src.bbbp.generate_text_features_bbbp import process_data as process_text_data

from repositories.ImageMol.utils.public_utils import setup_device
from repositories.ImageMol.dataloader.image_dataloader import get_datasets, load_filenames_and_labels_multitask, ImageDataset
from repositories.ImageMol.utils.public_utils import setup_device
from repositories.ImageMol.data_process.smiles2img_pretrain import loadSmilesAndSave

sys.path.insert(0, os.path.abspath(os.path.join(root_dir, "repositories", "ImageMol")))
from repositories.ImageMol.model.cnn_model_utils import load_model, evaluate_on_multitask
from repositories.ImageMol.model.train_utils import load_smiles
from repositories.ImageMol.model.evaluate import metric as utils_evaluate_metric
from repositories.ImageMol.model.evaluate import metric_multitask as utils_evaluate_metric_multitask
from repositories.ImageMol.model.evaluate import metric_reg as utils_evaluate_metric_reg
from repositories.ImageMol.model.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask
from repositories.ImageMol.utils.public_utils import cal_torch_model_params, setup_device
from repositories.ImageMol.utils.splitter import scaffold_split_train_val_test
sys.path.insert(0, root_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='script to compare ImageMol and TitanBBB on a specific dataset')

    parser.add_argument('--data_root', type=Path,
                    help='The relative path to the folder containing different datasets from "2026 Project"')

    parser.add_argument('--dataset', type=Path,
                    help='The name of the data folder in data_root, should be the same as the csv file')
    
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--titanbbb_set_name', type=str, default="bbbp",
                        help='the prefix/name of the titanbbb feature file to save features to')
    parser.add_argument('--fig_name', type=str,
                        help='the name of the figure that will be getting saved')
    parser.add_argument('--titanbbb_features', type=str, help='The relative path to the titanbbb feature directory from "2026 Project"')
    parser.add_argument('--titanbbb_models', type=str, help='The relative path to the titanbbb models directory from "2026 Project"')
    parser.add_argument('--process_data', action='store_true', help='whether or not to process the data')


    return parser.parse_args()

# will need to be changed based on the dataset, the order and names of df columns are for BBBP currently
def imol_process_dataset(data_root:str, dataset:str, label_name: str, smiles_name: str, index_name = None):


    raw_file_path = os.path.join(data_root, dataset, "{}.csv".format(dataset))
    img_save_root = os.path.join(data_root, dataset, "processed/224")
    csv_save_path = os.path.join(data_root, dataset, "processed/{}_processed_ac.csv".format(dataset))
    error_save_path = os.path.join(data_root, dataset, "error_smiles.csv")

    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path, sep=",")

    def process(index, smile):
            try:
                filename = "{}.png".format(index)
                img_save_path = os.path.join(img_save_root, filename)
                loadSmilesAndSave(smile, img_save_path)
                return smile
            except:
                print("returning None because smiles could not be loaded for index", index)
                return None
    
    

    if index_name:

        other_col_names = [c for c in df.columns if c not in [smiles_name, label_name, index_name]]

        df_rearranged = df.rename(columns={index_name: "index", smiles_name: "smiles", label_name: "label"})
        df_rearranged = df_rearranged[["index"] + other_col_names + ["smiles", "label"]]

        print("if", df_rearranged)

        processed_smiles = [process(index, smile) for index, smile in zip(df[index_name], df[smiles_name])]
        
    else:
        other_col_names = [c for c in df.columns if c not in [smiles_name, label_name]]

        df_rearranged = df.rename(columns={smiles_name: "smiles", label_name: "label"})        
        df_rearranged = df_rearranged[other_col_names + ["smiles", "label"]]
        df_rearranged.insert(0, "index", range(1, df.shape[0] + 1))

        processed_smiles = [process(index, smile) for index, smile in zip(df_rearranged["index"], df[smiles_name])]
        
    
    mask = [smile is not None for smile in processed_smiles]


    pd_processed_ac = pd.DataFrame(columns=["index"] + other_col_names + ["smiles", "label"])
    pd_processed_ac = df_rearranged[mask]

    error_smiles = []

    pd_processed_ac.to_csv(csv_save_path, index=False)

    if len(error_smiles) > 0:
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)






def imol_load_model_and_dataloader(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device, device_ids = setup_device(1)

    if args.process_data:
        imol_process_dataset(args.data_root, args.dataset, "p_np", "SMILES")

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.data_root, data_type="processed")
    args.verbose = True

    print(f"using imol dataset path: {args.data_root}/{args.dataset}")
    # architecture name
    if args.verbose:
        print('Architecture: {}'.format(args.image_model))

    ##################################### initialize some parameters #####################################
    if args.task_type == "classification":
        eval_metric = "rocauc"
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ##################################### load data #####################################
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    smiles = load_smiles(args.txt_file)
    train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, len(names))), smiles, frac_train=0,
                                                                 frac_valid=0, frac_test=1)

    name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[
        test_idx], labels[train_idx], labels[val_idx], labels[test_idx]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks)

    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            try:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint)
            except:
                checkpoint = torch.load(args.resume)["model_state_dict"]
                model.load_state_dict(checkpoint)
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("params: {}".format(cal_torch_model_params(model)))
    #model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))
    
    return model, test_dataloader, criterion, eval_metric




@torch.no_grad()
def evaluate_titanbbb_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification", return_data_dict=False):
    assert task_type in ["classification", "regression"]

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        tab, img, txt, labels = data
        tab, img, txt, labels = tab.to(device), img.to(device), txt.to(device), labels.to(device)
        sample_num += tab.shape[0]
        #tab = torch.unsqueeze(tab, 0)
        #img = torch.unsqueeze(img, 0)
        #txt = torch.unsqueeze(txt, 0)
        
        with torch.no_grad():
            pred = model(tab, img, txt)
            #for i, prd in enumerate(pred):
            #    print(i, prd.shape)
            #print("pred", pred)
            labels = labels.view(pred[0].shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred[0].double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred[0].double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred[0].shape))
        y_scores.append(pred[0])

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_true = np.reshape(y_true, (y_true.shape[0], 1))
    y_scores = np.reshape(y_scores, (y_scores.shape[0], 1))
    print("tb y_true shape", y_true.shape)
    if y_true.shape[1] == 1:
        if task_type == "classification":
            print("binary")
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            print("multi-task", y_true.shape, y_pred.shape, y_pro.shape)
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1]), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")


def titanbbb_process_bbbp_data(args, data_root:str, dataset:str):

    #===================================== images ===============================================
    # Get the root directory (2026 Project) from script location


    bbbp_image_dir = os.path.join(data_root, dataset, 'processed/224')
    bbbp_csv_path = os.path.join(data_root, dataset, f'processed/{dataset}_processed_ac.csv')
    
    bbbp = pd.read_csv(bbbp_csv_path)

    device = torch.device("cpu")

    encoder = get_resnet50_encoder(device)

    process_image_data(bbbp, bbbp_image_dir, root_dir, encoder, device, args.titanbbb_set_name, args.task_type)

    #============================================================================================

    #===================================== tabular ==============================================

    smiles_list = bbbp.smiles.values.tolist()

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator([i[0] for i in Descriptors.descList])
    desc_list = [calc.CalcDescriptors(mol) for mol in tqdm(mols)]
    desc_2d = np.array(desc_list, dtype=np.float32)
    desc_2d = np.nan_to_num(desc_2d, nan=0.0, posinf=0.0, neginf=0.0)

    desc_list = [MACCSkeys.GenMACCSKeys(mol) for mol in tqdm(mols)]
    desc_bitvects = [np.array(list(maccs.ToBitString()), dtype=int) for maccs in desc_list]
    desc_maccs = np.array(desc_bitvects, dtype=np.int8)

    combined = np.concatenate([desc_2d, desc_maccs], axis=1)

    np.save(os.path.join(root_dir, f'repositories/TitanBBB/TITAN_BBB/features/{args.titanbbb_set_name}-tabular-{args.task_type}.npy'), combined)

    #============================================================================================

    #===================================== text =================================================

    
    chembert_model_path = 'DeepChem/ChemBERTa-100M-MLM'

    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(chembert_model_path)
    model = AutoModel.from_pretrained(chembert_model_path, output_hidden_states=True).to(device)

    model.eval()

    process_text_data(bbbp, tokenizer, model, device, args.titanbbb_set_name, args.task_type, root_dir)

    #============================================================================================



def titanbbb_load_model_and_dataloader(args):
    
    proj_dim = 2048
    dropout = 0.1

    BASE_FEATURES_PATH = os.path.join(args.titanbbb_features, '{}-{}-{}.npy')
    model_path = os.path.join(args.titanbbb_models, f'model_{args.task_type}.pth')

    tabular_features = load_and_process_feature(BASE_FEATURES_PATH, args.dataset, 'tabular', args.task_type)
    image_features = load_and_process_feature(BASE_FEATURES_PATH, args.dataset, 'image', args.task_type)
    text_features = load_and_process_feature(BASE_FEATURES_PATH, args.dataset, 'text', args.task_type)

    d_tab = tabular_features.shape[1]
    d_img = image_features.shape[1]
    d_txt = text_features.shape[1]

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

    processed_smiles_data = pd.read_csv(os.path.join(args.data_root, args.dataset, f'processed/{args.dataset}_processed_ac.csv'))

    y = processed_smiles_data["label"]

    test_dataset = SMILESDataset(X_tab=tabular_features, X_img=image_features, X_txt=text_features, labels=y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
    criterion = nn.BCEWithLogitsLoss(reduction="none")



    return model, test_dataloader, criterion







def main(args):
    device, device_ids = setup_device(1)

    
    imol_model, imol_dataloader, imol_criterion, imol_eval_metric = imol_load_model_and_dataloader(args)


    test_loss, imol_test_results, test_data_dict = evaluate_on_multitask(model=imol_model, data_loader=imol_dataloader,
                                                                    criterion=imol_criterion, device=device, epoch=-1,
                                                                    task_type=args.task_type, return_data_dict=True)
    imol_test_result = imol_test_results[imol_eval_metric.upper()]
    imol_fpr, imol_tpr, imol_thresholds = imol_test_results["ROC_CURVE"]
    imol_bacc = imol_test_results["BACC"]
    imol_acc = imol_test_results["ACC"]
    imol_f1 = imol_test_results["F1"]
    imol_precision = imol_test_results["PRECISION"]
    imol_recall = imol_test_results["RECALL"]
    print("imol before recall", imol_recall)
    imol_precision_c, imol_recall_c, imol_thresholds = imol_test_results["PR_CURVE"]
    imol_pr_auc = imol_test_results["PR_AUC"]
    imol_specificity = imol_test_results["SPEC"]
    imol_mse = imol_test_results["MSE"]
    imol_mae = imol_test_results["MAE"]
    imol_rmse = imol_test_results["RMSE"]

    print("imol test ", imol_test_result)
    
    if args.process_data:
        titanbbb_process_bbbp_data(args, os.path.join(root_dir, args.data_root), args.dataset)

    tb_model, tb_dataloader, tb_criterion = titanbbb_load_model_and_dataloader(args)

    test_loss, tb_test_results, test_data_dict = evaluate_titanbbb_on_multitask(model=tb_model, data_loader=tb_dataloader,
                                                                    criterion=tb_criterion, device=device, epoch=-1,
                                                                    task_type="classification", return_data_dict=True)

    tb_test_result = tb_test_results["ROCAUC"]
    tb_fpr, tb_tpr, tb_thresholds = tb_test_results["ROC_CURVE"]
    tb_bacc = tb_test_results["BACC"]
    tb_acc = tb_test_results["ACC"]
    tb_f1 = tb_test_results["F1"]
    tb_precision = tb_test_results["PRECISION"]
    tb_recall = tb_test_results["RECALL"]
    tb_precision_c, tb_recall_c, tb_thresholds = tb_test_results["PR_CURVE"]
    tb_pr_auc = tb_test_results["PR_AUC"]
    tb_specificity = tb_test_results["SPEC"]
    tb_mse = tb_test_results["MSE"]
    tb_mae = tb_test_results["MAE"]
    tb_rmse = tb_test_results["RMSE"]


    print("[tb test] {}: {:.1f}%".format("rocauc", tb_test_result * 100))

    print("[imol test] {}: {:.1f}%".format(imol_eval_metric, imol_test_result * 100))

    plt.figure(1, figsize=(8, 6), dpi=300)

    plt.suptitle("ROC Curve")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(imol_fpr, imol_tpr, color="#ff7256", linestyle="solid", label="ImageMol")
    plt.plot(tb_fpr, tb_tpr, color="#6495ed", linestyle="solid", label="TitanBBB")

    print("[tb test] {}: {:.1f}% \n [imol test] {}: {:.1f}%".format("rocauc", tb_test_result * 100, imol_eval_metric, imol_test_result * 100))
    print("tb_bacc", tb_bacc)
    print("tb_acc", tb_acc)
    print("tb_f1", tb_f1)
    print("tb_precision", tb_precision)
    print("tb_recall", tb_recall)
    print("tb_specificity", tb_specificity)
    print("tb_mse", tb_mse)
    print("tb_mae", tb_mae)
    print("tb_rmse", tb_rmse)
    plt.legend()

    plt.savefig(args.fig_name + ".png", dpi=400)

    plt.figure(2, figsize=(8,6))
    plt.suptitle("Precision-Recall/Sensitivity Curve")

    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')

    plt.plot(imol_recall_c, imol_precision_c, color="#ff7256", linestyle="solid", label="ImageMol")
    plt.plot(tb_recall_c, tb_precision_c, color="#6495ed", linestyle="solid", label="TitanBBB")

    print("[tb test] {}: {:.1f}% \n [imol test] {}: {:.1f}%".format("rocpr", tb_pr_auc * 100, "rocpr", imol_pr_auc * 100))
    print("imol_bacc", imol_bacc)
    print("imol_acc", imol_acc)
    print("imol_f1", imol_f1)
    print("imol_precision", imol_precision)
    print("imol_recall", imol_recall)
    print("imol_specificity", imol_specificity)
    print("imol_mse", imol_mse)
    print("imol_mae", imol_mae)
    print("imol_rmse", imol_rmse)

    

    plt.legend()
    plt.savefig(args.fig_name + "1.png", dpi=400)
    plt.show()






if __name__ == "__main__":
    args = parse_args()
    main(args)

