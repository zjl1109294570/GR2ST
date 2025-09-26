# 在Jupyter Notebook的第一个和唯一单元格
import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import GR2ST
from dataset import HERDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R

import sys
from types import SimpleNamespace
from sklearn.metrics import mean_squared_error, mean_absolute_error



def build_loaders_inference():
    loaders = []
    for i in range(32):
        dataset = HERDataset(train=False, fold=i)
       
        batch_size = len(dataset)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        loaders.append(loader)
    print("Finished building loaders")
    return loaders


def get_embeddings(model_path, model):
    test_loaders = build_loaders_inference()
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to('cuda')
    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for loader in test_loaders:
           
            for batch in loader:
                
                batch = {k: v.cuda() for k, v in batch.items() if
                         k == "image_features" or k == "expression" or k == "position" or k == "cell_type"}
                
                
                image_features = batch["image_features"]
                image_embeddings = model.image_projection(image_features)
                test_image_embeddings.append(image_embeddings)

    
                spot_feature = batch["expression"]
                x = batch["position"][:, 0].long()
                y = batch["position"][:, 1].long()
                centers_x = model.x_embed(x)
                centers_y = model.y_embed(y)
                cell_type = batch["cell_type"].long()
                cell_type_embeddings = model.cell_type_embed(cell_type)
                spot_features = spot_feature + centers_x + centers_y + cell_type_embeddings
                
                
                dynamic_edge_index = model.build_cell_type_aware_graph(spot_features, cell_type)
                spatial_edge_index = model.build_spatial_graph(batch["position"])
                
                
                dynamic_features = model.dynamic_head_proj(spot_features)
                spatial_features = model.spatial_proj(spot_features)
                
                
                dynamic_features = model.dynamic_gat(dynamic_features, dynamic_edge_index)
                
               
                spatial_features = model.spatial_gat(spatial_features, spatial_edge_index)
                
                
                if model.fusion_type == 'concat':
                    fused_features = torch.cat([dynamic_features, spatial_features], dim=-1)
                elif model.fusion_type == 'sum':
                    fused_features = dynamic_features + spatial_features
                elif model.fusion_type == 'max':
                    fused_features = torch.max(dynamic_features, spatial_features)
                else:  # mean
                    fused_features = (dynamic_features + spatial_features) / 2
                    
                fused_features = model.fusion_layer(fused_features)
                
                
                spot_embedding = model.spot_projection(fused_features)
                spot_embeddings.append(spot_embedding)
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


def save_embeddings(model_path, save_path, datasize, dim):
    model = GR2ST(
                
                temperature=1.0,
                image_dim=1024,     
                spot_dim=785,      
                projection_dim=256,   
                heads_num=8,          
                dropout=0.1,      
                dynamic_topk=20,     
                spatial_topk=20,      
                fusion_type='sum' 
            )

    img_embeddings_all, spot_embeddings_all = get_embeddings(model_path, model)
    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    print("img_embeddings_all.shape", img_embeddings_all.shape)
    print("spot_embeddings_all.shape", spot_embeddings_all.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        print("image_embeddings.shape", image_embeddings.shape)
        print("spot_embeddings.shape", spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)


SAVE_EMBEDDINGS = True
# SAVE_EMBEDDINGS = False

names = os.listdir(r"../dataset/her2st/ST-cnts")
names.sort()
names = [i[:2] for i in names][1:33]

datasize = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy").shape[1] for
            name in names]

if SAVE_EMBEDDINGS:
    for fold in range(32):
        save_embeddings(model_path=f"./model_result/her2st/{names[fold]}/best_{fold}.pt",
                        save_path=f"./embedding_result/her2st_result/embeddings_{fold}/",
                        datasize=datasize, dim=785)


spot_expressions = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")
                    for name in names]
all_hvg_pcc_list = []
all_heg_pcc_list = []
all_mse_list = []
all_mae_list = []

for fold in range(32):

    save_path = f"./embedding_result/her2st_result/embeddings_{fold}/"
    spot_embeddings = [np.load(save_path + f"spot_embeddings_{i + 1}.npy") for i in range(32)]
    image_embeddings = np.load(save_path + f"img_embeddings_{fold + 1}.npy")

    image_query = image_embeddings
    expression_gt = spot_expressions[fold]
    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    method = "weighted"
    save_path = f"./her2st_pred_att/{names[fold]}/"
    os.makedirs(save_path, exist_ok=True)
    if image_query.shape[1] != 256:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)

    indices = find_matches(spot_key, image_query, top_k=200)
    matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
    for i in range(indices.shape[0]):
        a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
        reciprocal_of_square_a = np.reciprocal(a ** 2)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        weights = weights.flatten()
        matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0,
                                                        weights=weights)

    true = expression_gt
    pred = matched_spot_expression_pred

    gene_list_path = "../dataset/her2st/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))
    
  
    slice_name = names[fold]  
    slice_dir = os.path.join("./st_results/her2st/")
    os.makedirs(slice_dir, exist_ok=True)
    
   
    gene_pcc_list = []
    for j in range(len(gene_list)):
        true_j = true[:, j]
        pred_j = pred[:, j]
        pcc, _ = pearsonr(true_j, pred_j)
        gene_pcc_list.append((gene_list[j], pcc))
    
    
    gene_pcc_list.sort(key=lambda x: x[1], reverse=True)
    
    pcc_path = os.path.join(slice_dir, f"{slice_name}_corr.csv")
    with open(pcc_path, 'w') as f:
        f.write("Gene,PCC\n")
        for gene, pcc in gene_pcc_list:
            f.write(f"{gene},{pcc}\n")
    
    adata_ture = anndata.AnnData(true)
    adata_pred = anndata.AnnData(pred)

    adata_pred.var_names = gene_list
    adata_ture.var_names = gene_list

    

    hvg_pcc, hvg_p = get_R(adata_pred, adata_ture)

    hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

    all_hvg_pcc_list.append(np.mean(hvg_pcc))


print(f"avg hvg pcc: {np.mean(all_hvg_pcc_list):.4f}")