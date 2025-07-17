import numpy as np

import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import torch.optim as optim


from models.BranchBound import generate_proto_generator
from models.deconv_dataset import Spatial_Exp_Dataset
from models.deconv_model import HIDF
from models.deconv_trainer import HIDF_Trainer
from models.train import Context
from models.utils import set_seed, check_anndata, conver_adata_X_to_numpy, RMSE, \
    softmax_to_logits_matrix
import os
import pandas as pd
import scanpy as sc

def filter_genes(sc_rna_adata: sc.AnnData, st_rna_adata: sc.AnnData, cell_type_key='cell_type', n_genes=100,
                 log_transform=True, use_deg=True, logfoldchanges=0.5, p_value=0.05, mean_expression_threshold=0):
    sc.pp.filter_genes(sc_rna_adata, min_cells=3)
    sc.pp.filter_genes(st_rna_adata, min_cells=3)

    sc_rna_adata.var_names = sc_rna_adata.var_names.str.lower().tolist()
    st_rna_adata.var_names = st_rna_adata.var_names.str.lower().tolist()

    sc_rna_adata.var_names_make_unique()
    st_rna_adata.var_names_make_unique()

    gene_names_1 = set(sc_rna_adata.var_names.tolist())
    gene_names_2 = set(st_rna_adata.var_names.tolist())
    common_gene_names = list(gene_names_1 & gene_names_2)
    print(f'common gene number:{len(common_gene_names)}')

    new_single_data = sc_rna_adata[:, common_gene_names].copy()
    new_spatial_data = st_rna_adata[:, common_gene_names].copy()
    if use_deg:
        sc.pp.filter_cells(new_single_data, min_genes=1)
        if log_transform:
            sc.pp.normalize_total(new_single_data, target_sum=1e4)
            sc.pp.log1p(new_single_data)

        # select DEG
        sc.tl.rank_genes_groups(new_single_data, groupby=cell_type_key, n_genes=n_genes)
        celltype = new_single_data.obs[cell_type_key].unique().tolist()
        deg = sc.get.rank_genes_groups_df(new_single_data, group=celltype)
        deg = deg[deg['logfoldchanges']>=logfoldchanges]
        deg = deg[deg['pvals_adj']<p_value]
        deg_gene_names = deg['names'].str.lower().unique().tolist()

        mean_expression = np.mean(conver_adata_X_to_numpy(new_single_data.X), axis=0)
        keep_genes = mean_expression >= mean_expression_threshold  # 表达水平过滤
        keep_genes = new_single_data.var_names[keep_genes].tolist()
        deg_gene_names = list(set(deg_gene_names) & set(keep_genes))
        print(f'deg gene number:{len(deg_gene_names)}')

        new_single_data = sc_rna_adata[:, deg_gene_names].copy()
        new_spatial_data = st_rna_adata[:, deg_gene_names].copy()
    sc.pp.filter_cells(new_single_data, min_genes=1)
    return new_single_data, new_spatial_data

def train_HIDF(save_path, reg_lambda=1e-5, type=10000, times=0, resolution=0.3, k=9):
    set_seed(times)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sc_rna_origin_adata = check_anndata(
        f"../datasets/seqFISH/single/seqFISH_sc.h5ad", True)
    st_rna_origin_adata = check_anndata(
        f"../datasets/seqFISH//spatial/seqFISH_st{type}.h5ad", True)


    print(f'sc shape:{sc_rna_origin_adata.shape}')
    print(f'st shape:{st_rna_origin_adata.shape}')

    cell_type_key = 'cell_type'

    sc_rna_adata, st_rna_adata = \
        filter_genes(sc_rna_adata=sc_rna_origin_adata, st_rna_adata=st_rna_origin_adata, cell_type_key=cell_type_key,
                     n_genes=None, use_deg=True)

    sc.pp.normalize_total(st_rna_adata, target_sum=1e4)
    sc.pp.normalize_total(sc_rna_adata, target_sum=1e4)
    sc.pp.normalize_total(sc_rna_origin_adata, target_sum=1e4)
    sc.pp.log1p(sc_rna_origin_adata)
    proto_generator = generate_proto_generator(sc_rna_adata=sc_rna_adata,
                                               sc_origin_rna_adata=sc_rna_origin_adata,
                                               sc_omics_adata=None,
                                               resolution=resolution,
                                               save_path=save_path, save_bool=False)
    depth = proto_generator.calculate_depth()
    print(f'depth: {depth}')
    iterator_times = depth

    for i in range(iterator_times):
        if i == 0:
            proto_exp_matrix, proto_cell_type_matrix = proto_generator.generate_rna_proto_matrix()
            sc_proto_adata = sc.AnnData(X=proto_exp_matrix)

            cell_type_set_list = proto_generator.current_cell_type_set_list

            device_name = 'cuda:0'
            lr = 1e-3
            epoch = 300
            batch_size = 1024

            st_rna_matrix = conver_adata_X_to_numpy(st_rna_adata.X)
            sc_rna_proto_matrix = np.array(sc_proto_adata.X)

            proto_number = sc_rna_proto_matrix.shape[0]
            st_number = st_rna_matrix.shape[0]

            spatial = st_rna_origin_adata.obsm['spatial']
            neighbors = k
            neigh = NearestNeighbors(n_neighbors=neighbors)
            mtx = np.array(spatial)
            neigh.fit(mtx)
            A = neigh.kneighbors_graph(mtx)

            neighbor_index = np.zeros(shape=(st_number, neighbors))
            for i in range(st_number):
                indices = A[i, :].indices
                neighbor_index[i,:] = indices


            deconv_model = HIDF(sc_rna_proto_matrix,
                                              st_rna_matrix,
                                              proto_number,
                                              st_number,
                                              proto_cell_type_matrix)

            st_rna_dataset = Spatial_Exp_Dataset(data=st_rna_matrix, neighbor_index=neighbor_index)
            st_train_loader = DataLoader(st_rna_dataset, batch_size=batch_size, shuffle=True)

            deconv_trainer = HIDF_Trainer(model=deconv_model, train_dataset=st_rna_dataset,
                                          test_dataset=None, continue_train=False,
                                          trained_model_path=None, device_name=device_name, lr=lr,
                                          save_path=save_path)
            deconv_trainer.opt = optim.AdamW([
                                              {'params': deconv_model.gene_offset_parameter, 'lr':0.1},
                                              {'params': deconv_model.st_offset_parameter, 'lr':0.1},
                                              {'params':deconv_model.mapping_matrix, 'lr':0.1}],
                                              lr=lr)
            ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=None, random_seed=None)
            ctx.st_train_loader = st_train_loader
            ctx.pre_cell_type_matrix = None
            ctx.reg_lambda = reg_lambda
            ctx.constrain_loss_list = []
            ctx.regular_loss_list = []
            ctx.rec_gene_loss_list = []
            deconv_trainer.train(ctx)
            constrain_loss_list = ctx.constrain_loss_list
            regular_loss_list = ctx.regular_loss_list
            rec_gene_loss_list = ctx.rec_gene_loss_list

            target_dataloader = DataLoader(st_rna_dataset, batch_size=batch_size, shuffle=False)
            ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=None, random_seed=None)
            ctx.st_train_loader = target_dataloader
            deconv_trainer.deconv(ctx)
            st_cell_type_matrix = ctx.st_cell_type_matrix
            mapping_matrix = ctx.mapping_matrix
            print(f'st_cell_type_matrix: {st_cell_type_matrix.shape}')
            print(f'mapping_matrix: {mapping_matrix.shape}')

            weight_list = np.max(np.array(mapping_matrix), axis=0)

        else:
            _, new_sim_matrix = proto_generator.update_current_proto_matrix_with_sim(
                weight_list=weight_list,
                threshold=0,
                proto_latent_matrix=None,
                sim_matrix=mapping_matrix)

            print(f'new_sim_matrix:{new_sim_matrix.shape}')

            proto_exp_matrix, proto_cell_type_matrix = proto_generator.generate_rna_proto_matrix()
            print(f'proto shape:{proto_exp_matrix.shape}')

            cell_type_set_list = proto_generator.current_cell_type_set_list
            # new sim matrix相当于经过softmax后的输出，不能直接使用，需要反推为softmax的输入
            new_sim_matrix = softmax_to_logits_matrix(new_sim_matrix)
            # shape: (proto_number, st_number) <- shape:(st_number, proto_number)
            new_sim_matrix = new_sim_matrix.transpose()

            deconv_model.update_after_train(new_proto_gene_matrix=proto_exp_matrix,
                                                  new_proto_mapping_matrix=new_sim_matrix,
                                                  new_proto_cell_type_matrix=proto_cell_type_matrix)

            deconv_trainer.opt = optim.AdamW([
                                              {'params': deconv_model.gene_offset_parameter, 'lr':0.1},
                                              {'params': deconv_model.st_offset_parameter, 'lr':0.1},
                                              {'params':deconv_model.mapping_matrix, 'lr':0.1}],
                                              lr=lr)

            ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=None, random_seed=None)
            ctx.st_train_loader = DataLoader(st_rna_dataset, batch_size=batch_size, shuffle=True)
            ctx.pre_cell_type_matrix = torch.tensor(st_cell_type_matrix, dtype=torch.float32, device=device_name)
            ctx.reg_lambda = reg_lambda
            ctx.constrain_loss_list = []
            ctx.regular_loss_list = []
            ctx.rec_gene_loss_list = []
            deconv_trainer.train(ctx)
            constrain_loss_list = ctx.constrain_loss_list
            regular_loss_list = ctx.regular_loss_list
            rec_gene_loss_list = ctx.rec_gene_loss_list

            ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=None, random_seed=None)
            ctx.st_train_loader = DataLoader(st_rna_dataset, batch_size=batch_size, shuffle=False)
            deconv_trainer.deconv(ctx)

            st_cell_type_matrix = ctx.st_cell_type_matrix
            mapping_matrix = ctx.mapping_matrix
            weight_list = np.max(np.array(mapping_matrix), axis=0)

    new_df = pd.DataFrame(st_cell_type_matrix)
    new_df.columns = cell_type_set_list
    new_df.index = [f'X{i}' for i in range(st_rna_adata.shape[0])]
    new_df.to_csv(f'{save_path}/test_results.csv')


    sc.settings.figdir = save_path
    cell_type_number = len(cell_type_set_list)
    for i in range(cell_type_number):
        ct = cell_type_set_list[i]
        st_rna_adata.obs[ct] = st_cell_type_matrix[:, i]
        new_ct = ct.replace('/', '-')
        sc.pl.embedding(st_rna_adata, basis='spatial', color=[ct], title=f'{ct}', s=50, show=False,
                        save=f'test_rna_st_spatial_{new_ct}.png')

if __name__ == '__main__':

    # type_list = [3000, 6000, 10000]
    type_list = [3000]

    reg_lambda = [1e-2]
    resolution_list = [0.3]
    k_list = [9]
    for type in type_list:
        for reg in reg_lambda:
            for resol in resolution_list:
                for k in k_list:
                    save_path = f'seqFISH_{type}_simulating_reg{reg}_neighbor_{k}'
                    train_HIDF(save_path=save_path, reg_lambda=reg, type=type, resolution=resol, k=k)
                    st_rna_origin_adata = check_anndata(
                        f"../datasets/seqFISH/spatial/seqFISH_st{type}.h5ad",
                        False)
                    ct_list = st_rna_origin_adata.uns['cell_type'].tolist()
                    prd_result = pd.read_csv(
                        f'{save_path}/test_results.csv')
                    deconv_result = {}
                    total = 0
                    for ct in ct_list:
                        if ct in prd_result.columns.tolist():
                            predict_list = prd_result[ct].tolist()
                            true_list = st_rna_origin_adata.obs[ct].tolist()
                            rmse = RMSE(predict_list, true_list)
                            print(f'RMSE {ct}:{rmse}')
                            deconv_result[ct] = [rmse]
                            total += rmse
                            # jsd = JSD(predict_list, true_list)
                            # print(f'JSD {ct}:{jsd}')
                    print(total)
                    deconv_result_df = pd.DataFrame(deconv_result)
                    deconv_result_df.to_csv(
                        f'{save_path}/result.csv')