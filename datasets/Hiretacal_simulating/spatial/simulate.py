import os.path
from collections import Counter

import numpy as np
import pandas as pd
import scipy

from models.utils import check_anndata, write_to_h5ad
import scanpy as sc
import random

def select_cell(rna_adata: sc.AnnData, cell_number, cell_type_list, cell_radio):
    type_list = random.choices(cell_type_list, weights=cell_radio, k=cell_number)
    exp_rna = None
    i = 0
    type_Counter = Counter(type_list)
    for type, num in type_Counter.items():
        tmp_rna_adata = rna_adata[rna_adata.obs['cell_type'] == type, :]
        index_list = range(tmp_rna_adata.shape[0])
        index = random.sample(index_list, num)
        sample_rna = np.array(tmp_rna_adata[index, :].X.toarray())

        if exp_rna is None:
            exp_rna = sample_rna
        else:
            exp_rna = np.concatenate((exp_rna, sample_rna), axis=0)
        i += num
    return exp_rna, type_list


def pattern_one(rna_adata_path, save_path, spot_cell_number, sigma):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cell_type_list = ['Astro', 'L2/3 IT', 'L4', 'L5 IT','L6 IT', 'Oligo']
    rna_adata = check_anndata(rna_adata_path)
    rna_adata.obs['cell_type'] = rna_adata.obs['cell_subclass'].tolist()
    x_number, y_number = 30, 15

    def gassuian_distance(distance):
        return np.exp(-np.power(distance, 2) / (2 * np.power(sigma, 2))) / (sigma * np.sqrt(2 * np.pi))

    spatial_exp_rna, spatial_exp_atac = None, None
    x_list, y_list = [], []
    xy_list = []
    cell_type_radio = []
    for x in range(x_number):
        distance_0, distance_1, distance_2 = x - 2.5, x - 7.5, x - 12.5
        distance_3, distance_4, distance_5 = x - 17.5, x - 22.5, x - 27.5
        g_d_0, g_d_1, g_d_2, g_d_3, g_d_4, g_d_5 = \
            gassuian_distance(distance_0), gassuian_distance(distance_1), gassuian_distance(distance_2), gassuian_distance(distance_3), gassuian_distance(distance_4), gassuian_distance(distance_5)
        for y in range(y_number):
            exp_rna, type_list = select_cell(rna_adata, cell_number=spot_cell_number,
                                                       cell_type_list=cell_type_list,
                                                       cell_radio=[g_d_0, g_d_1, g_d_2, g_d_3, g_d_4, g_d_5])
            exp_rna = exp_rna.sum(axis=0, keepdims=True)
            if spatial_exp_rna is None:
                spatial_exp_rna = exp_rna
            else:
                spatial_exp_rna = np.concatenate((spatial_exp_rna, exp_rna), axis=0)
            cell_type_Counter = Counter(type_list)
            type_radio = []
            for type in cell_type_list:
                type_radio.append(cell_type_Counter[type]/spot_cell_number)
            cell_type_radio.append(type_radio)
            x_list.append(x)
            y_list.append(y)
            xy_list.append([x,y])
    cell_type_radio = np.array(cell_type_radio)
    xy = np.array(xy_list)
    print(f'spatial rna shape:{spatial_exp_rna.shape}')
    print(f'cell type radio shape:{cell_type_radio.shape}')
    print(f'xy shape:{xy.shape}')

    spatial_rna_adata = sc.AnnData(X=spatial_exp_rna)
    spatial_rna_adata.var_names = rna_adata.var_names.tolist()
    spatial_rna_adata.obsm['cell_radio'] = cell_type_radio
    spatial_rna_adata.obs['x'] = x_list
    spatial_rna_adata.obs['y'] = y_list
    spatial_rna_adata.obsm['spatial'] = xy


    spatial_rna_adata.uns['cell_type'] = {'cell_type_list':cell_type_list}

    write_to_h5ad(spatial_rna_adata, f'{save_path}/spatial_rna_pattern_1.h5ad')
    pass

def generate_simulated_spatial_data():
    pattern_type = 'pattern1'
    spot_cell_number_list = [5]
    sigma_list = [1]
    for spot_cell_number in spot_cell_number_list:
        for sigma in sigma_list:
            save_path = f'{pattern_type}_{spot_cell_number}_{sigma}'
            pattern_one(rna_adata_path=f'../single/sc_adata.h5ad',
                        save_path=save_path,
                        spot_cell_number=spot_cell_number, sigma=sigma)
            spatial_rna_adata = check_anndata(f'{save_path}/spatial_rna_pattern_1.h5ad')
            sc.pp.normalize_total(spatial_rna_adata)
            sc.pp.log1p(spatial_rna_adata)
            cell_type_list = ['Astro', 'L2/3 IT', 'L4', 'L5 IT','L6 IT', 'Oligo']
            sc.pp.neighbors(spatial_rna_adata)
            sc.tl.umap(spatial_rna_adata)
            sc.settings.figdir = save_path

            for i in range(len(cell_type_list)):
                spatial_rna_adata.obs[cell_type_list[i]] = spatial_rna_adata.obsm['cell_radio'][:, i]
                ct_name = cell_type_list[i]
                ct_name = ct_name.replace('/', '-')
                sc.pl.embedding(spatial_rna_adata, basis='spatial', color=cell_type_list[i],
                                s=70, show=False,
                                save=f'spatial rna pattern 1 {ct_name}.png')
                sc.pl.umap(spatial_rna_adata, color=cell_type_list[i], show=False,
                           save=f'spatial rna pattern 1 {ct_name}.png')

if __name__ == '__main__':
    generate_simulated_spatial_data()

    st_adata = check_anndata('pattern1_5_1/spatial_rna_pattern_1.h5ad')
    cell_type_list = st_adata.uns['cell_type']['cell_type_list'].tolist()
    total_GLUT = np.zeros(shape=(st_adata.shape[0],))
    for i in range(len(cell_type_list)):
        ct = cell_type_list[i]
        st_adata.obs[ct] = st_adata.obsm['cell_radio'][:, i]
        if ct != 'Astro' and ct != 'Oligo':
            total_GLUT += st_adata.obsm['cell_radio'][:, i]
    st_adata.obs['Glutamatergic'] = total_GLUT
