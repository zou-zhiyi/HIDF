import datetime
import glob
import random

import scipy
import seaborn as sns
import pickle
import os
from collections import Counter

import sklearn.metrics
from matplotlib.patches import Wedge
from scipy import spatial
from sklearn.cluster import KMeans
from typing import List

import scipy.stats as stats
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score, silhouette_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from torch import nn
import copy
import pandas as pd
import gc
from distinctipy import get_colors

def generate_distinct_colors(n_colors):

    colors = get_colors(n_colors)
    return colors

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


class WordIdxDic:

    def __init__(self):
        self.word2idx_dic = {}
        self.idx2word_dic = {}
        self.current_idx = 0

    def insert(self, gene):
        if gene in self.word2idx_dic.keys():
            return
        else:
            while self.current_idx in self.idx2word_dic.keys():
                self.current_idx += 1
            self.word2idx_dic[gene] = self.current_idx
            self.idx2word_dic[self.current_idx] = gene

    def getGene(self, idx):
        return self.idx2word_dic.get(idx, None)

    def getIdx(self, gene):
        return self.word2idx_dic.get(gene, None)


def write_file_to_pickle(data, save_path):
    with open(save_path, 'wb') as file_to_write:
        pickle.dump(data, file_to_write)


def read_file_from_pickle(save_path):
    with open(save_path, 'rb') as file_to_read:
        data = pickle.load(file_to_read)
    return data


def merger_gene_dic(adata: sc.AnnData, gene_idx_dic=None) -> WordIdxDic:
    if gene_idx_dic is None:
        gene_idx_dic = WordIdxDic()
    for gene in adata.var_names:
        gene_idx_dic.insert(gene.lower())
    if 'cell_type' in adata.obs.keys():
        for cell_type in set(adata.obs['cell_type']):
            gene_idx_dic.insert(str(cell_type).lower())
    if adata.uns.get('tissues', None) is not None:
        for tissue in adata.uns['tissues']:
            gene_idx_dic.insert(tissue.lower())
    return gene_idx_dic


def merger_gene_dic_from_varname(var_names, cell_types, gene_idx_dic=None) -> WordIdxDic:
    if gene_idx_dic is None:
        gene_idx_dic = WordIdxDic()
    for gene in var_names:
        if isinstance(gene, str):
            gene_idx_dic.insert(gene.lower())
        else:
            gene_idx_dic.insert(gene)
    for cell_type in set(cell_types):
        if isinstance(cell_type, str):
            gene_idx_dic.insert(cell_type.lower())
        else:
            gene_idx_dic.insert(cell_type)
    # if adata.uns.get('tissues', None) is not None:
    #     for tissue in adata.uns['tissues']:
    #         gene_idx_dic.insert(tissue.lower())
    return gene_idx_dic

def read_data_from_csv(data_path, cell_type_path, dataset_prefix, check_data=False):
    rna_data = pd.read_csv(data_path, header=None, low_memory=False)
    cell_type_data = pd.read_csv(cell_type_path, header=None, low_memory=False)
    adata = sc.AnnData(np.array(rna_data.iloc[1:, 1:].transpose(), dtype=np.float32))
    # print(adata.shape)
    # print(rna_data.iloc[1:, 0])
    # print(rna_data.iloc[1:, 0].shape)
    adata.var['gene_name'] = np.array(rna_data.iloc[1:, 0])
    adata.var_names = np.array(rna_data.iloc[1:, 0])
    adata.obs_names = np.array(rna_data.iloc[0, 1:])
    print(dataset_prefix)
    adata.obs['cell_name'] = np.array(rna_data.iloc[0, 1:] + dataset_prefix)
    # print(cell_type_data.iloc[:, 2])
    adata.obs['cell_type'] = np.array(cell_type_data.iloc[1:, 2])

    if check_data:
        check_anndata_direct(adata)
    return adata
    # cell_type_data = pd.read_csv(cell_type_path, header=None, low_memory=False)
    # check_anndata_direct(adata)
    # write_to_h5ad(adata, f'Bone_h5/{data_path}')

def topk_values_and_indices_per(matrix, k, axis=1):
    sorted_indices = np.argsort(matrix, axis=axis)
    if axis == 0:
        topk_indices = sorted_indices[-k:, :]
    else:
        topk_indices = sorted_indices[:, -k:]
    topk_indices = np.flip(topk_indices, axis=axis)

    # 提取对应的值
    topk_values = np.take_along_axis(matrix, topk_indices, axis=axis)

    return topk_values, topk_indices


def label_transform(adata: sc.AnnData, filepath):
    cell_type_set = set(adata.obs['cell_type'])
    cell_type_dic = {}
    cnt = 0
    for ct in cell_type_set:
        cell_type_dic[ct] = cnt
        cnt += 1
    adata.obs['cell_type_idx'] = adata.obs['cell_type'].map(lambda x: cell_type_dic[x])
    # adata.uns['cell_type_nums'] = cnt
    # adata.uns['cell_type_dic'] = cell_type_dic
    check_anndata_direct(adata)
    print(f"start to write: {filepath}")
    print(f'before gene nums:{len(adata.var_names)}')
    sc.pp.filter_genes(adata, min_cells=1)
    print(f'after gene nums:{len(adata.var_names)}')
    write_to_h5ad(adata, filepath)


def merge_files(file_prefixes, save_filename, tissue_name):
    cell_nums = 0
    gene_set = set()
    cell_type_set = set()
    total_adata = None
    if len(file_prefixes) == 0:
        return
    # print(data_files)
    for file_prefix in file_prefixes:
        print(file_prefix)
        data_files = glob.glob(f'{file_prefix}*_data.csv')
        cnt = 0
        for data_file in data_files:
            father_path = os.path.abspath((os.path.dirname(data_file)))
            # print(father_path)
            # print(data_file)

            cell_type_file = data_file.split(os.sep)[-1].split('_')
            cell_type_file[-1] = 'celltype.csv'
            cell_type_file = '_'.join(cell_type_file)
            print(f'{cell_type_file}')
            adata = read_data_from_csv(data_file, f'{father_path}{os.sep}{cell_type_file}', '_' + cell_type_file, True)
            cell_nums += len(adata.obs)
            gene_set.update(set(np.array(adata.var['gene_name'])))
            cell_type_set.update(set(np.array(adata.obs['cell_type'])))
            adata.obs['tissues'] = tissue_name
            adata.obs['batch_id'] = cnt
            cnt += 1
            if total_adata is None:
                total_adata = adata
            else:
                total_adata = total_adata.concatenate(adata, join='outer', fill_value=0, uns_merge="first")
            gc.collect()
        print(f'cell nums: {cell_nums}')
        print(f'gene set nums: {len(gene_set)}')
        print(f'cell type set nums: {len(cell_type_set)}')
        print(cell_type_set)
        if total_adata.uns.get('tissues', None) is None:
            total_adata.uns['tissues'] = [tissue_name]
        else:
            total_adata.uns['tissues'].append(tissue_name)
    sc.pp.highly_variable_genes(total_adata)
    label_transform(total_adata, save_filename)
    check_anndata_direct(total_adata)
    print(total_adata.obs["cell_type_idx"])
    print(set(total_adata.obs["cell_type_idx"]))
    return total_adata

def merge_multi_files(fileprefx, save_filename, tissue_name=None):
    if isinstance(fileprefx, list):
        return merge_files(fileprefx, save_filename, tissue_name)
    else:
        return merge_files([fileprefx], save_filename, tissue_name)

def loss_visual(loss_total, test_loss_total, save_path=''):
    plt.cla()
    plt.clf()
    y = loss_total
    print(loss_total)
    if not os.path.exists('loss'):
        os.mkdir('loss')
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    x = [i for i in range(len(test_loss_total))]
    plt.plot(x, test_loss_total)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    if not os.path.exists(f'{save_path}/loss'):
        os.makedirs(f'{save_path}/loss')
    plt.savefig(f'{save_path}/loss/loss_' + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg')
    # plt.show()
    print('plt saved')
    plt.close()


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#" + ''.join([random.choice(colorArr) for i in range(6)])
    return color


def randommarker():
    marker_list = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'D', 'P', 'X']
    marker = random.choice(marker_list)
    return marker


def umap_plot(data, label_name, save_file_name):
    plt.figure(figsize=(10, 10), dpi=300)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    # nan_error_value = adata.obs['cell_type_idx'].min()
    # max_type_value = adata.obs['cell_type_idx'].max()
    # adata.obs['cell_type_idx'][adata.obs['cell_type_idx'] == nan_error_value] = 0
    # target = adata.obs['cell_type_idx']
    # plt.subplots(300)
    # print(embbeding.shape)
    label_set = set(label_name.tolist())
    cnt = 0
    for l in label_set:
        tmp = (label_name == l)
        # print(tmp.shape)
        # print(tmp.sum())
        plt.scatter(embedding[tmp, 0], embedding[tmp, 1], marker='o', c=randomcolor(), s=5, label=l)
        cnt += 1
    plt.legend(loc="upper right", title="Classes")

    # legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    # ax.add_artist(legend)
    plt.savefig(save_file_name)
    plt.show()
    plt.close()


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def check_anndata_direct(data):
    print("Data matrix:")
    print(data.shape)
    print(data.X)
    print("=======================")
    print("Data obs:")
    print(data.obs)
    print("=======================")
    print("Data obs keys")
    print(data.obs.keys())
    print("=======================")
    print("Data var:")
    print(data.var)
    print("=======================")
    print("Data var keys")
    print(data.var.keys())
    print("=======================")
    print("Data uns:")
    print(data.uns)
    print("=======================")


def check_anndata(filepath, is_print=False):
    data = ad.read_h5ad(filepath)
    if is_print:
        print("Data matrix:")
        print(data.shape)
        print(data.X)
        print("=======================")
        print("Data obs:")
        print(data.obs)
        print("=======================")
        print("Data obs keys")
        print(data.obs.keys())
        print("=======================")
        print("Data var:")
        print(data.var)
        print("=======================")
        print("Data var keys")
        print(data.var.keys())
        print("=======================")
        print("Data uns:")
        print(data.uns)
        print("=======================")
    return data


def write_to_h5ad(anndata, filepath, copy=False):
    anndata.write_h5ad(filepath)
    if copy:
        anndata.write_h5ad(filepath + "_copy")

def stratify_split(total_data, random_seed=None, test_size=0.1, label_list=None):
    # print(Counter(total_data.obs['Cluster']))
    if label_list is not None:
        train_data, test_data = train_test_split(total_data, stratify=label_list, test_size=test_size,
                                                 random_state=random_seed)
    else:
        train_data, test_data = train_test_split(total_data, test_size=test_size, random_state=random_seed)
    # print(Counter(test_data.obs['Cluster']))
    return train_data, test_data


def calculate_score(true_label, pred_label):
    ari = adjusted_rand_score(true_label, pred_label)
    acc = accuracy_score(true_label, pred_label)
    # acc = precision_score(true_label,pred_label,average='macro')
    f1_scores_median = f1_score(true_label, pred_label, average=None)
    # print(f'f1 list: {f1_scores_median}')
    f1_scores_median = np.median(f1_scores_median)
    f1_scores_macro = f1_score(true_label, pred_label, average='macro')
    f1_scores_micro = f1_score(true_label, pred_label, average='micro')
    f1_scores_weighted = f1_score(true_label, pred_label, average='weighted')
    return acc, ari, f1_scores_median, f1_scores_macro, f1_scores_micro, f1_scores_weighted


def save_model(model, opt, model_path):
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, model_path)


def conver_adata_X_to_numpy(adata_X):
    if type(adata_X) == scipy.sparse.csr_matrix or type(adata_X) == scipy.sparse.csc_matrix:
        adata_X = adata_X.toarray()
    else:
        adata_X = np.array(adata_X)
    return adata_X

def set_seed(seed=None):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True



def drawPieMarker(xs, ys, ratios, sizes, colors, save_path, rasterize=False, show=False):
    """
    使用 Wedge 绘制饼图标记。

    参数：
    - xs: 每个点的 x 坐标（列表或数组）
    - ys: 每个点的 y 坐标（列表或数组）
    - ratios: 每个点的扇形比例（二维列表，每行和为 1）
    - sizes: 每个点的饼图半径（列表或数组）
    - colors: 扇形的颜色列表
    """
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    # 遍历每个点并绘制饼图标记
    for i in range(len(xs)):
        if i % 100 == 0:
            print(f"Drawing pie {i}/{len(xs)}")

        x, y = xs[i], ys[i]  # 当前点的坐标
        size = sizes  # 当前点的饼图半径
        rat = ratios[i]  # 当前点的扇形比例

        start_angle = 0  # 初始角度
        for color, ratio in zip(colors, rat):
            # 计算当前扇形的角度范围
            end_angle = start_angle + 360 * ratio

            # 创建 Wedge 对象并添加到坐标轴
            wedge = Wedge(
                (x, y),  # 圆心位置
                size,  # 半径
                start_angle,  # 起始角度
                end_angle,  # 结束角度
                facecolor=color,  # 填充颜色
                edgecolor='none',  # 边框颜色（无边框）
                rasterized=rasterize
            )
            ax.add_patch(wedge)

            # 更新起始角度
            start_angle = end_angle

    # 设置图形范围
    max_size = sizes
    ax.set_xlim(min(xs) - max_size*10, max(xs) + max_size*10)
    ax.set_ylim(min(ys) - max_size*10, max(ys) + max_size*10)
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def calculate_skewness_kurtosis(data):
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    return abs(skewness), kurtosis

def JSD(p, q, base=np.e):
    p, q = np.array(p), np.array(q)
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base) + 0.5 * scipy.stats.entropy(q, M, base)

def RMSE(p, q):
    p, q = np.array(p), np.array(q)
    return np.sqrt(((p - q) ** 2).mean())

def kl_divergence(p, q):
    # KL散度计算（处理0概率问题）
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def jsd_manual(p, q):
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2

def calculate_RMSE_JSD(save_path, times=5):

    cell_type_jsd_dic, cell_type_rmse_dic = {}, {}
    total_jsd, total_rmse = [], []
    for i in range(times):
        predict_csv = pd.read_csv(f'{save_path}/test_{i}.csv')
        true_csv = pd.read_csv(f'{save_path}/test_true_{i}.csv')
        for col in predict_csv.columns:
            if col == 'Unnamed: 0':
                continue
            p = predict_csv.loc[:, col]
            q = true_csv.loc[:, col]
            p = np.array(p)
            q = np.array(q)
            jsd, rmse = JSD(p, q), RMSE(p, q)
            if col not in cell_type_jsd_dic.keys():
                cell_type_jsd_dic[col] = []
            if col not in cell_type_rmse_dic.keys():
                cell_type_rmse_dic[col] = []
            cell_type_jsd_dic[col].append(jsd)
            cell_type_rmse_dic[col].append(rmse)
            total_jsd.append(jsd)
            total_rmse.append(rmse)
    # print(cell_type_jsd_dic)
    jsd_df = pd.DataFrame(cell_type_jsd_dic)
    rmse_df = pd.DataFrame(cell_type_rmse_dic)
    jsd_df.to_csv(f'{save_path}/test_jsd.csv')
    rmse_df.to_csv(f'{save_path}/test_rmse.csv')
    # print(f'cell type"{col}, jsd:{jsd}, rmse:{rmse}')
    return np.mean(total_jsd), np.mean(total_rmse)

def softmax_to_logits_matrix(softmax_matrix, reference_index=None):
    """
    对矩阵的每一行进行反推，从softmax输出反推一个可能的原始数据向量（logits）。

    参数:
        softmax_matrix (np.array): 一个二维矩阵，每一行是一个softmax输出。
        reference_index (int, optional): 选择一个参考索引。如果为None，则选择每行最大值的索引作为参考。

    返回:
        np.array: 一个二维矩阵，每一行是反推后的原始数据向量。
    """
    # 确保输入是二维矩阵
    assert softmax_matrix.ndim == 2, "输入必须是一个二维矩阵"

    # 获取矩阵的行数和列数
    rows, cols = softmax_matrix.shape

    # 初始化反推后的logits矩阵
    logits_matrix = np.zeros_like(softmax_matrix)

    # 遍历每一行进行反推
    for i in range(rows):
        # 如果没有指定参考索引，则选择当前行的最大值索引
        if reference_index is None:
            reference_index = np.argmax(softmax_matrix[i])

        # 计算对数差值
        logits_matrix[i] = np.log(softmax_matrix[i] + 1e-9) - np.log(softmax_matrix[i, reference_index] + 1e-9)

    return logits_matrix

def softmax_to_logits_multi_matrix(softmax_matrix, reference_index=None):
    """
    对矩阵的每一行进行反推，从softmax输出反推一个可能的原始数据向量（logits）。

    参数:
        softmax_matrix (np.array): 一个三维矩阵，最后行是一个softmax输出。
        reference_index (int, optional): 选择一个参考索引。如果为None，则选择每行最大值的索引作为参考。

    返回:
        np.array: 一个二维矩阵，每一行是反推后的原始数据向量。
    """
    # 确保输入是三维矩阵
    assert softmax_matrix.ndim == 3, "输入必须是一个三维矩阵"

    # 获取矩阵的行数和列数
    *batch_dims, C = softmax_matrix.shape

    # 确定参考索引
    if reference_index is None:
        # 对每个向量沿最后一个轴找到最大值索引
        ref_idx = np.argmax(softmax_matrix, axis=-1)
    else:
        if not (0 <= reference_index < C):
            raise ValueError(f"reference_index必须在0到{C - 1}之间")
        # 生成与batch维度匹配的参考索引数组
        ref_idx = np.full(batch_dims, reference_index, dtype=np.int64)

    # 使用take_along_axis获取参考值
    ref_values = np.take_along_axis(
        softmax_matrix,
        ref_idx[..., np.newaxis],
        axis=-1
    )

    # 计算logits（使用广播机制）
    logits = np.log(softmax_matrix + 1e-9) - np.log(ref_values + 1e-9)

    return logits
