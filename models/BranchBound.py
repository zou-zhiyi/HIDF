import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from graphviz import Digraph, Graph

from models.utils import check_anndata, conver_adata_X_to_numpy, read_file_from_pickle, write_file_to_pickle
import scanpy as sc
import os


class ProtoNode:

    def __init__(self, node_type, node_name_list, node_type_list, child_node_list, isLeaf):
        self.node_name_list = node_name_list
        self.node_type_list = node_type_list
        self.node_type = node_type
        # self.child_node_type_list = child_node_type_list
        self.child_node_list = child_node_list
        self.isLeaf = isLeaf
        self.weight = 0


class ProtoTree:

    def __init__(self, proto_root: ProtoNode):
        self.root = proto_root
        self.current_division_node = []
        self.current_node_name_list = self.root.node_name_list
        self.current_node_type_map_list = []
        child_node_type_list = []
        for child_node in self.root.child_node_list:
            self.current_division_node.append(child_node)
            child_node_type_list.append(child_node.node_type)

        self.current_node_type_map_list.append({'root':child_node_type_list})

        self.current_node_type_list = []
        for child_node in self.current_division_node:
            self.current_node_type_list.extend([child_node.node_type for _ in range(len(child_node.node_type_list))])


    def is_all_leaf(self):
        leaf_flag = True
        for child_node in self.current_division_node:
            leaf_flag = leaf_flag & child_node.isLeaf
        return leaf_flag

    def division(self, threshold=0.001):
        next_division_node = []
        next_node_name_list, next_node_type_list = [], []
        next_proto_type_map_list = []
        for child_node in self.current_division_node:
            current_child_node_weight = child_node.weight
            if current_child_node_weight > threshold:
                # 符合阈值，节点分裂
                next_node_name_list.extend(child_node.node_name_list)
                next_node_type_list.extend(child_node.node_type_list)
                if child_node.isLeaf:
                    # 已是叶子节点，添加自身
                    next_division_node.append(child_node)
                    node_map_dic = {child_node.node_type: [child_node.node_type]}
                    next_proto_type_map_list.append(node_map_dic)
                else:
                    # 节点分裂，加入其子节点信息
                    next_division_node.extend(child_node.child_node_list)
                    node_map_dic = {child_node.node_type: [x.node_type for x in child_node.child_node_list]}
                    next_proto_type_map_list.append(node_map_dic)
            # else:
            #     if child_node.isLeaf:
            #         # 叶子节点不被剔除
            #         next_node_name_list.extend(child_node.node_name_list)
            #         next_node_type_list.extend(child_node.node_type_list)
            #
            #         next_division_node.append(child_node)
            #         node_map_dic = {child_node.node_type: [child_node.node_type]}
            #         next_proto_type_map_list.append(node_map_dic)
        self.current_division_node = next_division_node
        self.current_node_name_list = next_node_name_list
        self.current_node_type_map_list = next_proto_type_map_list

        self.current_node_type_list = []
        for child_node in self.current_division_node:
            self.current_node_type_list.extend([child_node.node_type for _ in range(len(child_node.node_type_list))])

        return next_proto_type_map_list

    def update_weight(self, weight_list, node_type_list):
        current_node_number = len(self.current_division_node)
        idx_map = {}
        for i1 in range(len(node_type_list)):
            idx_map[node_type_list[i1]] = i1

        for i in range(current_node_number):
            node_type = self.current_division_node[i].node_type
            idx = idx_map[node_type]
            current_node = self.current_division_node[i]
            current_node.weight = weight_list[idx]




class Proto_Generator:
    '''
    根据给定的anndata数据，生成层次化的聚类结果，并给出每次聚类结果的原型表达矩阵和类型映射矩阵
    '''

    def __init__(self, sc_rna_adata: sc.AnnData, sc_origin_rna_adata:sc.AnnData,
                 sc_omics_adata: sc.AnnData, resolution=1.0, cell_type_key='cell_type',
                 cluster_number_threshold=50):
        self.sc_rna_adata = sc_rna_adata.copy()
        if sc_omics_adata is None:
            root = generate_tree(sc_origin_rna_adata.copy(), cell_type_key, resolution=resolution,
                                 cluster_number_threshold=cluster_number_threshold)
        else:
            self.sc_omics_adata = sc_omics_adata.copy()
            root = generate_tree_multiomics(sc_origin_rna_adata.copy(), sc_omics_adata.copy(), cell_type_key,
                                            resolution=resolution, cluster_number_threshold=cluster_number_threshold)

        self.root = root
        self.cell_type_key = cell_type_key
        self.tree = ProtoTree(proto_root=root)

        self.current_proto_type_list = self.tree.current_node_type_list
        # self.current_proto_type_set_list = self.current_proto_type_list
        self.current_proto_type_set_list = list(set(self.current_proto_type_list))
        self.current_proto_type_set_list.sort()
        self.current_proto_type_number = len(self.current_proto_type_set_list)
        self.current_cell_obs_name_list = self.tree.current_node_name_list
        self.current_cell_number = len(self.tree.current_node_name_list)
        rna_data = self.sc_rna_adata[self.current_cell_obs_name_list, :]
        self.current_cell_type_set_list = rna_data.obs[self.cell_type_key].unique().tolist()
        self.current_cell_type_number = len(self.current_cell_type_set_list)



        # self.current_proto_exp_matrix, self.current_proto_cell_type_matrix = self.generate_proto_matrix()

    def calculate_depth(self):
        return caculate_depth(self.root)

    def is_all_leaf(self):
        return self.tree.is_all_leaf()

    def update_current_proto_matrix(self, weight_list, threshold, proto_latent_matrix):
        self.tree.update_weight(weight_list, self.current_proto_type_set_list)
        next_proto_type_map_list = self.tree.division(threshold)

        pre_proto_type_set_list = self.current_proto_type_set_list

        self.current_proto_type_list = self.tree.current_node_type_list
        # self.current_proto_type_set_list = self.current_proto_type_list
        self.current_proto_type_set_list = list(set(self.current_proto_type_list))
        self.current_proto_type_set_list.sort()
        self.current_proto_type_number = len(self.current_proto_type_set_list)
        self.current_cell_obs_name_list = self.tree.current_node_name_list
        self.current_cell_number = len(self.tree.current_node_name_list)
        rna_data = self.sc_rna_adata[self.current_cell_obs_name_list, :]
        self.current_cell_type_set_list = rna_data.obs[self.cell_type_key].unique().tolist()
        self.current_cell_type_number = len(self.current_cell_type_set_list)

        if proto_latent_matrix is None:
            return None

        new_proto_latent_matrix = np.zeros(shape=(len(self.current_proto_type_set_list), proto_latent_matrix.shape[1]))
        for proto_type_map in next_proto_type_map_list:
            for k, v_list in proto_type_map.items():
                index1 = pre_proto_type_set_list.index(k)
                for v in v_list:
                    index2 = self.current_proto_type_set_list.index(v)
                    new_proto_latent_matrix[index2, :] = proto_latent_matrix[index1, :]
        return new_proto_latent_matrix

    def update_current_proto_matrix_with_sim(self, weight_list, threshold, proto_latent_matrix, sim_matrix):
        # proto_latent_matrix:(proto_number, proto_dim)
        # sim_matrix:(st_number, proto_number)
        self.tree.update_weight(weight_list, self.current_proto_type_set_list)
        next_proto_type_map_list = self.tree.division(threshold)

        pre_proto_type_set_map = {}
        for i in range(len(self.current_proto_type_set_list)):
            pre_proto_type_set_map[self.current_proto_type_set_list[i]] = i

        self.current_proto_type_list = self.tree.current_node_type_list
        # self.current_proto_type_set_list = self.current_proto_type_list
        self.current_proto_type_set_list = list(set(self.current_proto_type_list))
        self.current_proto_type_set_list.sort()
        current_proto_type_set_map = {}
        for i in range(len(self.current_proto_type_set_list)):
            current_proto_type_set_map[self.current_proto_type_set_list[i]] = i

        self.current_proto_type_number = len(self.current_proto_type_set_list)
        self.current_cell_obs_name_list = self.tree.current_node_name_list
        self.current_cell_number = len(self.tree.current_node_name_list)
        rna_data = self.sc_rna_adata[self.current_cell_obs_name_list, :]
        self.current_cell_type_set_list = rna_data.obs[self.cell_type_key].unique().tolist()
        self.current_cell_type_number = len(self.current_cell_type_set_list)

        if proto_latent_matrix is None:
            new_proto_latent_matrix = None
        else:
            new_proto_latent_matrix = np.zeros(shape=(len(self.current_proto_type_set_list), proto_latent_matrix.shape[1]), dtype=np.float32)
            for proto_type_map in next_proto_type_map_list:
                for k, v_list in proto_type_map.items():
                    index1 = pre_proto_type_set_map[k]
                    for v in v_list:
                        index2 = current_proto_type_set_map[v]
                        new_proto_latent_matrix[index2, :] = proto_latent_matrix[index1, :]

        new_sim_matrix = np.zeros(shape=(sim_matrix.shape[0], len(self.current_proto_type_set_list)), dtype=np.float32)
        for proto_type_map in next_proto_type_map_list:
            for k, v_list in proto_type_map.items():
                index1 = pre_proto_type_set_map[k]
                next_proto_number = len(v_list)
                for v in v_list:
                    index2 = current_proto_type_set_map[v]
                    tmp_sim_data = sim_matrix[:, index1]
                    # tmp_sim_data = tmp_sim_data / next_proto_number
                    new_sim_matrix[:, index2] = tmp_sim_data

        return new_proto_latent_matrix, new_sim_matrix

    def update_current_proto_matrix_with_multi_sim(self, weight_list, threshold, proto_latent_matrix, sim_matrix):
        # proto_latent_matrix:(proto_number, proto_dim)
        # sim_matrix:(st_number, proto_number)
        self.tree.update_weight(weight_list, self.current_proto_type_set_list)
        next_proto_type_map_list = self.tree.division(threshold)

        pre_proto_type_set_list = self.current_proto_type_set_list

        self.current_proto_type_list = self.tree.current_node_type_list
        # self.current_proto_type_set_list = self.current_proto_type_list
        self.current_proto_type_set_list = list(set(self.current_proto_type_list))
        self.current_proto_type_set_list.sort()

        self.current_proto_type_number = len(self.current_proto_type_set_list)
        self.current_cell_obs_name_list = self.tree.current_node_name_list
        self.current_cell_number = len(self.tree.current_node_name_list)
        rna_data = self.sc_rna_adata[self.current_cell_obs_name_list, :]
        self.current_cell_type_set_list = rna_data.obs[self.cell_type_key].unique().tolist()
        self.current_cell_type_number = len(self.current_cell_type_set_list)

        if proto_latent_matrix is None:
            new_proto_latent_matrix = None
        else:
            new_proto_latent_matrix = np.zeros(shape=(len(self.current_proto_type_set_list), proto_latent_matrix.shape[1]), dtype=np.float32)
            for proto_type_map in next_proto_type_map_list:
                for k, v_list in proto_type_map.items():
                    index1 = pre_proto_type_set_list.index(k)
                    for v in v_list:
                        index2 = self.current_proto_type_set_list.index(v)
                        new_proto_latent_matrix[index2, :] = proto_latent_matrix[index1, :]

        new_sim_matrix = np.zeros(shape=(sim_matrix.shape[0], sim_matrix.shape[1], len(self.current_proto_type_set_list)), dtype=np.float32)
        for proto_type_map in next_proto_type_map_list:
            for k, v_list in proto_type_map.items():
                index1 = pre_proto_type_set_list.index(k)
                next_proto_number = len(v_list)
                for v in v_list:
                    index2 = self.current_proto_type_set_list.index(v)
                    tmp_sim_data = sim_matrix[:, :, index1]
                    # tmp_sim_data = tmp_sim_data / next_proto_number
                    new_sim_matrix[:, :, index2] = tmp_sim_data

        return new_proto_latent_matrix, new_sim_matrix

    def generate_rna_proto_matrix(self):
        rna_adata = self.sc_rna_adata[self.current_cell_obs_name_list, :].copy()
        rna_adata.obs['proto_type'] = self.current_proto_type_list
        rna_mtx = conver_adata_X_to_numpy(rna_adata.X)

        # shape(proto_type_number, rna_gene)
        proto_exp_matrix = np.zeros(shape=(self.current_proto_type_number, rna_mtx.shape[1]), dtype=np.float32)
        # shape(proto_type_number, cell_type)
        proto_cell_type_matrix = np.zeros(shape=(self.current_proto_type_number, self.current_cell_type_number), dtype=np.float32)
        for i in range(len(self.current_proto_type_set_list)):
            proto_type = self.current_proto_type_set_list[i]
            tmp_rna_mtx = np.mean(rna_mtx[rna_adata.obs['proto_type'] == proto_type, :], axis=0)
            proto_exp_matrix[i, :] = tmp_rna_mtx

            # cell_obs_names = self.current_cell_obs_name_list[i]
            # proto_cell_type = rna_adata[cell_obs_names, :].obs['cell_type'].tolist()[0]
            proto_cell_type = proto_type.split('_')[0]
            # tmp_rna_mtx2 = np.mean(self.sc_rna_adata.X[self.sc_rna_adata.obs['cell_type'] == proto_cell_type, :], axis=0)
            # print(rna_mtx[rna_adata.obs['proto_type'] == proto_type, :].shape)
            # print(self.sc_rna_adata[self.sc_rna_adata.obs['cell_type'] == proto_cell_type, :].shape)

            cell_type_index = self.current_cell_type_set_list.index(proto_cell_type)
            proto_cell_type_matrix[i, cell_type_index] = 1


        return proto_exp_matrix, proto_cell_type_matrix

    def generate_rna_omics_proto_matrix(self):
        rna_adata = self.sc_rna_adata[self.current_cell_obs_name_list].copy()
        rna_adata.obs['proto_type'] = self.current_proto_type_list
        rna_mtx = conver_adata_X_to_numpy(rna_adata.X)
        omics_adata = self.sc_omics_adata[self.current_cell_obs_name_list].copy()
        omics_adata.obs['proto_type'] = self.current_proto_type_list
        omics_mtx = conver_adata_X_to_numpy(omics_adata.X)

        # shape(proto_type_number, rna_gene)
        proto_rna_exp_matrix = np.zeros(shape=(self.current_proto_type_number, rna_mtx.shape[1]), dtype=np.float32)
        proto_omics_exp_matrix = np.zeros(shape=(self.current_proto_type_number, omics_mtx.shape[1]), dtype=np.float32)
        proto_cell_type_matrix = np.zeros(shape=(self.current_proto_type_number, self.current_cell_type_number), dtype=np.int)
        for i in range(len(self.current_proto_type_set_list)):
            proto_type = self.current_proto_type_set_list[i]
            tmp_rna_mtx = np.mean(rna_mtx[rna_adata.obs['proto_type'] == proto_type, :], axis=0)
            proto_rna_exp_matrix[i, :] = tmp_rna_mtx

            tmp_omics_mtx = np.mean(omics_mtx[omics_adata.obs['proto_type'] == proto_type, :], axis=0)
            proto_omics_exp_matrix[i, :] = tmp_omics_mtx

            # cell_obs_names = self.current_cell_obs_name_list[i]
            # proto_cell_type = rna_adata[cell_obs_names, :].obs['cell_type'].tolist()[0]
            proto_cell_type = proto_type.split('_')[0]

            cell_type_index = self.current_cell_type_set_list.index(proto_cell_type)
            proto_cell_type_matrix[i, cell_type_index] = 1

        return proto_rna_exp_matrix, proto_omics_exp_matrix, proto_cell_type_matrix


def recursion_leiden(adata: sc.AnnData, resolution, prefix_key, cluster_number_threshold=50):
    number = adata.shape[0]
    # print(f'current node:{prefix_key}, current number:{number}')
    if number <= cluster_number_threshold:
        leiden_list = [i for i in range(number)]
        new_leiden = [f"{prefix_key}_{l}" for l in leiden_list]
        child_node_list = []
        leaf_node_obs_names_list = adata.obs_names.tolist()
        for i in range(number):
            leaf_node = ProtoNode(node_type=f"{prefix_key}_{i}", node_name_list=[leaf_node_obs_names_list[i]],
                                  node_type_list=[f"{prefix_key}_{i}"], child_node_list=None, isLeaf=True)
            child_node_list.append(leaf_node)
        node = ProtoNode(node_type=prefix_key, node_name_list=leaf_node_obs_names_list,
                         node_type_list=new_leiden,
                         child_node_list=child_node_list, isLeaf=False)
        # node = {'node_name': prefix_key, 'current': new_leiden,'obs_name': adata.obs_names.tolist(), 'isLeaf':True}
        return node
    new_adata = adata

    # neigh = NearestNeighbors(n_neighbors=16, metric='euclidean')
    # rna_array = conver_adata_X_to_numpy(new_adata.X)
    # rna_array = normalize(rna_array, norm='l2')

    # neigh.fit(rna_array)
    # A = neigh.kneighbors_graph(rna_array)
    # A[A != 0] = 1
    # sc.tl.leiden(new_adata, adjacency=A, resolution=resolution)
    # sc.pp.log1p(new_adata)
    sc.pp.neighbors(new_adata)
    sc.tl.leiden(new_adata, resolution=resolution)

    leiden_list = new_adata.obs['leiden'].tolist()
    new_leiden = [f"{prefix_key}_{l}" for l in leiden_list]
    new_adata.obs['new_leiden'] = new_leiden
    leiden_set = new_adata.obs['new_leiden'].unique().tolist()

    if len(leiden_set) == 1:
        leiden_list = [i for i in range(number)]
        new_leiden = [f"{prefix_key}_{l}" for l in leiden_list]
        child_node_list = []
        leaf_node_obs_names_list = adata.obs_names.tolist()
        for i in range(number):
            leaf_node = ProtoNode(node_type=f"{prefix_key}_{i}", node_name_list=[leaf_node_obs_names_list[i]],
                                  node_type_list=[f"{prefix_key}_{i}"], child_node_list=None, isLeaf=True)
            child_node_list.append(leaf_node)
        node = ProtoNode(node_type=prefix_key, node_name_list=leaf_node_obs_names_list,
                         node_type_list=new_leiden,
                         child_node_list=child_node_list, isLeaf=False)
        # node = {'node_name': prefix_key, 'current': new_leiden,'obs_name': adata.obs_names.tolist(), 'isLeaf':True}
        return node

    node = ProtoNode(node_type=prefix_key, node_name_list=[], node_type_list=[],
                     child_node_list=[], isLeaf=False)
    # node = {'parent_name': prefix_key, 'current': new_leiden, 'isLeaf':False}


    for l in leiden_set:
        tmp_adata = new_adata[new_adata.obs['new_leiden'] == l, :].copy()
        child_node = recursion_leiden(tmp_adata, resolution, l, cluster_number_threshold)
        node.node_name_list.extend(child_node.node_name_list)
        node.node_type_list.extend(child_node.node_type_list)
        node.child_node_list.append(child_node)

    return node


def caculate_depth(node: ProtoNode):
    max_number = 0
    if node.isLeaf:
        return max_number
    child_node_list = node.child_node_list
    for child_node in child_node_list:
        number = caculate_depth(child_node)
        max_number = max(number, max_number)
    return max_number + 1





def recursion_leiden_multiomics(rna_adata: sc.AnnData, omics_adata: sc.AnnData, resolution, prefix_key):
    number = rna_adata.shape[0]
    print(f'current node:{prefix_key}, current number:{number}')
    if number <= 50:
        leiden_list = [i for i in range(number)]
        new_leiden = [f"{prefix_key}_{l}" for l in leiden_list]
        child_node_list = []
        leaf_node_obs_names_list = rna_adata.obs_names.tolist()
        for i in range(number):
            leaf_node = ProtoNode(node_type=f"{prefix_key}_{i}", node_name_list=[leaf_node_obs_names_list[i]],
                                  node_type_list=[f"{prefix_key}_{i}"], child_node_list=None, isLeaf=True)
            child_node_list.append(leaf_node)
        node = ProtoNode(node_type=prefix_key, node_name_list=leaf_node_obs_names_list,
                         node_type_list=new_leiden,
                         child_node_list=child_node_list, isLeaf=False)
        return node
    new_rna_adata, new_omics_adata = rna_adata, omics_adata


    neigh = NearestNeighbors(n_neighbors=16, metric='euclidean')
    rna_array = conver_adata_X_to_numpy(new_rna_adata.X)

    # rna_array = normalize(rna_array, norm='l2')
    neigh.fit(rna_array)
    A_rna = neigh.kneighbors_graph(rna_array)

    neigh = NearestNeighbors(n_neighbors=16, metric='euclidean')

    omics_array = conver_adata_X_to_numpy(new_omics_adata.X)

    # omics_array = normalize(omics_array, norm='l2')
    neigh.fit(omics_array)
    A_omics = neigh.kneighbors_graph(omics_array)

    A = A_omics + A_rna
    A[A != 0] = 1

    sc.tl.leiden(new_rna_adata, adjacency=A, resolution=resolution)
    leiden_list = new_rna_adata.obs['leiden'].tolist()
    new_leiden = [f"{prefix_key}_{l}" for l in leiden_list]
    new_rna_adata.obs['new_leiden'] = new_leiden
    new_omics_adata.obs['new_leiden'] = new_leiden
    leiden_set = new_rna_adata.obs['new_leiden'].unique().tolist()
    node = ProtoNode(node_type=prefix_key, node_name_list=[], node_type_list=[],
                     child_node_list=[], isLeaf=False)
    for l in leiden_set:
        tmp_rna_adata = new_rna_adata[new_rna_adata.obs['new_leiden'] == l, :].copy()
        tmp_omics_adata = new_omics_adata[new_omics_adata.obs['new_leiden'] == l, :].copy()
        child_node = recursion_leiden_multiomics(tmp_rna_adata, tmp_omics_adata, resolution, l)
        node.node_name_list.extend(child_node.node_name_list)
        node.node_type_list.extend(child_node.node_type_list)
        node.child_node_list.append(child_node)

    return node

def generate_tree(sc_rna_adata, cell_type_key='cell_type', resolution=0.3, cluster_number_threshold=50):
    # sc.pp.normalize_total(sc_rna_adata)
    # sc.pp.log1p(sc_rna_adata)
    # sc.pp.neighbors(sc_rna_adata)

    cell_type_list = sc_rna_adata.obs[cell_type_key].unique().tolist()

    # root = {'current':cell_type_list, 'isLeaf':False}
    root = ProtoNode(node_type='root', node_name_list=[], node_type_list=[], child_node_list=[], isLeaf=False)
    for ct in cell_type_list:
        tmp_adata = sc_rna_adata[sc_rna_adata.obs[cell_type_key] == ct, :].copy()
        node = recursion_leiden(tmp_adata, resolution=resolution, prefix_key=ct, cluster_number_threshold=cluster_number_threshold)
        root.node_name_list.extend(node.node_name_list)
        root.node_type_list.extend(node.node_type_list)
        root.child_node_list.append(node)

    # print(caculate_depth(root))
    return root

def generate_tree_multiomics(rna_adata, omics_adata, cell_type_key='cell_type', resolution=1.0):
    sc_rna_adata = rna_adata.copy()
    sc_omics_adata = omics_adata.copy()

    # sc.pp.normalize_total(sc_rna_adata)
    # sc.pp.log1p(sc_rna_adata)
    # sc.pp.normalize_total(sc_omics_adata)
    # sc.pp.log1p(sc_omics_adata)

    # sc.pp.neighbors(sc_rna_adata)
    # sc.pp.neighbors(sc_omics_adata)

    cell_type_list = sc_rna_adata.obs[cell_type_key].unique().tolist()

    root = ProtoNode(node_type='root', node_name_list=[], node_type_list=[], child_node_list=[], isLeaf=False)
    for ct in cell_type_list:
        tmp_rna_adata = sc_rna_adata[sc_rna_adata.obs[cell_type_key] == ct, :].copy()
        tmp_omics_adata = sc_omics_adata[sc_omics_adata.obs[cell_type_key] == ct, :].copy()
        node = recursion_leiden_multiomics(tmp_rna_adata, tmp_omics_adata, resolution=resolution, prefix_key=ct)
        root.node_name_list.extend(node.node_name_list)
        root.node_type_list.extend(node.node_type_list)
        root.child_node_list.append(node)

    return root

def generate_proto_generator(sc_rna_adata, sc_origin_rna_adata, sc_omics_adata, save_path, resolution=0.3, save_bool=False,
                             cell_type_key='cell_type'):
    if os.path.exists(f"{save_path}/proto_generator.pk"):
        proto_generator = read_file_from_pickle(f"{save_path}/proto_generator.pk")
    else:
        proto_generator = Proto_Generator(sc_rna_adata=sc_rna_adata,sc_origin_rna_adata=sc_origin_rna_adata,
                                          sc_omics_adata=sc_omics_adata,
                                          resolution=resolution,cell_type_key=cell_type_key)
        if save_bool:
            write_file_to_pickle(proto_generator, save_path=f"{save_path}/proto_generator.pk")
    return proto_generator

def proto_tree_visualization(root:ProtoNode):
    g = Graph(engine='dot',
              node_attr={'shape':'egg'})
    tree_viusalization(g, root)
    g.view()

def tree_viusalization(g:Graph, node:ProtoNode):
    if node.isLeaf:
        return
    current_node_type = node.node_type
    child_node_list = node.child_node_list
    for child in child_node_list:
        if child.weight != 0:
            g.edge(current_node_type, child.node_type, label=str(child.weight))
            tree_viusalization(g, child)
