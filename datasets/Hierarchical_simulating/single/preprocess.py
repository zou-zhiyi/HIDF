from models.utils import check_anndata, write_to_h5ad
import scanpy as sc
from scipy import sparse
import pandas as pd

# all input file can be downloaded at GSE115746
def generate_h5ad():
    annotation_df = pd.read_csv("GSE115746/GSE115746_complete_metadata_28706-cells.csv/GSE115746_complete_metadata_28706-cells.csv")
    print(annotation_df)
    cell_type_map = {}
    sample_name_list = annotation_df['sample_name'].tolist()
    cell_class_list = annotation_df['cell_class'].tolist()
    cell_subclass_list = annotation_df['cell_subclass'].tolist()
    cell_cluster_list = annotation_df['cell_cluster'].tolist()
    source_name_list = annotation_df['source_name'].tolist()

    annotation_length = annotation_df.shape[0]
    for i in range(annotation_length):
        sample_name = sample_name_list[i]
        cell_class = cell_class_list[i]
        cell_subclass = cell_subclass_list[i]
        cell_cluster = cell_cluster_list[i]
        source_name = source_name_list[i]
        cell_type_map[sample_name] = [cell_class, cell_subclass, cell_cluster, source_name]

    exp_matrix_df = pd.read_csv("GSE115746/GSE115746_cells_exon_counts.csv/GSE115746_cells_exon_counts.csv", index_col=0)

    exp_matrix = exp_matrix_df.to_numpy()
    print(exp_matrix.shape)

    cell_obs_names = exp_matrix_df.columns.tolist()
    print(cell_obs_names)
    gene_names = exp_matrix_df.index.tolist()
    print(annotation_df.shape)

    new_cell_class_list, new_cell_subclass_list, new_cell_cluster_list= [], [], []
    new_source_name_list = []
    for obs_name in cell_obs_names:
        new_cell_class_list.append(cell_type_map[obs_name][0])
        new_cell_subclass_list.append(cell_type_map[obs_name][1])
        new_cell_cluster_list.append(cell_type_map[obs_name][2])
        new_source_name_list.append(cell_type_map[obs_name][3])

    adata = sc.AnnData(exp_matrix.transpose())
    adata.obs_names = cell_obs_names
    adata.var_names = gene_names
    adata.obs['cell_class'] = new_cell_class_list
    adata.obs['cell_subclass'] = new_cell_subclass_list
    adata.obs['cell_cluster'] = new_cell_cluster_list
    adata.obs['source_name'] = new_source_name_list

    write_to_h5ad(adata, 'sc_adata_with_nan.h5ad')

def filter_nan_data():
    adata = check_anndata('sc_adata_with_nan.h5ad', False)
    print(adata.shape)
    print(adata.obs['source_name'].unique().tolist())
    new_adata = adata[~adata.obs['cell_subclass'].isna(), :]
    new_adata = new_adata[new_adata.obs['cell_subclass']!='Low Quality', :]
    new_adata = new_adata[new_adata.obs['cell_subclass']!='Doublet', :]
    new_adata = new_adata[new_adata.obs['cell_subclass']!='Batch Grouping', :]
    new_adata = new_adata[new_adata.obs['cell_subclass']!='No Class', :]
    new_adata = new_adata[new_adata.obs['cell_subclass']!='High Intronic', :]
    new_adata = new_adata[new_adata.obs['source_name']=='Primary Visual Cortex (VISp)', :]
    print(new_adata.shape)
    adata.X = sparse.csr_matrix(adata.X)
    sc.pp.filter_genes(new_adata, min_cells=1)
    write_to_h5ad(new_adata, 'sc_adata.h5ad')

def filter_cells_for_simulating():
    adata = check_anndata('sc_adata.h5ad')
    print(adata.shape)
    print(adata.obs['cell_subclass'].unique().tolist())
    # return
    adata = adata[adata.obs['cell_subclass'].isin(['Oligo', 'L2/3 IT', 'L4', 'L5 IT','L6 IT', 'Astro']), :]
    print(adata.shape)
    print(adata.obs['cell_subclass'].unique().tolist())
    # return
    sc.pp.filter_genes(adata, min_cells=5)
    write_to_h5ad(adata, 'sc_adata.h5ad')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color = 'cell_subclass', save=True, show=False)

if __name__ == '__main__':
    generate_h5ad()
    filter_nan_data()
    filter_cells_for_simulating()
