from typing import Optional

from networkx.algorithms.structuralholes import constraint


from models.train import Trainer, Context
import torch.nn as nn
import numpy as np

import torch

def graph_regularization_loss(X, A):
    if A.is_sparse:
        # 稀疏矩阵处理
        indices = A.coalesce().indices()
        values = A.coalesce().values()
        rows, cols = indices[0], indices[1]

        X_rows = X[rows]
        X_cols = X[cols]
        squared_diff = (X_rows - X_cols).pow(2).sum(dim=1)

        reg = 0.5 * torch.sum(values * squared_diff)
    else:
        D = torch.sum(A, dim=1)
        norm_sq = torch.sum(X.pow(2), dim=1)
        term1 = torch.sum(D * norm_sq)

        XXT = torch.mm(X, X.t())
        term2 = torch.sum(XXT * A)

        reg = term1 - term2

    return reg


class HIDF_Trainer(Trainer):

    def __init__(self, model: nn.Module, train_dataset, test_dataset, device_name='cpu', lr=0.001,
                 weight_decay=1e-2, trained_model_path=None, continue_train=False, save_path=''):
        super().__init__(model, train_dataset, test_dataset, lr=lr, device_name=device_name,
                         weight_decay=weight_decay, trained_model_path=trained_model_path,
                         continue_train=continue_train, save_path=save_path)
        print(f'device :{self.device}')
        self.rec_loss = torch.nn.L1Loss(reduction='mean')
        self.constrain_loss = torch.nn.MSELoss(reduction='sum')

        self.sim_loss = torch.nn.CosineEmbeddingLoss()
        self.neighbor_loss = None

        self.sc_cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def train_inner(self, train_loader, context: Optional[Context]):

        st_cell_type_matrix = context.pre_cell_type_matrix
        st_train_loss_sum = []

        st_train_loader = context.st_train_loader
        sim_mask = None
        if hasattr(context, "sim_mask") and context.sim_mask is not None:
            sim_mask = context.sim_mask
            if not torch.is_tensor(sim_mask):
                sim_mask = torch.tensor(sim_mask, device=self.device, dtype=torch.long)
        reg_lambda = context.reg_lambda
        constrain_alpha = 1
        constrain_loss_list = []
        regular_loss_list = []
        rec_gene_loss_list = []

        for sampled_data in st_train_loader:
            self.opt.zero_grad()
            index, data, neighbor_index =sampled_data
            index = index.to(self.device)
            data = data.to(self.device)
            neighbor_index = neighbor_index.to(self.device)
            batch_size = data.shape[0]
            mapping_matrix, st_rec_gene_matrix, st_gene_matrix, cell_type_matrix \
                = self.model.forward_sc_st(index)

            if st_cell_type_matrix is None:
                batch_pre_st_cell_type = None
            else:
                batch_pre_st_cell_type = st_cell_type_matrix[index, :]

            st_gene_matrix = torch.log(st_gene_matrix + 1)
            tmp_st_offset_parameter = self.model.st_offset_parameter[index, :]
            st_rec_gene_matrix = torch.log(st_rec_gene_matrix + 1)
            st_rec_gene_matrix =  torch.nn.functional.softplus(self.model.gene_offset_parameter) + torch.nn.functional.softplus(tmp_st_offset_parameter) + st_rec_gene_matrix

            loss_flag_1 = torch.ones(st_rec_gene_matrix.shape[0], device=self.device, dtype=torch.float32)
            sim_loss_1 = self.sim_loss(st_gene_matrix, st_rec_gene_matrix, loss_flag_1)
            total_loss1 = sim_loss_1
            detach_cell_type_matrix = cell_type_matrix.detach()
            self.model.update_memory_bank(index, detach_cell_type_matrix)
            moment_cell_type_matrix = self.model.moment_cell_type_mapping_matrix[neighbor_index]
            tmp_cell_type_matrix = torch.unsqueeze(cell_type_matrix, dim=1)
            # 正则化
            squared_diff = (tmp_cell_type_matrix - moment_cell_type_matrix).pow(2).sum(dim=1)
            reg_loss =  torch.sum(squared_diff)
            reg_loss = reg_loss / batch_size
            reg_loss = reg_loss / moment_cell_type_matrix.shape[1]


            if batch_pre_st_cell_type is None:
                constraint_loss = None
            else:
                constraint_loss = self.constrain_loss(batch_pre_st_cell_type, cell_type_matrix) / batch_size

            if constraint_loss is not None:
                if context.c_epoch >= 0:
                    # print(f'RUN WITH REG')
                    total_loss = total_loss1 + constrain_alpha * constraint_loss + reg_lambda * reg_loss
                    constrain_loss_list.append(constraint_loss.item())
                    regular_loss_list.append(reg_loss.item())
                    rec_gene_loss_list.append(total_loss1.item())
                else:
                    total_loss = total_loss1 + constrain_alpha * constraint_loss
                    constrain_loss_list.append(constraint_loss.item())
                    rec_gene_loss_list.append(total_loss1.item())
            else:
                if context.c_epoch >= 0:
                    # print(f'RUN WITH REG')
                    total_loss = total_loss1 + reg_lambda * reg_loss
                    regular_loss_list.append(reg_loss.item())
                    rec_gene_loss_list.append(total_loss1.item())
                else:
                    total_loss = total_loss1
                    rec_gene_loss_list.append(total_loss1.item())

            total_loss = total_loss.to(torch.float32)
            total_loss.backward()
            self.opt.step()
            st_train_loss_sum.append(total_loss.item())
        epoch_loss = np.mean(st_train_loss_sum)
        context.epoch_loss = epoch_loss
        context.constrain_loss_list.append(np.mean(constrain_loss_list))
        context.regular_loss_list.append(np.mean(regular_loss_list))
        context.rec_gene_loss_list.append(np.mean(rec_gene_loss_list))

    @torch.no_grad()
    def deconv(self, context: Optional[Context]):
        self.model.eval()
        target_st_cell_type_matrix = None
        target_mapping_matrix = None
        target_rec_st_gene_matrix = None

        st_train_loader = context.st_train_loader
        for sampled_data in st_train_loader:
            index, data, neighbor_index =sampled_data
            index = index.to(self.device)
            data = data.to(self.device)
            batch_size = data.shape[0]
            mapping_matrix, st_rec_gene_matrix, st_gene_matrix, deconv_cell_type\
                = self.model.forward_sc_st(index)
            deconv_cell_type = deconv_cell_type / torch.sum(deconv_cell_type, dim=1, keepdim=True)
            tmp_st_offset_parameter = self.model.st_offset_parameter[index, :]
            st_rec_gene_matrix = torch.log(st_rec_gene_matrix + 1)
            st_rec_gene_matrix = torch.nn.functional.softplus(
                self.model.gene_offset_parameter) + torch.nn.functional.softplus(
                tmp_st_offset_parameter) + st_rec_gene_matrix
            if target_st_cell_type_matrix is not None:
                target_st_cell_type_matrix = np.concatenate(
                    (target_st_cell_type_matrix, deconv_cell_type.detach().cpu().numpy()), axis=0)
                target_mapping_matrix = np.concatenate(
                    (target_mapping_matrix, mapping_matrix.detach().cpu().numpy()), axis=0)
                target_rec_st_gene_matrix = np.concatenate(
                    (target_rec_st_gene_matrix, st_rec_gene_matrix.detach().cpu().numpy()), axis=0)
            else:
                target_st_cell_type_matrix = deconv_cell_type.detach().cpu().numpy()
                target_mapping_matrix = mapping_matrix.detach().cpu().numpy()
                target_rec_st_gene_matrix = st_rec_gene_matrix.detach().cpu().numpy()

            print(f'target mapping matrix shape:{target_mapping_matrix.shape}')
        context.st_cell_type_matrix = target_st_cell_type_matrix
        context.mapping_matrix = target_mapping_matrix
        context.st_rec_gene_matrix = target_rec_st_gene_matrix



