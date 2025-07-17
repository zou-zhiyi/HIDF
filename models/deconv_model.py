import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, inputs, output, layer_number, layer_list, bias=False, dropout=0.1):
        super().__init__()
        self.input = inputs
        self.output = output
        self.input_layer = nn.Sequential(
            nn.Linear(self.input, layer_list[0], bias=bias),
            nn.BatchNorm1d(layer_list[0]),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential()
        for index in range(layer_number - 1):
            self.hidden_layer.extend([
                nn.Linear(layer_list[index], layer_list[index + 1], bias=bias),
                nn.BatchNorm1d(layer_list[index + 1]),
                nn.ReLU()
            ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], self.output, bias=bias)
        )

    def forward(self, x):
        inputs = self.input_layer(x)
        if len(self.hidden_layer)!=0:
            hidden = self.dropout(self.hidden_layer(inputs))
        else:
            hidden = inputs
        output = self.output_layer(hidden)
        return output

class HIDF(nn.Module):

    def __init__(self, proto_matrix, st_matrix, proto_number, st_number, proto_cell_type_matrix):
        super().__init__()
        m_matrix = torch.randn(size=(proto_number, st_number), dtype=torch.float32)
        self.mapping_matrix = nn.Parameter(m_matrix)
        self.proto_gene_matrix = nn.Parameter(torch.tensor(proto_matrix, dtype=torch.float32))
        self.st_gene_matrix = nn.Parameter(torch.tensor(st_matrix, dtype=torch.float32))

        self.proto_cell_type_matrix = torch.tensor(proto_cell_type_matrix, dtype=torch.float32)
        self.proto_cell_type_matrix = nn.Parameter(self.proto_cell_type_matrix)

        self.gene_number = proto_matrix.shape[1]
        self.input_dim = proto_matrix.shape[1]

        self.gene_offset_parameter = nn.Parameter(torch.ones(size=(1, self.gene_number), dtype=torch.float32))
        self.st_offset_parameter = nn.Parameter(torch.ones(size=(self.st_gene_matrix.shape[0], 1), dtype=torch.float32))

        self.moment_cell_type_mapping_matrix = nn.Parameter(torch.zeros(size=(st_number, proto_cell_type_matrix.shape[-1]),
                                                           dtype=torch.float32), requires_grad=False)

        # updated after each batch forward
        self.proto_gene_matrix.requires_grad = False
        self.st_gene_matrix.requires_grad = False
        self.proto_cell_type_matrix.requires_grad = False

        self.a = torch.nn.ReLU()

    def update_after_train(self, new_proto_gene_matrix, new_proto_mapping_matrix, new_proto_cell_type_matrix, device='cuda:0'):
        del self.mapping_matrix
        del self.proto_cell_type_matrix
        del self.proto_gene_matrix
        torch.cuda.empty_cache()
        self.mapping_matrix = nn.Parameter(torch.tensor(new_proto_mapping_matrix, dtype=torch.float32,
                                                        device=device), requires_grad=True)
        self.proto_cell_type_matrix = nn.Parameter(torch.tensor(new_proto_cell_type_matrix, dtype=torch.float32,
                                                                device=device))
        self.proto_gene_matrix = nn.Parameter(torch.tensor(new_proto_gene_matrix, dtype=torch.float32,
                                                                device=device))

        self.proto_cell_type_matrix.requires_grad = False
        self.proto_gene_matrix.requires_grad = False

    def update_memory_bank(self, st_index, tmp_st_cell_type_matrix):
        self.moment_cell_type_mapping_matrix[st_index, :] = tmp_st_cell_type_matrix

    def forward_sc_st(self, index):
        st_index = index
        index_mapping_matrix = self.mapping_matrix[:, st_index]
        mapping_matrix = torch.softmax(index_mapping_matrix, dim=0)
        mapping_matrix = torch.transpose(mapping_matrix, dim0=0, dim1=1)
        st_rec_gene_matrix = torch.matmul(mapping_matrix, self.proto_gene_matrix)
        st_gene_matrix = self.st_gene_matrix[st_index, :]
        cell_type_matrix = torch.mm(mapping_matrix, self.proto_cell_type_matrix)

        return mapping_matrix, st_rec_gene_matrix, st_gene_matrix, cell_type_matrix
