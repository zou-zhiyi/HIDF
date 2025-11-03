
# This code is used for subclass analysis

class AnalysisTreeNode:

    def __init__(self, node_name, weight_matrix, is_leaf=False):
        self.is_leaf = is_leaf
        self.parent_node = None
        self.weight_matrix = weight_matrix
        self.node_name = node_name
        self.child_node_dic = {}

    def insert_child_node(self, node):
        self.child_node_dic[node.node_name] = node

    def calculate_weights(self):
        if self.is_leaf:
            # print(self.node_name)
            # print(self.weight_matrix)
            # print(self.weight_matrix.shape)
            return self.weight_matrix.copy()
        # cnter = 0
        total_weights = None
        for node_name, child_node in self.child_node_dic.items():
            # cnter += 1
            # print(child_node)
            if total_weights is None:
                total_weights = child_node.calculate_weights()
            else:
                total_weights += child_node.calculate_weights()
        # print(self.node_name)
        # print(total_weights)
        # print(total_weights.shape)
        self.weight_matrix = total_weights.copy()
        return total_weights

def generate_analysis_treenode(node_name_list, weight_matrix):
    root = AnalysisTreeNode(node_name='root', weight_matrix=None, is_leaf=False)
    leaf_node_list = []
    for j in range(len(node_name_list)):
        node_name = node_name_list[j]
        split_list = node_name.split('_')
        current_node = root
        node_name = ''
        for i in range(len(split_list)):
            is_leaf = False
            if i == len(split_list) - 1:
                is_leaf = True
            sp = split_list[i]
            if node_name == '':
                node_name = sp
            else:
                node_name = f'{node_name}_{sp}'
            tmp_node = current_node.child_node_dic.get(node_name, None)
            if tmp_node is None:
                tmp_node = AnalysisTreeNode(node_name=node_name,
                                                           weight_matrix=weight_matrix[:, j].copy(),
                                                           is_leaf=is_leaf)
                if is_leaf:
                    leaf_node_list.append(tmp_node)
                current_node.child_node_dic[node_name] = tmp_node
                tmp_node.parent_node = current_node

            current_node = tmp_node
    root.calculate_weights()
    # print(root.weight_matrix)
    return root, leaf_node_list

def get_parent_node(node_list):
    parent_node_list = []
    parent_node_name_set = set()
    for node in node_list:
        if node.parent_node.node_name == 'root':
            if node.node_name not in parent_node_name_set:
                parent_node_list.append(node)
                parent_node_name_set.add(node.node_name)
        else:
            if node.parent_node.node_name not in parent_node_name_set:
                parent_node_list.append(node.parent_node)
                parent_node_name_set.add(node.parent_node.node_name)
    return parent_node_list

def get_child_node(node_list):
    child_node_list = []
    child_node_name_set = set()
    for node in node_list:
        if node.is_leaf:
            if node.node_name not in child_node_name_set:
                child_node_list.append(node)
                child_node_name_set.add(node.node_name)
        else:
            for node_name, child_node in node.child_node_dic.items():
                if node_name not in child_node_name_set:
                    child_node_list.append(child_node)
                    child_node_name_set.add(child_node.node_name)
    return child_node_list

def get_next_level_wo_leaf(node_list):
    child_node_list = []
    child_node_name_set = set()
    for node in node_list:
        if node.is_leaf:
            pass
        else:
            for node_name, child_node in node.child_node_dic.items():
                if child_node.is_leaf:
                    if node.node_name not in child_node_name_set:
                        child_node_list.append(node)
                        child_node_name_set.add(node.node_name)
                else:
                    if node_name not in child_node_name_set:
                        child_node_list.append(child_node)
                        child_node_name_set.add(child_node.node_name)
    return child_node_list