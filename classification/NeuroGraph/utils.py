import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import numpy as np
import pandas as pd
import os
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import tqdm


softmax = torch.nn.LogSoftmax(dim=1)

# Patterned data generator
def generate_complex_patterned_data(num_samples, num_rois, time_steps):
    X = torch.zeros(num_samples, num_rois, time_steps)
    y = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        for roi in range(num_rois):
            # Mix of sinusoidal, linear, and random noise patterns
            if i < num_samples // 2:
                # Class 0: Mix of patterns with more emphasis on sinusoidal
                signal = (
                    0.6 * torch.sin(torch.linspace(0, 2 * np.pi, time_steps)) +
                    0.3 * torch.linspace(0, 1, time_steps) +
                    0.1 * torch.randn(time_steps)
                )
            else:
                # Class 1: Mix of patterns with more emphasis on linear
                signal = (
                    0.3 * torch.sin(torch.linspace(0, 2 * np.pi, time_steps)) +
                    0.6 * torch.linspace(0, 1, time_steps) +
                    0.1 * torch.randn(time_steps)
                )
            X[i, roi, :] = signal

        y[i] = 0 if i < num_samples // 2 else 1

    return X, y

class TimeSeriesEncoder(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size):
        super(TimeSeriesEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=time_steps, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=embedding_size, kernel_size=3, stride=1, padding=1)
        # Removing the linear layer that caused the mismatch
        # If you want to add a linear layer, ensure its input matches the output of conv2
        # self.fc = nn.Linear(num_rois, num_rois)  # Commented out for now

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, num_rois) -> (batch_size, num_rois, time_steps)
        x = F.relu(self.conv1(x))  # (batch_size, embedding_size, num_rois)
        x = F.relu(self.conv2(x))  # (batch_size, embedding_size, num_rois)
        x = x.permute(0, 2, 1)  # (batch_size, num_rois, embedding_size)
        return x




# Graph Generator
class GraphGenerator(nn.Module):
    def __init__(self, embedding_size, num_rois):
        super(GraphGenerator, self).__init__()
        self.fc = nn.Linear(embedding_size, num_rois)

    def forward(self, x):
        # print(x.size())
        hA = F.softmax(self.fc(x), dim=-1)  # hA should now be (batch_size, num_rois, num_rois)
        # print(hA.size())
        A = torch.bmm(hA, hA.transpose(1, 2))  # This ensures A has the shape (batch_size, num_rois, num_rois)
        return A

# Graph Predictor using GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# End-to-end model
class FBNetGen(nn.Module):
    def __init__(self, num_rois, time_steps, embedding_size, hidden_channels, num_classes):
        super(FBNetGen, self).__init__()
        self.encoder = TimeSeriesEncoder(num_rois, time_steps, embedding_size)
        self.graph_generator = GraphGenerator(embedding_size, num_rois)
        self.gcn = GCN(embedding_size, hidden_channels, num_classes)

    def forward(self, x, save_graph=False, batch_idx=None):
        x = self.encoder(x)
        # print('x', x.size())
        A = self.graph_generator(x)
        if save_graph and batch_idx is not None:
            torch.save(A, f"saved_graphs/graph_batch_{batch_idx}.pt")
        
        node_features = x.reshape(-1, x.size(2))
        num_nodes = node_features.size(0)
        edge_index = torch.randint(0, num_nodes, (2, 500)).to(x.device)
        batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1)).to(x.device)
        output = self.gcn(node_features, edge_index, batch)
        return output

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        # 
        num_features = train_dataset[0].num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        elif args.model=="GINConv":
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GNN(mlp))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GNN(mlp))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)+ (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)+ (num_features/2))
            
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        xs = [x]
        for conv in self.convs:
            # xs += [conv(xs[-1], edge_index).tanh()] # ===================================== controlls if edges are weighted or not =====================================
            xs += [conv(xs[-1], edge_index, edge_attr).tanh()] # ===================================== controlls if edges are weighted or not =====================================

        h = []
        upper_tri_indices = torch.triu_indices(x.shape[1], x.shape[1])
        
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t[upper_tri_indices[0], upper_tri_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)

        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)


# class ResidualGNNs(torch.nn.Module):
#     def __init__(self,args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
#         super().__init__()
#         self.convs = ModuleList()
#         self.aggr = aggr.MeanAggregation()
#         self.hidden_channels = hidden_channels

#         num_features = train_dataset[0].num_features
#         if args.model=="ChebConv":
#             if num_layers>0:
#                 self.convs.append(GNN(num_features, hidden_channels,K=5))
#                 for i in range(0, num_layers - 1):
#                     self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
#         elif args.model=="GINConv":
#             mlp = nn.Sequential(
#                 nn.Linear(num_features, hidden_channels),
#                 nn.ReLU(),
#                 nn.Linear(hidden_channels, hidden_channels)
#             )
#             self.convs.append(GNN(mlp))
#             for _ in range(num_layers - 1):
#                 mlp = nn.Sequential(
#                     nn.Linear(hidden_channels, hidden_channels),
#                     nn.ReLU(),
#                     nn.Linear(hidden_channels, hidden_channels)
#                 )
#                 self.convs.append(GNN(mlp))
#         else:
#             if num_layers>0:
#                 self.convs.append(GNN(num_features, hidden_channels))
#                 for i in range(0, num_layers - 1):
#                     self.convs.append(GNN(hidden_channels, hidden_channels))
        
#         # input_dim1 = int(((num_features * num_features)/2) + (num_features/2)+(hidden_channels*num_layers))
#         # input_dim = int(((num_features * num_features)/2) + (num_features/2)) 

#         input_dim1 = int(((num_features * num_features)/2) - (num_features/2)+(hidden_channels*num_layers))
#         input_dim = int(((num_features * num_features)/2) - (num_features/2)) 

#         # input_dim1 = num_features
#         # input_dim = num_features
            
#         self.bn = nn.BatchNorm1d(input_dim)
#         self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim1, hidden),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden, hidden//2),
#             nn.BatchNorm1d(hidden//2),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden//2, hidden//2),
#             nn.BatchNorm1d(hidden//2),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear((hidden//2), args.num_classes),
#         )


#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None


#         xs = [x]
#         for i, conv in enumerate(self.convs):
#             new_x = conv(xs[-1], edge_index).tanh()
#             xs += [new_x]
        
#         h = []
#         for i, xx in enumerate(xs):
#             if i == 0:
#                 xx = xx.reshape(data.num_graphs, x.shape[1], -1)
#                 # x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
#                 # x = torch.stack([t.triu().flatten()[t.triu().flatten()] for t in xx])
#                 # 提取上三角部分并展平
#                 x_list = [t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx]
#                 # 找到最大长度
#                 # print('x_list:', x_list)
#                 max_length = max([len(t) for t in x_list])
#                 # 填充到相同长度
#                 x_list = [F.pad(t, (0, max_length - len(t)), 'constant', 0) for t in x_list]
#                 print('max_length:', max_length)
#                 # print('len(t):', len(t))
#                 x = torch.stack(x_list)
#                 # print('input_dim:', input_dim)
#                 # print('input_dim1:', input_dim1)
#                 print('x.shape:', x.shape)
#                 print('bns input size:', self.bn)
#                 x = self.bn(x)
#                 # print(f"Upper triangle flattened x:", x)
#                 # x = self.bn(x)
#                 # print(f"Batch normalized x:", x)
#             else:
#                 xx = self.aggr(xx, batch)
#                 h.append(xx)
        
#         h = torch.cat(h, dim=1)
#         h = self.bnh(h)
#         x = torch.cat((x, h), dim=1)
#         x = self.mlp(x)
#         # print('h.shape:', h.shape)
#         return F.softmax(x, dim=1), h

    

def load_mat_data(iid):
    data_dir = 'data/functional_networks_132/'
    labels_file = 'data/y_ASR_Witd_Pct.csv'
    labels_df = pd.read_csv(labels_file)
    mat_file = os.path.join(data_dir, f'{iid}.mat')
    try:
        mat = sio.loadmat(mat_file)
        correlation_matrix = mat['rs_mat']
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        edge_index = torch.tensor(np.indices(correlation_matrix.shape), dtype=torch.long).view(2, -1)
        edge_attr = torch.tensor(correlation_matrix.flatten(), dtype=torch.float)
        
        x = torch.tensor(correlation_matrix, dtype=torch.float)
        # # use identity matrix as node features
        # x = torch.eye(correlation_matrix.shape[0], dtype=torch.float)
        # # use all one as node features
        # x = torch.randn((correlation_matrix.shape[0], correlation_matrix.shape[0]), dtype=torch.float)
        # x.fill_diagonal_(0)  # Set the diagonal elements to 0

        label = labels_df[labels_df['IID'] == iid]['DIA'].values[0]
        y = torch.tensor(label, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, iid=iid)
        return data
    except FileNotFoundError:
        print(f'File not found: {mat_file}')
        return None
    



def load_mat_data_ASR(iid, labels_file):
    data_dir = 'data/functional_networks_132/'
    labels_df = pd.read_csv(labels_file)
    mat_file = os.path.join(data_dir, f'{iid}.mat')
    try:
        mat = sio.loadmat(mat_file)
        correlation_matrix = mat['rs_mat']
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        edge_index = torch.tensor(np.indices(correlation_matrix.shape), dtype=torch.long).view(2, -1)
        edge_attr = torch.tensor(correlation_matrix.flatten(), dtype=torch.float)
        
        x = torch.tensor(correlation_matrix, dtype=torch.float)
        # # use identity matrix as node features
        # x = torch.eye(correlation_matrix.shape[0], dtype=torch.float)
        # # use all one as node features
        # x = torch.randn((correlation_matrix.shape[0], correlation_matrix.shape[0]), dtype=torch.float)
        # x.fill_diagonal_(0)  # Set the diagonal elements to 0

        label = labels_df[labels_df['IID'] == iid]['DIA'].values[0]
        y = torch.tensor(label, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    except FileNotFoundError:
        print(f'File not found: {mat_file}')
        return None
    
def load_mat_data_ASR_classifier(iid, labels_file):
    data_dir = 'data/functional_networks_132/'
    labels_df = pd.read_csv(labels_file)
    mat_file = os.path.join(data_dir, f'{iid}.mat')
    try:
        mat = sio.loadmat(mat_file)
        correlation_matrix = mat['rs_mat']
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        # Extract upper triangle excluding the diagonal
        upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
        upper_tri = correlation_matrix[upper_tri_indices]

        x = torch.tensor(upper_tri, dtype=torch.float32)

        label = labels_df[labels_df['IID'] == iid]['DIA'].values[0]
        y = torch.tensor(label, dtype=torch.long)

        return x, y
    except FileNotFoundError:
        print(f'File not found: {mat_file}')
        return None, None
    



def load_data_structural(iid, labels_file, attribute='density'):
    data_dir = 'data/structural_networks_132/'
    labels_df = pd.read_csv(labels_file)
    txt_file = os.path.join(data_dir, f'{iid}_{attribute}.txt')
    
    try:
        with open(txt_file, 'r') as f:
            # Read the file and convert it into a 2D list of floats
            matrix = []
            for line in f:
                matrix.append([float(x) for x in line.split()])

            # Convert the 2D list to a numpy array for easier manipulation
            matrix = np.array(matrix)
            
            # Get the number of nodes
            n = matrix.shape[0]

            # Extract the edges and their attributes from the matrix
            edge_index = []
            edge_attr = []
            for i in range(n):
                for j in range(n):
                    if matrix[i, j] != 0:
                        edge_index.append([i, j])
                        edge_attr.append(matrix[i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            # x = torch.eye(n, dtype=torch.float)
            x = torch.tensor(matrix, dtype=torch.float)
            label = labels_df[labels_df['IID'] == iid]['DIA'].values[0]
            y = torch.tensor(label, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, iid=iid)
            return data
    
    except FileNotFoundError:
        print(f'File not found: {txt_file}')
        return None