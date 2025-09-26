import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv



class GR2ST(nn.Module):
    def __init__(self, temperature, image_dim, spot_dim, projection_dim, 
                 heads_num, dropout=0.0,
                 dynamic_topk=5, spatial_topk=5, fusion_type='concat'):
        super().__init__()
        self.temperature = temperature
        self.heads_num = heads_num  

       
        self.x_embed = nn.Embedding(65536, spot_dim)
        self.y_embed = nn.Embedding(65536, spot_dim)
        
        
        self.cell_type_embed = nn.Sequential(
            nn.Embedding(6, spot_dim//2),
            nn.Linear(spot_dim//2, spot_dim),
            nn.ReLU(),
            nn.LayerNorm(spot_dim),
            nn.Dropout(dropout)
        )
       
        self.image_projection = ProjectionHead(
            embedding_dim=image_dim, 
            projection_dim=projection_dim,
            dropout=dropout
        )
        
      
        self.dynamic_topk = dynamic_topk
        self.spatial_topk = spatial_topk
        self.fusion_type = fusion_type
        
       
        self.dynamic_head_proj = nn.Linear(spot_dim, projection_dim)
        self.dynamic_tail_proj = nn.Linear(spot_dim, projection_dim)
        
        
        self.spatial_proj = nn.Linear(spot_dim, projection_dim)
        
       
        self.dynamic_gat = GATConv(projection_dim, projection_dim, heads=heads_num, dropout=dropout)
        self.spatial_gat = GATConv(projection_dim, projection_dim, heads=heads_num, dropout=dropout)
        
        
        if fusion_type == 'concat':
            self.fusion_dim = heads_num * projection_dim * 2  
        else:
            self.fusion_dim = heads_num * projection_dim * 1  
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, projection_dim),  
            nn.ReLU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout)
        )
        
        self.spot_projection = ProjectionHead(
            embedding_dim=projection_dim, 
            projection_dim=projection_dim,
            dropout=dropout
        )

    def build_cell_type_aware_graph(self, spot_features, cell_types):
        """构建考虑细胞类型相似性的动态图"""
     
        head_features = self.dynamic_head_proj(spot_features)
        tail_features = self.dynamic_tail_proj(spot_features)
        
        feature_sim = torch.mm(head_features, tail_features.t()) / self.temperature
        
        cell_type_sim = (cell_types.unsqueeze(1) == cell_types.unsqueeze(0)).float()
        
       
        combined_sim = feature_sim + 0.3 * cell_type_sim
        combined_sim = F.softmax(combined_sim, dim=-1)
        
       
        k = min(self.dynamic_topk, spot_features.size(0))
        if k <= 0:
            k = 1
        
       
        topk_values, topk_indices = torch.topk(combined_sim, k, dim=-1)
        
       
        edge_index = []
        for i in range(spot_features.size(0)):
            for j in range(k):
                edge_index.append([i, topk_indices[i, j]])
        
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(spot_features.device)
        
        return edge_index

    def build_spatial_graph(self, positions):
        """构建空间图：基于位置邻近性"""
       
        dist_matrix = torch.cdist(positions, positions)
        
        
        k = min(self.spatial_topk, positions.size(0))
        if k <= 0:
            k = 1
        
      
        topk_values, topk_indices = torch.topk(-dist_matrix, k, dim=-1)
        
       
        edge_index = []
        for i in range(positions.size(0)):
            for j in range(k):
                edge_index.append([i, topk_indices[i, j]])
        
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(positions.device)
        
        return edge_index


    def forward(self, batch):
        image_features = batch["image_features"]
        image_embeddings = self.image_projection(image_features)
        
       
        spot_feature = batch["expression"]
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)
        
    
        cell_type = batch["cell_type"].long()
        cell_type_embeddings = self.cell_type_embed(cell_type)
        
       
        spot_features = spot_feature + centers_x + centers_y + cell_type_embeddings
        
        
        dynamic_edge_index = self.build_cell_type_aware_graph(spot_features, cell_type)
        spatial_edge_index = self.build_spatial_graph(batch["position"])
     
        dynamic_features = self.dynamic_head_proj(spot_features)
        spatial_features = self.spatial_proj(spot_features)
        
        dynamic_features = self.dynamic_gat(dynamic_features, dynamic_edge_index)
        
        spatial_features = self.spatial_gat(spatial_features, spatial_edge_index)
        
       
        if self.fusion_type == 'concat':
            fused_features = torch.cat([dynamic_features, spatial_features], dim=-1)
        elif self.fusion_type == 'sum':
            fused_features = dynamic_features + spatial_features
        elif self.fusion_type == 'max':
            fused_features = torch.max(dynamic_features, spatial_features)
        else:  # mean
            fused_features = (dynamic_features + spatial_features) / 2
        fused_features = self.fusion_layer(fused_features)
        
        
        spot_embeddings = self.spot_projection(fused_features)
        
       
        cos_sim = (spot_embeddings @ image_embeddings.T) / self.temperature
        labels = torch.eye(cos_sim.shape[0], device=cos_sim.device)
        spots_loss = F.cross_entropy(cos_sim, labels)
        images_loss = F.cross_entropy(cos_sim.T, labels.T)
        loss = (spots_loss + images_loss) / 2.0
        
        total_loss = loss
        return total_loss.mean()

         
        


class ProjectionHead(nn.Module):
    """投影头模块"""
    def __init__(self, embedding_dim, projection_dim, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    