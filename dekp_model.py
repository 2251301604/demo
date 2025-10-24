"""
DEKP: Deep Learning Model for Enzyme Kinetic Parameter Prediction
基于预训练模型和图神经网络的酶动力学参数预测深度学习模型

This module implements the DEKP model architecture combining:
1. Pre-trained protein language models for sequence representation
2. Graph Neural Networks for molecular structure representation
3. Multi-modal fusion for enzyme kinetic parameter prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinSequenceEncoder(nn.Module):
    """
    蛋白质序列编码器，使用预训练的语言模型
    """
    
    def __init__(self, 
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 hidden_dim: int = 1280,
                 output_dim: int = 512,
                 freeze_backbone: bool = False):
        super(ProteinSequenceEncoder, self).__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Load pre-trained protein language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode protein sequences
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Tokenize sequences
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings from pre-trained model
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.backbone(**encoded)
        
        # Use [CLS] token representation
        sequence_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Project to output dimension
        projected_embeddings = self.projection(sequence_embeddings)
        
        return projected_embeddings


class MolecularGraphEncoder(nn.Module):
    """
    分子图编码器，使用图神经网络处理分子结构
    """
    
    def __init__(self,
                 node_features: int = 78,  # RDKit atom features
                 edge_features: int = 10,  # Bond features
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(MolecularGraphEncoder, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Encode molecular graphs
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level embeddings [batch_size, hidden_dim]
        """
        # Graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        graph_embeddings = global_mean_pool(x, batch)
        
        return graph_embeddings


class MultiModalFusion(nn.Module):
    """
    多模态融合模块，结合蛋白质序列和分子图信息
    """
    
    def __init__(self,
                 protein_dim: int = 512,
                 molecular_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 256):
        super(MultiModalFusion, self).__init__()
        
        self.protein_dim = protein_dim
        self.molecular_dim = molecular_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Attention mechanism for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection layers
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.molecular_proj = nn.Linear(molecular_dim, hidden_dim)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, protein_embeddings: torch.Tensor, 
                molecular_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse protein and molecular embeddings
        
        Args:
            protein_embeddings: [batch_size, protein_dim]
            molecular_embeddings: [batch_size, molecular_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Project to common dimension
        protein_proj = self.protein_proj(protein_embeddings)  # [batch_size, hidden_dim]
        molecular_proj = self.molecular_proj(molecular_embeddings)  # [batch_size, hidden_dim]
        
        # Stack for attention
        stacked = torch.stack([protein_proj, molecular_proj], dim=1)  # [batch_size, 2, hidden_dim]
        
        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Flatten and fuse
        flattened = attended.view(attended.size(0), -1)  # [batch_size, 2 * hidden_dim]
        fused = self.fusion_layers(flattened)
        
        return fused


class DEKPModel(nn.Module):
    """
    DEKP: Deep Learning Model for Enzyme Kinetic Parameter Prediction
    主模型类，整合所有组件
    """
    
    def __init__(self,
                 protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
                 protein_hidden_dim: int = 1280,
                 protein_output_dim: int = 512,
                 molecular_node_features: int = 78,
                 molecular_hidden_dim: int = 256,
                 molecular_num_layers: int = 3,
                 fusion_hidden_dim: int = 512,
                 fusion_output_dim: int = 256,
                 num_kinetic_params: int = 3,  # Km, Kcat, Ki
                 freeze_protein_backbone: bool = False):
        super(DEKPModel, self).__init__()
        
        # Protein sequence encoder
        self.protein_encoder = ProteinSequenceEncoder(
            model_name=protein_model_name,
            hidden_dim=protein_hidden_dim,
            output_dim=protein_output_dim,
            freeze_backbone=freeze_protein_backbone
        )
        
        # Molecular graph encoder
        self.molecular_encoder = MolecularGraphEncoder(
            node_features=molecular_node_features,
            hidden_dim=molecular_hidden_dim,
            num_layers=molecular_num_layers
        )
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            protein_dim=protein_output_dim,
            molecular_dim=molecular_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim
        )
        
        # Kinetic parameter prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_output_dim // 2, fusion_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_output_dim // 4, num_kinetic_params)
        )
        
    def forward(self, 
                protein_sequences: List[str],
                molecular_x: torch.Tensor,
                molecular_edge_index: torch.Tensor,
                molecular_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DEKP model
        
        Args:
            protein_sequences: List of protein sequences
            molecular_x: Molecular node features
            molecular_edge_index: Molecular edge indices
            molecular_batch: Molecular batch assignment
            
        Returns:
            Predicted kinetic parameters [batch_size, num_kinetic_params]
        """
        # Encode protein sequences
        protein_embeddings = self.protein_encoder(protein_sequences)
        
        # Encode molecular graphs
        molecular_embeddings = self.molecular_encoder(
            molecular_x, molecular_edge_index, molecular_batch
        )
        
        # Fuse multi-modal information
        fused_embeddings = self.fusion(protein_embeddings, molecular_embeddings)
        
        # Predict kinetic parameters
        predictions = self.prediction_head(fused_embeddings)
        
        return predictions
    
    def predict_kinetic_parameters(self, 
                                  protein_sequences: List[str],
                                  molecular_x: torch.Tensor,
                                  molecular_edge_index: torch.Tensor,
                                  molecular_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict kinetic parameters with interpretable output
        
        Args:
            protein_sequences: List of protein sequences
            molecular_x: Molecular node features
            molecular_edge_index: Molecular edge indices
            molecular_batch: Molecular batch assignment
            
        Returns:
            Dictionary with predicted parameters
        """
        with torch.no_grad():
            predictions = self.forward(
                protein_sequences, molecular_x, 
                molecular_edge_index, molecular_batch
            )
        
        # Apply appropriate activation functions
        km_pred = torch.exp(predictions[:, 0])  # Km (concentration)
        kcat_pred = torch.exp(predictions[:, 1])  # Kcat (turnover number)
        ki_pred = torch.exp(predictions[:, 2])  # Ki (inhibition constant)
        
        return {
            'Km': km_pred,
            'Kcat': kcat_pred,
            'Ki': ki_pred,
            'raw_predictions': predictions
        }


class DEKPLoss(nn.Module):
    """
    自定义损失函数，适用于酶动力学参数预测
    """
    
    def __init__(self, 
                 km_weight: float = 1.0,
                 kcat_weight: float = 1.0,
                 ki_weight: float = 1.0,
                 use_log_space: bool = True):
        super(DEKPLoss, self).__init__()
        
        self.km_weight = km_weight
        self.kcat_weight = kcat_weight
        self.ki_weight = ki_weight
        self.use_log_space = use_log_space
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Predicted kinetic parameters [batch_size, 3]
            targets: Ground truth kinetic parameters [batch_size, 3]
            
        Returns:
            Total loss
        """
        if self.use_log_space:
            # Convert to log space for better numerical stability
            predictions = torch.log(predictions + 1e-8)
            targets = torch.log(targets + 1e-8)
        
        # Individual parameter losses
        km_loss = self.mse_loss(predictions[:, 0], targets[:, 0])
        kcat_loss = self.mse_loss(predictions[:, 1], targets[:, 1])
        ki_loss = self.mse_loss(predictions[:, 2], targets[:, 2])
        
        # Weighted total loss
        total_loss = (self.km_weight * km_loss + 
                     self.kcat_weight * kcat_loss + 
                     self.ki_weight * ki_loss)
        
        return total_loss


def create_dekp_model(config: Dict) -> DEKPModel:
    """
    创建DEKP模型的工厂函数
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized DEKP model
    """
    model = DEKPModel(
        protein_model_name=config.get('protein_model_name', 'facebook/esm2_t33_650M_UR50D'),
        protein_hidden_dim=config.get('protein_hidden_dim', 1280),
        protein_output_dim=config.get('protein_output_dim', 512),
        molecular_node_features=config.get('molecular_node_features', 78),
        molecular_hidden_dim=config.get('molecular_hidden_dim', 256),
        molecular_num_layers=config.get('molecular_num_layers', 3),
        fusion_hidden_dim=config.get('fusion_hidden_dim', 512),
        fusion_output_dim=config.get('fusion_output_dim', 256),
        num_kinetic_params=config.get('num_kinetic_params', 3),
        freeze_protein_backbone=config.get('freeze_protein_backbone', False)
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    config = {
        'protein_model_name': 'facebook/esm2_t33_650M_UR50D',
        'protein_hidden_dim': 1280,
        'protein_output_dim': 512,
        'molecular_node_features': 78,
        'molecular_hidden_dim': 256,
        'molecular_num_layers': 3,
        'fusion_hidden_dim': 512,
        'fusion_output_dim': 256,
        'num_kinetic_params': 3,
        'freeze_protein_backbone': False
    }
    
    # Create model
    model = create_dekp_model(config)
    
    # Example input
    protein_sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    molecular_x = torch.randn(10, 78)  # 10 atoms, 78 features each
    molecular_edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
    molecular_batch = torch.zeros(10, dtype=torch.long)  # Single molecule
    
    # Forward pass
    predictions = model(protein_sequences, molecular_x, molecular_edge_index, molecular_batch)
    print(f"Model output shape: {predictions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
