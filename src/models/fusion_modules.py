"""
Fusion Modules for Multi-Modal Medical Image Classification
=============================================================

Implements various fusion strategies for combining CNN image features with
structured clinical indicators (CDR, ISNT, vessel density, entropy).

Part of ARC Phase E: Architecture Grammar System
Dev 2 implementation - Week 1

Fusion Strategies:
- FiLM: Feature-wise Linear Modulation (clinical indicators modulate CNN features)
- CrossAttention: Clinical indicators as queries, CNN features as keys/values
- Gated: Learned per-sample gates that weight CNN vs clinical contribution
- LateFusion: Baseline concatenation + MLP

All modules accept:
    cnn_features: [B, C, H, W] - CNN feature maps
    clinical_vector: [B, K] - Structured clinical indicators

All modules return:
    fused_representation: [B, D] - Ready for classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)

    Clinical indicators generate scale (gamma) and shift (beta) parameters
    that modulate CNN feature maps channel-wise. This allows clinical context
    to adaptively re-weight spatial features.

    Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"

    Args:
        cnn_channels: Number of CNN feature map channels (e.g., 512 for EfficientNet-B3)
        clinical_dim: Dimension of clinical indicator vector (e.g., 4-8)
        hidden_dim: Hidden dimension for gamma/beta generators
        output_dim: Final fused representation dimension
        use_global_pool: Whether to global average pool after modulation

    Forward:
        cnn_features: [B, C, H, W]
        clinical_vector: [B, K]
        -> [B, output_dim]
    """

    def __init__(
        self,
        cnn_channels: int,
        clinical_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
        use_global_pool: bool = True
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.clinical_dim = clinical_dim
        self.output_dim = output_dim
        self.use_global_pool = use_global_pool

        # Clinical indicator encoder (generates gamma and beta)
        self.gamma_generator = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cnn_channels)
        )

        self.beta_generator = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cnn_channels)
        )

        # Post-modulation projection
        if use_global_pool:
            self.projection = nn.Linear(cnn_channels, output_dim)
        else:
            # If not pooling, project spatial features
            self.projection = nn.Conv2d(cnn_channels, output_dim, kernel_size=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        cnn_features: torch.Tensor,
        clinical_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning and return fused representation.

        Args:
            cnn_features: [B, C, H, W] - CNN feature maps
            clinical_vector: [B, K] - Clinical indicators

        Returns:
            fused: [B, output_dim] - Modulated and pooled features
        """
        B, C, H, W = cnn_features.shape

        # Generate modulation parameters from clinical indicators
        gamma = self.gamma_generator(clinical_vector)  # [B, C]
        beta = self.beta_generator(clinical_vector)    # [B, C]

        # Reshape for broadcasting: [B, C, 1, 1]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # Apply FiLM: features = gamma * features + beta (channel-wise)
        modulated_features = gamma * cnn_features + beta  # [B, C, H, W]

        # Global average pooling and projection
        if self.use_global_pool:
            pooled = F.adaptive_avg_pool2d(modulated_features, 1)  # [B, C, 1, 1]
            pooled = pooled.view(B, C)  # [B, C]
            fused = self.projection(pooled)  # [B, output_dim]
        else:
            projected = self.projection(modulated_features)  # [B, output_dim, H, W]
            fused = self.adaptive_pool(projected).view(B, self.output_dim)

        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion

    Clinical indicators are used as queries to attend to CNN spatial features
    (keys/values). This allows the model to selectively focus on image regions
    relevant to specific clinical measurements.

    Args:
        cnn_channels: Number of CNN feature map channels
        clinical_dim: Dimension of clinical indicator vector
        num_heads: Number of attention heads
        hidden_dim: Dimension of query/key/value projections per head
        output_dim: Final fused representation dimension
        dropout: Dropout probability for attention weights

    Forward:
        cnn_features: [B, C, H, W]
        clinical_vector: [B, K]
        -> [B, output_dim]
    """

    def __init__(
        self,
        cnn_channels: int,
        clinical_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.clinical_dim = clinical_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Project clinical indicators to queries
        self.query_proj = nn.Linear(clinical_dim, hidden_dim)

        # Project CNN features to keys and values
        self.key_proj = nn.Conv2d(cnn_channels, hidden_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(cnn_channels, hidden_dim, kernel_size=1)

        # Multi-head attention
        self.attention_dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(
        self,
        cnn_features: torch.Tensor,
        clinical_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention and return fused representation.

        Args:
            cnn_features: [B, C, H, W] - CNN feature maps
            clinical_vector: [B, K] - Clinical indicators

        Returns:
            fused: [B, output_dim] - Attended features
        """
        B, C, H, W = cnn_features.shape
        N = H * W  # Number of spatial locations

        # Generate queries from clinical indicators
        queries = self.query_proj(clinical_vector)  # [B, hidden_dim]
        queries = queries.view(B, self.num_heads, 1, self.head_dim)  # [B, num_heads, 1, head_dim]

        # Generate keys and values from CNN features
        keys = self.key_proj(cnn_features)  # [B, hidden_dim, H, W]
        values = self.value_proj(cnn_features)  # [B, hidden_dim, H, W]

        # Reshape for multi-head attention: [B, num_heads, N, head_dim]
        keys = keys.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        values = values.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)

        # Scaled dot-product attention
        # Q: [B, num_heads, 1, head_dim]
        # K: [B, num_heads, N, head_dim]
        # scores: [B, num_heads, 1, N]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, N]
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        # attention_weights: [B, num_heads, 1, N]
        # values: [B, num_heads, N, head_dim]
        # attended: [B, num_heads, 1, head_dim]
        attended = torch.matmul(attention_weights, values)

        # Concatenate heads and flatten
        attended = attended.view(B, self.hidden_dim)  # [B, hidden_dim]

        # Output projection
        fused = self.output_proj(attended)  # [B, output_dim]

        return fused


class GatedFusion(nn.Module):
    """
    Gated Fusion

    Learns per-sample soft gates that weight the contribution of CNN features
    vs clinical indicators. The gate is conditioned on both modalities, allowing
    the model to adapt fusion strategy per sample.

    Args:
        cnn_channels: Number of CNN feature map channels
        clinical_dim: Dimension of clinical indicator vector
        hidden_dim: Hidden dimension for gate computation
        output_dim: Final fused representation dimension
        gate_activation: Activation for gate ('sigmoid' or 'softmax')

    Forward:
        cnn_features: [B, C, H, W]
        clinical_vector: [B, K]
        -> [B, output_dim]
    """

    def __init__(
        self,
        cnn_channels: int,
        clinical_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
        gate_activation: str = 'sigmoid'
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.clinical_dim = clinical_dim
        self.output_dim = output_dim
        self.gate_activation = gate_activation

        # CNN feature encoder
        self.cnn_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnn_channels, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Clinical indicator encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Gate network (conditions on both modalities)
        if gate_activation == 'sigmoid':
            # Single gate: lambda in [0, 1]
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif gate_activation == 'softmax':
            # Two gates: [lambda_cnn, lambda_clinical] that sum to 1
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2)
            )
        else:
            raise ValueError(f"gate_activation must be 'sigmoid' or 'softmax', got {gate_activation}")

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        cnn_features: torch.Tensor,
        clinical_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gated fusion and return fused representation.

        Args:
            cnn_features: [B, C, H, W] - CNN feature maps
            clinical_vector: [B, K] - Clinical indicators

        Returns:
            fused: [B, output_dim] - Gated fusion of both modalities
        """
        B = cnn_features.size(0)

        # Encode both modalities
        cnn_encoded = self.cnn_encoder(cnn_features)  # [B, hidden_dim]
        clinical_encoded = self.clinical_encoder(clinical_vector)  # [B, hidden_dim]

        # Compute gate based on both modalities
        concat_features = torch.cat([cnn_encoded, clinical_encoded], dim=1)  # [B, hidden_dim * 2]

        if self.gate_activation == 'sigmoid':
            # Single gate: fused = lambda * cnn + (1 - lambda) * clinical
            gate = self.gate_network(concat_features)  # [B, 1]
            gated = gate * cnn_encoded + (1 - gate) * clinical_encoded  # [B, hidden_dim]
            # Concatenate with clinical for residual information
            combined = torch.cat([gated, clinical_encoded], dim=1)  # [B, hidden_dim * 2]
        else:  # softmax
            # Two gates that sum to 1
            gates = self.gate_network(concat_features)  # [B, 2]
            gates = F.softmax(gates, dim=1)  # [B, 2]
            gate_cnn = gates[:, 0:1]  # [B, 1]
            gate_clinical = gates[:, 1:2]  # [B, 1]
            gated = gate_cnn * cnn_encoded + gate_clinical * clinical_encoded  # [B, hidden_dim]
            combined = torch.cat([gated, clinical_encoded], dim=1)  # [B, hidden_dim * 2]

        # Output projection
        fused = self.output_proj(combined)  # [B, output_dim]

        return fused


class LateFusion(nn.Module):
    """
    Late Fusion (Baseline)

    Simple concatenation of CNN features (after global pooling) and clinical
    indicators, followed by MLP projection. This is the baseline fusion strategy
    currently used in AcuVue.

    Args:
        cnn_channels: Number of CNN feature map channels
        clinical_dim: Dimension of clinical indicator vector
        hidden_dim: Hidden dimension for MLP
        output_dim: Final fused representation dimension
        dropout: Dropout probability

    Forward:
        cnn_features: [B, C, H, W]
        clinical_vector: [B, K]
        -> [B, output_dim]
    """

    def __init__(
        self,
        cnn_channels: int,
        clinical_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.clinical_dim = clinical_dim
        self.output_dim = output_dim

        # Global pooling for CNN features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # MLP projection
        concat_dim = cnn_channels + clinical_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        clinical_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply late fusion and return fused representation.

        Args:
            cnn_features: [B, C, H, W] - CNN feature maps
            clinical_vector: [B, K] - Clinical indicators

        Returns:
            fused: [B, output_dim] - Concatenated and projected features
        """
        B = cnn_features.size(0)

        # Global average pooling
        cnn_pooled = self.global_pool(cnn_features).view(B, self.cnn_channels)  # [B, C]

        # Concatenate modalities
        concat = torch.cat([cnn_pooled, clinical_vector], dim=1)  # [B, C + K]

        # MLP projection
        fused = self.mlp(concat)  # [B, output_dim]

        return fused


# Factory function for easy instantiation
def create_fusion_module(
    fusion_type: str,
    cnn_channels: int,
    clinical_dim: int,
    output_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion modules by name.

    Args:
        fusion_type: One of ['film', 'cross_attention', 'gated', 'late']
        cnn_channels: Number of CNN feature map channels
        clinical_dim: Dimension of clinical indicator vector
        output_dim: Final fused representation dimension
        **kwargs: Additional arguments passed to specific fusion module

    Returns:
        Fusion module instance

    Example:
        >>> fusion = create_fusion_module('film', cnn_channels=512, clinical_dim=4, output_dim=256)
    """
    fusion_type = fusion_type.lower()

    if fusion_type == 'film':
        return FiLMLayer(cnn_channels, clinical_dim, output_dim=output_dim, **kwargs)
    elif fusion_type == 'cross_attention':
        return CrossAttentionFusion(cnn_channels, clinical_dim, output_dim=output_dim, **kwargs)
    elif fusion_type == 'gated':
        return GatedFusion(cnn_channels, clinical_dim, output_dim=output_dim, **kwargs)
    elif fusion_type == 'late':
        return LateFusion(cnn_channels, clinical_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown fusion_type: {fusion_type}. "
            f"Must be one of ['film', 'cross_attention', 'gated', 'late']"
        )
