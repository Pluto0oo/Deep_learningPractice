"""
深度学习第十章与第十一章：注意力机制与Transformer
本章将实现注意力机制和Transformer架构的核心组件，并添加丰富的可视化内容
"""

import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import requests
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 80)


def plot_heatmap(data, ax=None, annot=True, fmt='.3f', cmap='Blues', 
                 xticklabels=None, yticklabels=None, cbar_label='值'):
    """
    自定义热力图绘制函数（替代seaborn.heatmap）
    
    参数:
        data: 2D数组
        ax: matplotlib轴对象
        annot: 是否显示数值
        fmt: 数值格式
        cmap: 颜色映射
        xticklabels: x轴标签
        yticklabels: y轴标签
        cbar_label: 颜色条标签
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # 绘制热力图
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # 添加数值标注
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, format(data[i, j], fmt),
                             ha="center", va="center", color="black", fontsize=10)
    
    # 设置标签
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    
    return im


# ============================================================================
# 1. 注意力机制基础
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力机制
    
    参数:
        Q: 查询矩阵 (batch_size, num_heads, seq_len_q, d_k)
        K: 键矩阵 (batch_size, num_heads, seq_len_k, d_k)
        V: 值矩阵 (batch_size, num_heads, seq_len_v, d_v)
        mask: 遮蔽矩阵 (batch_size, 1, seq_len_q, seq_len_k)
    
    返回:
        output: 注意力输出
        attn_weights: 注意力权重
    """
    d_k = Q.size(-1)
    
    # 计算注意力分数: Q * K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用遮蔽（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重（softmax）
    attn_weights = F.softmax(scores, dim=-1)
    
    # 计算输出: 注意力权重 * V
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights


def visualize_attention_weights(attn_weights, title="注意力权重可视化", 
                                 x_labels=None, y_labels=None):
    """
    可视化注意力权重
    
    参数:
        attn_weights: 注意力权重矩阵
        title: 图表标题
        x_labels: x轴标签
        y_labels: y轴标签
    """
    plt.figure(figsize=(10, 8))
    
    # 转换为numpy数组
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    # 如果是4维张量，取第一个样本的第一个头
    if len(attn_weights.shape) == 4:
        attn_weights = attn_weights[0, 0]
    
    # 绘制热力图
    plot_heatmap(attn_weights, annot=True, fmt='.3f', cmap='Blues',
                 xticklabels=x_labels, yticklabels=y_labels, cbar_label='注意力权重')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('键位置', fontsize=12)
    plt.ylabel('查询位置', fontsize=12)
    plt.tight_layout()
    plt.show()


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("\n" + "=" * 80)
    print("1. 测试缩放点积注意力")
    print("=" * 80)
    
    batch_size = 2
    num_heads = 1
    seq_len = 4
    d_k = 64
    
    # 创建随机输入
    Q = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    K = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    V = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    
    # 计算注意力
    output, attn = scaled_dot_product_attention(Q, K, V)
    
    print(f"输入Q形状: {Q.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn.shape}")
    print(f"注意力权重行和: {attn[0, 0].sum(dim=-1)}")
    
    # 可视化注意力权重
    visualize_attention_weights(attn, "缩放点积注意力权重")
    
    return output, attn


# ============================================================================
# 2. 遮蔽注意力（Masked Attention）
# ============================================================================

def create_mask(seq_len):
    """
    创建上三角遮蔽矩阵，用于自回归任务
    
    参数:
        seq_len: 序列长度
    
    返回:
        mask: 遮蔽矩阵 (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask


def test_masked_attention():
    """测试遮蔽注意力"""
    print("\n" + "=" * 80)
    print("2. 测试遮蔽注意力")
    print("=" * 80)
    
    seq_len = 4
    mask = create_mask(seq_len).to(device)
    
    print("遮蔽矩阵:")
    print(mask[0, 0])
    
    # 使用之前创建的Q, K, V
    batch_size = 2
    num_heads = 1
    d_k = 64
    Q = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    K = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    V = torch.randn(batch_size, num_heads, seq_len, d_k).to(device)
    
    output_masked, attn_masked = scaled_dot_product_attention(Q, K, V, mask)
    
    print("\n遮蔽后的注意力权重:")
    print(attn_masked[0, 0])
    
    # 可视化遮蔽注意力权重
    visualize_attention_weights(attn_masked, "遮蔽注意力权重")
    
    # 可视化遮蔽矩阵
    plt.figure(figsize=(8, 6))
    plot_heatmap(mask[0, 0].cpu().numpy(), annot=True, fmt='.0f', 
                 cmap='RdYlGn', cbar_label='遮蔽值')
    plt.title("遮蔽矩阵（下三角矩阵）", fontsize=16, fontweight='bold')
    plt.xlabel('键位置', fontsize=12)
    plt.ylabel('查询位置', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return output_masked, attn_masked


# ============================================================================
# 3. 多头注意力（Multi-Head Attention）
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout率
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        """将最后一维分成num_heads个头"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        参数:
            Q: 查询 (batch_size, seq_len, d_model)
            K: 键 (batch_size, seq_len, d_model)
            V: 值 (batch_size, seq_len, d_model)
            mask: 遮蔽矩阵
        
        返回:
            output: 输出 (batch_size, seq_len, d_model)
            attn_weights: 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分成多头
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # 缩放点积注意力
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.W_o(output)
        
        return output, attn_weights


def visualize_multi_head_attention(attn_weights, num_heads, title="多头注意力可视化"):
    """
    可视化多头注意力
    
    参数:
        attn_weights: 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        num_heads: 注意力头数
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(num_heads):
        ax = axes[i]
        
        # 获取第i个头的注意力权重
        attn = attn_weights[0, i].detach().cpu().numpy()
        
        # 绘制热力图
        plot_heatmap(attn, ax=ax, annot=True, fmt='.3f', cmap='viridis', cbar_label='权重')
        
        ax.set_title(f'注意力头 {i+1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('键位置', fontsize=10)
        ax.set_ylabel('查询位置', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_multi_head_attention():
    """测试多头注意力"""
    print("\n" + "=" * 80)
    print("3. 测试多头注意力")
    print("=" * 80)
    
    d_model = 128
    num_heads = 4
    batch_size = 2
    seq_len = 4
    
    # 创建多头注意力模型
    mha = MultiHeadAttention(d_model, num_heads).to(device)
    print(f"多头注意力模型创建成功")
    print(f"参数数量: {sum(p.numel() for p in mha.parameters())}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 前向传播
    output, attn = mha(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn.shape}")
    print(f"多头数: {num_heads}, 每个头维度: {d_model // num_heads}")
    
    # 可视化多头注意力
    visualize_multi_head_attention(attn, num_heads, "多头注意力权重可视化")
    
    return output, attn


# ============================================================================
# 4. 位置编码（Positional Encoding）
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码
    
    参数:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: dropout率
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 位置编码公式
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        返回:
            output: 添加位置编码后的输出
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


def visualize_positional_encoding(pe, max_positions=50, d_model=128):
    """
    可视化位置编码
    
    参数:
        pe: 位置编码矩阵 (max_len, d_model)
        max_positions: 可视化的最大位置数
        d_model: 模型维度
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 热力图
    ax1 = axes[0]
    pe_np = pe[:max_positions, :].cpu().numpy()
    plot_heatmap(pe_np, ax=ax1, cmap='RdBu_r', cbar_label='编码值')
    ax1.set_title('位置编码热力图', fontsize=14, fontweight='bold')
    ax1.set_xlabel('维度', fontsize=12)
    ax1.set_ylabel('位置', fontsize=12)
    
    # 2. 波形图
    ax2 = axes[1]
    positions = range(min(10, max_positions))
    for i in range(min(4, d_model)):
        ax2.plot(positions, pe_np[positions, i], label=f'维度 {i}', linewidth=2)
    
    ax2.set_title('位置编码波形图（前4个维度）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('位置', fontsize=12)
    ax2.set_ylabel('编码值', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_positional_encoding():
    """测试位置编码"""
    print("\n" + "=" * 80)
    print("4. 测试位置编码")
    print("=" * 80)
    
    d_model = 128
    max_len = 100
    
    # 创建位置编码
    pe = PositionalEncoding(d_model, max_len).to(device)
    
    # 测试
    x = torch.randn(1, 5, d_model).to(device)
    output = pe(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 可视化位置编码
    visualize_positional_encoding(pe.pe, max_positions=50, d_model=d_model)
    
    return output


# ============================================================================
# 5. Transformer编码器层
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dim_feedforward: 前馈网络维度
        dropout: dropout率
    """
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入 (batch_size, seq_len, d_model)
            mask: 遮蔽矩阵
        
        返回:
            output: 输出 (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


def test_transformer_encoder_layer():
    """测试Transformer编码器层"""
    print("\n" + "=" * 80)
    print("5. 测试Transformer编码器层")
    print("=" * 80)
    
    d_model = 128
    num_heads = 4
    dim_feedforward = 512
    batch_size = 2
    seq_len = 5
    
    # 创建编码器层
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward).to(device)
    print(f"Transformer编码器层创建成功")
    print(f"参数数量: {sum(p.numel() for p in encoder_layer.parameters())}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 前向传播
    output = encoder_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    return output


# ============================================================================
# 6. Transformer解码器层
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dim_feedforward: 前馈网络维度
        dropout: dropout率
    """
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        参数:
            x: 解码器输入 (batch_size, seq_len, d_model)
            enc_output: 编码器输出 (batch_size, seq_len, d_model)
            tgt_mask: 目标序列遮蔽
            memory_mask: 编码器输出遮蔽
        
        返回:
            output: 输出 (batch_size, seq_len, d_model)
        """
        # 自注意力（带遮蔽）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 交叉注意力
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = x + self.dropout2(cross_output)
        x = self.norm2(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x


def test_transformer_decoder_layer():
    """测试Transformer解码器层"""
    print("\n" + "=" * 80)
    print("6. 测试Transformer解码器层")
    print("=" * 80)
    
    d_model = 128
    num_heads = 4
    dim_feedforward = 512
    batch_size = 2
    seq_len = 5
    
    # 创建解码器层
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward).to(device)
    print(f"Transformer解码器层创建成功")
    print(f"参数数量: {sum(p.numel() for p in decoder_layer.parameters())}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    enc_output = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 创建遮蔽
    tgt_mask = create_mask(seq_len).to(device)
    
    # 前向传播
    output = decoder_layer(x, enc_output, tgt_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"编码器输出形状: {enc_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    
    return output


# ============================================================================
# 7. 完整的Transformer模型
# ============================================================================

class TransformerModel(nn.Module):
    """
    完整的Transformer模型用于序列建模
    
    参数:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈网络维度
        dropout: dropout率
    """
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def encode(self, x, mask=None):
        """编码器前向传播"""
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x
    
    def decode(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """解码器前向传播"""
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        参数:
            src: 源序列 (batch_size, seq_len)
            tgt: 目标序列 (batch_size, seq_len)
            src_mask: 源序列遮蔽
            tgt_mask: 目标序列遮蔽
        
        返回:
            output: 输出logits (batch_size, seq_len, vocab_size)
        """
        # 编码
        enc_output = self.encode(src, src_mask)
        
        # 解码
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        
        # 输出
        output = self.fc_out(dec_output)
        
        return output


# ============================================================================
# 8. 字符级语言模型
# ============================================================================

class CharLanguageModel(nn.Module):
    """
    字符级语言模型（简化版Transformer）
    
    参数:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 层数
        dim_feedforward: 前馈网络维度
        dropout: dropout率
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 使用编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入序列 (batch_size, seq_len)
            mask: 遮蔽矩阵
        
        返回:
            output: 输出logits (batch_size, seq_len, vocab_size)
        """
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出
        output = self.fc_out(x)
        
        return output


def train_language_model():
    """训练字符级语言模型"""
    print("\n" + "=" * 80)
    print("8. 训练字符级语言模型")
    print("=" * 80)
    
    # 下载文本数据
    url = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    try:
        response = requests.get(url)
        text = response.text
        print(f"下载文本成功，长度: {len(text)} 字符")
    except:
        # 使用本地文件
        try:
            with open('timemachine.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"使用本地文件，长度: {len(text)} 字符")
        except:
            # 使用简单文本
            text = "hello world " * 100
            print(f"使用简单文本，长度: {len(text)} 字符")
    
    # 创建字符到索引的映射
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"字符示例: {chars[:20]}")
    
    # 准备数据
    seq_length = 100
    sequences = []
    for i in range(0, len(text) - seq_length, 1):
        seq = text[i:i + seq_length + 1]
        sequences.append([char_to_idx[ch] for ch in seq])
    
    print(f"序列数量: {len(sequences)}")
    
    # 创建数据集
    class TextDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            seq = self.sequences[idx]
            return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)
    
    # 创建数据加载器
    dataset = TextDataset(sequences[:1000])  # 使用前1000个序列
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    d_model = 128
    num_heads = 4
    num_layers = 2
    
    model = CharLanguageModel(vocab_size, d_model, num_heads, num_layers).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练几个epoch
    num_epochs = 3
    losses = []
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
    
    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linewidth=2, markersize=8)
    plt.title('训练损失曲线', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return model, char_to_idx, idx_to_char


# ============================================================================
# 9. 文本生成
# ============================================================================

def generate_text(model, start_text, char_to_idx, idx_to_char, max_len=100, temperature=1.0):
    """
    使用模型生成文本
    
    参数:
        model: 训练好的模型
        start_text: 起始文本
        char_to_idx: 字符到索引的映射
        idx_to_char: 索引到字符的映射
        max_len: 最大生成长度
        temperature: 温度参数（控制随机性）
    
    返回:
        generated_text: 生成的文本
    """
    model.eval()
    
    # 初始序列
    input_ids = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    
    generated = list(start_text)
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_ids)
            
            # 取最后一个位置的预测
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # 采样
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            
            next_char = idx_to_char[next_char_idx]
            generated.append(next_char)
            
            # 更新输入
            input_ids = torch.cat([input_ids, torch.tensor([[next_char_idx]], device=device)], dim=1)
            
            # 限制输入长度
            if input_ids.size(1) > 100:
                input_ids = input_ids[:, -100:]
    
    return ''.join(generated)


def test_text_generation(model, char_to_idx, idx_to_char):
    """测试文本生成"""
    print("\n" + "=" * 80)
    print("9. 测试文本生成")
    print("=" * 80)
    
    start_texts = ["the ", "hello ", "machine "]
    
    for start_text in start_texts:
        if all(ch in char_to_idx for ch in start_text):
            generated = generate_text(model, start_text, char_to_idx, idx_to_char, max_len=50)
            print(f"\n起始文本: '{start_text}'")
            print(f"生成文本: '{generated}'")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("深度学习第十章与第十一章：注意力机制与Transformer")
    print("=" * 80)
    
    # 1. 测试缩放点积注意力
    test_scaled_dot_product_attention()
    
    # 2. 测试遮蔽注意力
    test_masked_attention()
    
    # 3. 测试多头注意力
    test_multi_head_attention()
    
    # 4. 测试位置编码
    test_positional_encoding()
    
    # 5. 测试Transformer编码器层
    test_transformer_encoder_layer()
    
    # 6. 测试Transformer解码器层
    test_transformer_decoder_layer()
    
    # 7. 训练字符级语言模型
    model, char_to_idx, idx_to_char = train_language_model()
    
    # 8. 测试文本生成
    test_text_generation(model, char_to_idx, idx_to_char)
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
