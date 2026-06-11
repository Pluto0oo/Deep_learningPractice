"""
注意力机制与Transformer相关论文复现
本章包含以下论文的简化复现：

1. Attention Is All You Need (Vaswani et al., 2017) - Transformer
2. Bahdanau Attention (Bahdanau et al., 2014) - 加性注意力
3. Relative Position Attention (2018) - 相对位置编码
4. Linformer (2020) - 线性复杂度注意力
5. Performer (2020) - 线性注意力机制

所有代码均不依赖d2l库，可以直接运行
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# 创建输出文件夹
output_dir = 'paper_plots'
os.makedirs(output_dir, exist_ok=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置Matplotlib科研论文风格（支持中文）
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# 论文1: Bahdanau Attention (2014)
# 论文链接: https://arxiv.org/abs/1409.0473
# ============================================================================

def paper_1_bahdanau_attention():
    """
    Bahdanau注意力机制复现
    这是最早的注意力机制之一，使用加性注意力（Additive Attention）
    
    核心思想：
    - 使用一个前馈网络来计算注意力分数
    - 公式: score(h_j, s_{i-1}) = v^T tanh(W_h h_j + W_s s_{i-1})
    """
    print("\n" + "=" * 80)
    print("论文复现1: Bahdanau Attention (2014)")
    print("论文链接: https://arxiv.org/abs/1409.0473")
    print("=" * 80)
    
    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.W_h = nn.Linear(hidden_size, hidden_size)  # 编码器隐藏状态变换
            self.W_s = nn.Linear(hidden_size, hidden_size)  # 解码器隐藏状态变换
            self.v = nn.Linear(hidden_size, 1)              # 注意力权重计算
        
        def forward(self, encoder_hidden, decoder_hidden):
            """
            参数:
                encoder_hidden: (batch_size, seq_len, hidden_size) - 编码器隐藏状态
                decoder_hidden: (batch_size, hidden_size) - 解码器隐藏状态
            """
            batch_size, seq_len, hidden_size = encoder_hidden.size()
            
            # 变换编码器隐藏状态
            Wh_h = self.W_h(encoder_hidden)  # (batch, seq_len, hidden)
            
            # 变换解码器隐藏状态并扩展维度
            Ws_s = self.W_s(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden)
            
            # 计算注意力分数
            scores = self.v(torch.tanh(Wh_h + Ws_s))  # (batch, seq_len, 1)
            scores = scores.squeeze(-1)                # (batch, seq_len)
            
            # 计算注意力权重
            attn_weights = F.softmax(scores, dim=-1)   # (batch, seq_len)
            
            # 计算上下文向量
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_hidden)  # (batch, 1, hidden)
            context = context.squeeze(1)                                    # (batch, hidden)
            
            return context, attn_weights
    
    # 测试
    batch_size = 2
    seq_len = 5
    hidden_size = 32
    
    attention = BahdanauAttention(hidden_size).to(device)
    
    encoder_hidden = torch.randn(batch_size, seq_len, hidden_size).to(device)
    decoder_hidden = torch.randn(batch_size, hidden_size).to(device)
    
    context, attn = attention(encoder_hidden, decoder_hidden)
    
    print(f"编码器隐藏状态形状: {encoder_hidden.shape}")
    print(f"解码器隐藏状态形状: {decoder_hidden.shape}")
    print(f"上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {attn.shape}")
    print(f"注意力权重行和: {attn[0].sum().item():.4f}")
    
    # 可视化注意力权重
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, seq_len+1), attn[0].detach().cpu().numpy(), 
                  color='#1f77b4', edgecolor='black', linewidth=1.5, width=0.6)
    
    # 添加数值标注
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('Bahdanau注意力权重分布', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('编码器位置', fontsize=12)
    ax.set_ylabel('注意力权重', fontsize=12)
    ax.set_xticks(range(1, seq_len+1))
    ax.set_ylim(0, 1.1 * max(attn[0].detach().cpu().numpy()))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_1_bahdanau_attention.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Bahdanau Attention 复现完成")


# ============================================================================
# 论文2: Attention Is All You Need (2017) - Transformer
# 论文链接: https://arxiv.org/abs/1706.03762
# ============================================================================

def paper_2_transformer():
    """
    Transformer论文复现（简化版）
    这是深度学习领域最重要的论文之一，提出了纯注意力驱动的架构
    
    核心创新：
    1. 缩放点积注意力
    2. 多头注意力
    3. 位置编码
    4. 残差连接 + 层归一化
    """
    print("\n" + "=" * 80)
    print("论文复现2: Attention Is All You Need (2017)")
    print("论文链接: https://arxiv.org/abs/1706.03762")
    print("=" * 80)
    
    # 缩放点积注意力
    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    # 多头注意力
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
        
        def split_heads(self, x):
            batch_size = x.size(0)
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)
            
            Q = self.split_heads(self.W_q(Q))
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))
            
            output, attn = scaled_dot_product_attention(Q, K, V, mask)
            
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(output)
            
            return output, attn
    
    # 位置编码
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            return x + self.pe[:x.size(1), :].unsqueeze(0)
    
    # 可视化位置编码
    pe = PositionalEncoding(64, 50)
    pe_matrix = pe.pe.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pe_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Transformer位置编码', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('特征维度', fontsize=12)
    ax.set_ylabel('位置', fontsize=12)
    plt.colorbar(im, ax=ax, label='编码值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_2_transformer_positional_encoding.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Transformer编码器层
    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
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
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = x + self.dropout1(attn_output)
            x = self.norm1(x)
            
            ff_output = self.feed_forward(x)
            x = x + self.dropout2(ff_output)
            x = self.norm2(x)
            
            return x
    
    # 完整Transformer编码器
    class TransformerEncoder(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model)
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads)
                for _ in range(num_layers)
            ])
        
        def forward(self, x):
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            
            for layer in self.layers:
                x = layer(x)
            
            return x
    
    # 测试
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 2
    batch_size = 2
    seq_len = 10
    
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers).to(device)
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    output = encoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in encoder.parameters())}")
    
    print("✓ Transformer 复现完成")
    
    return encoder


# ============================================================================
# 论文3: Self-Attention with Relative Position Representations (2018)
# 论文链接: https://arxiv.org/abs/1803.02155
# ============================================================================

def paper_3_relative_position_attention():
    """
    相对位置编码注意力机制复现
    这篇论文提出了相对位置编码，改进了Transformer的位置编码方式
    
    核心思想：
    - 不仅考虑绝对位置，还考虑相对位置
    - 在注意力计算中引入相对位置偏差
    """
    print("\n" + "=" * 80)
    print("论文复现3: Relative Position Attention (2018)")
    print("论文链接: https://arxiv.org/abs/1803.02155")
    print("=" * 80)
    
    class RelativePositionAttention(nn.Module):
        def __init__(self, d_model, num_heads, max_relative_positions=10):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.max_relative_positions = max_relative_positions
            
            # 线性变换
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            # 相对位置偏差参数
            self.relative_pos_embeddings = nn.Parameter(
                torch.randn(2 * max_relative_positions + 1, self.d_k)
            )
        
        def get_relative_positions(self, seq_len):
            """生成相对位置索引"""
            range_vec = torch.arange(seq_len)
            relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
            relative_positions = torch.clamp(relative_positions, -self.max_relative_positions, self.max_relative_positions)
            relative_positions = relative_positions + self.max_relative_positions  # 转换为非负索引
            return relative_positions
        
        def forward(self, Q, K, V):
            batch_size = Q.size(0)
            seq_len = Q.size(1)
            
            # 线性变换并分头
            Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # 获取相对位置
            relative_positions = self.get_relative_positions(seq_len).to(Q.device)
            
            # 获取相对位置嵌入 - 需要扩展batch和heads维度
            # rel_embeddings: (seq_len, seq_len, d_k)
            rel_embeddings = self.relative_pos_embeddings[relative_positions]  # (seq_len, seq_len, d_k)
            rel_embeddings = rel_embeddings.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1, -1)
            # rel_embeddings: (batch_size, num_heads, seq_len, seq_len, d_k)
            
            # 计算注意力分数（包含相对位置）
            # 标准注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq_len, seq_len)
            
            # 相对位置分数 - Q: (batch, heads, seq_len, d_k)
            # 需要计算每个位置对所有其他位置的相对位置得分
            Q_expanded = Q.unsqueeze(-2)  # (batch, heads, seq_len, 1, d_k)
            rel_scores = torch.matmul(Q_expanded, rel_embeddings.transpose(-2, -1)).squeeze(-2) / math.sqrt(self.d_k)
            # rel_scores: (batch, heads, seq_len, seq_len)
            
            # 合并分数
            scores = scores + rel_scores
            
            # 计算注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            
            # 计算输出
            output = torch.matmul(attn_weights, V)
            
            # 合并多头
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output, attn_weights
    
    # 测试
    d_model = 128
    num_heads = 4
    max_relative_positions = 5
    batch_size = 2
    seq_len = 10
    
    attention = RelativePositionAttention(d_model, num_heads, max_relative_positions).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn = attention(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"相对位置嵌入形状: {attention.relative_pos_embeddings.shape}")
    
    # 可视化相对位置注意力
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn[0, 0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
    
    ax.set_title('相对位置注意力权重', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('键位置', fontsize=12)
    ax.set_ylabel('查询位置', fontsize=12)
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.grid(False)
    
    plt.colorbar(im, ax=ax, label='注意力权重')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_3_relative_position_attention.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Relative Position Attention 复现完成")


# ============================================================================
# 论文4: Linformer (2020) - 线性复杂度注意力
# 论文链接: https://arxiv.org/abs/2006.04768
# ============================================================================

def paper_4_linformer():
    """
    Linformer论文复现（简化版）
    这篇论文提出了线性复杂度的注意力机制，解决Transformer的O(n²)复杂度问题
    
    核心思想：
    - 使用低秩近似减少计算复杂度
    - 将复杂度从O(n²)降为O(n)
    """
    print("\n" + "=" * 80)
    print("论文复现4: Linformer (2020)")
    print("论文链接: https://arxiv.org/abs/2006.04768")
    print("=" * 80)
    
    class LinformerAttention(nn.Module):
        def __init__(self, d_model, num_heads, k=64):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.k = k  # 低秩投影维度
            
            # 线性变换
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            # 低秩投影矩阵（用于K和V）
            self.E = nn.Parameter(torch.randn(self.k, self.d_k))  # 用于K的投影
            self.F = nn.Parameter(torch.randn(self.k, self.d_k))  # 用于V的投影
        
        def forward(self, Q, K, V):
            batch_size = Q.size(0)
            seq_len = Q.size(1)
            
            # 线性变换并分头
            Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # 低秩投影（将K和V投影到k维）
            K_proj = torch.matmul(K, self.E.T)  # (batch, heads, seq_len, k)
            V_proj = torch.matmul(V, self.F.T)  # (batch, heads, seq_len, k)
            
            # 计算注意力分数（使用投影后的K）
            scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq_len, k)
            
            # 计算注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, seq_len, k)
            
            # 计算输出（使用投影后的V）
            output = torch.matmul(attn_weights, V_proj)  # (batch, heads, seq_len, d_k)
            
            # 合并多头
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output, attn_weights
    
    # 测试
    d_model = 128
    num_heads = 4
    k = 32  # 低秩维度
    batch_size = 2
    seq_len = 100  # 使用较长序列测试
    
    attention = LinformerAttention(d_model, num_heads, k).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn = attention(x, x, x)
    
    print(f"输入序列长度: {seq_len}")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn.shape}")
    print(f"低秩维度 k: {k}")
    
    # 计算复杂度对比
    print(f"\n复杂度对比（假设 d_model={d_model}）:")
    print(f"标准自注意力: O(n²d) = O({seq_len}² × {d_model}) = O({seq_len**2 * d_model})")
    print(f"Linformer: O(nkd) = O({seq_len} × {k} × {d_model}) = O({seq_len * k * d_model})")
    
    # 可视化复杂度对比
    n_values = np.arange(10, 201, 10)
    standard_complexity = n_values ** 2 * d_model
    linformer_complexity = n_values * k * d_model
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(n_values, standard_complexity, label=f'标准自注意力 $O(n^2d)$', 
              linewidth=2, color='#1f77b4', marker='o', markersize=6)
    ax.loglog(n_values, linformer_complexity, label=f'Linformer $O(nkd), k={k}$', 
              linewidth=2, color='#ff7f0e', marker='s', markersize=6)
    
    ax.set_title('Linformer vs 标准自注意力复杂度对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('序列长度 $n$', fontsize=12)
    ax.set_ylabel('计算复杂度', fontsize=12)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_4_linformer_complexity.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Linformer 复现完成")


# ============================================================================
# 论文5: Performer (2020) - 线性注意力机制
# 论文链接: https://arxiv.org/abs/2009.14794
# ============================================================================

def paper_5_performer():
    """
    Performer论文复现（简化版）
    这篇论文提出了基于正随机特征的线性注意力机制
    
    核心思想：
    - 使用正随机特征（Positive Random Features）近似softmax
    - 将复杂度从O(n²)降为O(n)
    """
    print("\n" + "=" * 80)
    print("论文复现5: Performer (2020)")
    print("论文链接: https://arxiv.org/abs/2009.14794")
    print("=" * 80)
    
    class PerformerAttention(nn.Module):
        def __init__(self, d_model, num_heads, random_features=256):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.random_features = random_features
            
            # 线性变换
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            # 随机特征矩阵（用于近似softmax）
            self.random_matrix = nn.Parameter(
                torch.randn(self.random_features, self.d_k) / math.sqrt(self.random_features)
            )
        
        def forward(self, Q, K, V):
            batch_size = Q.size(0)
            seq_len = Q.size(1)
            
            # 线性变换并分头
            Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # 使用正随机特征近似softmax
            # φ(x) = exp(-||x||²/2) * [cos(ωx), sin(ωx)]
            Q_proj = torch.matmul(Q, self.random_matrix.T)  # (batch, heads, seq_len, m)
            K_proj = torch.matmul(K, self.random_matrix.T)  # (batch, heads, seq_len, m)
            
            # 计算正特征
            Q_prime = torch.exp(Q_proj - Q.norm(dim=-1, keepdim=True) ** 2 / 2)
            K_prime = torch.exp(K_proj - K.norm(dim=-1, keepdim=True) ** 2 / 2)
            
            # 计算线性注意力
            # A = (Q'K'^T)^{-1} Q'K'^T V 简化版本
            KV = torch.matmul(K_prime.transpose(-2, -1), V)  # (batch, heads, m, d_k)
            Z = torch.sum(K_prime, dim=-2, keepdim=True)      # (batch, heads, 1, m)
            
            # 归一化因子
            D = 1 / (torch.matmul(Q_prime, Z.transpose(-2, -1)) + 1e-6)  # (batch, heads, seq_len, 1)
            
            # 计算输出
            output = D * torch.matmul(Q_prime, KV)  # (batch, heads, seq_len, d_k)
            
            # 合并多头
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output, None  # 不返回注意力权重（线性注意力没有显式权重）
    
    # 测试
    d_model = 128
    num_heads = 4
    random_features = 128
    batch_size = 2
    seq_len = 100
    
    attention = PerformerAttention(d_model, num_heads, random_features).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, _ = attention(x, x, x)
    
    print(f"输入序列长度: {seq_len}")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"随机特征数: {random_features}")
    
    print("✓ Performer 复现完成")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有论文复现"""
    print("=" * 80)
    print("注意力机制与Transformer相关论文复现")
    print("=" * 80)
    
    # 论文1: Bahdanau Attention (2014)
    paper_1_bahdanau_attention()
    
    # 论文2: Attention Is All You Need (2017)
    paper_2_transformer()
    
    # 论文3: Relative Position Attention (2018)
    paper_3_relative_position_attention()
    
    # 论文4: Linformer (2020)
    paper_4_linformer()
    
    # 论文5: Performer (2020)
    paper_5_performer()
    
    print("\n" + "=" * 80)
    print("所有论文复现完成！")
    print(f"图表已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()