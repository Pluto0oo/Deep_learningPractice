# 深度学习第十章与第十一章：注意力机制与Transformer

---

## 目录

1. 注意力机制概述
2. 缩放点积注意力
3. 遮蔽注意力
4. 多头注意力
5. 位置编码
6. Transformer架构
7. 代码实现详解
8. 训练与生成
9. 可视化展示

---

## 1. 注意力机制概述

### 什么是注意力机制？

**核心思想：** 选择性地关注输入的某些部分

**类比：** 人类阅读时的注意力 - 读到"猫"时会关注"捉"、"老鼠"等

### 为什么需要注意力？

- **解决长距离依赖问题**
  - RNN难以处理长序列
  - 注意力可以一次看到整个序列
  
- **并行计算**
  - 不像RNN必须顺序处理
  - 大幅提升训练速度

---

## 2. 缩放点积注意力

### 核心公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### 三个关键向量

| 向量 | 含义 | 作用 |
|------|------|------|
| Q (Query) | 查询 | 我想查找什么 |
| K (Key) | 键 | 我有什么信息 |
| V (Value) | 值 | 信息的实际内容 |

### 代码实现

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用遮蔽（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 计算输出
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

### 为什么除以√d_k？

- **问题**：当d_k很大时，点积值会很大
- **影响**：softmax进入饱和区，梯度接近0
- **解决**：除以√d_k保持方差稳定

---

## 3. 遮蔽注意力

### 应用场景

**自回归任务**：文本生成、机器翻译解码

**问题**：训练时不能看到未来的信息

### 解决方案

创建下三角遮蔽矩阵：

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

### 实现代码

```python
def create_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
```

### 效果

- 位置4只能看到位置1-4
- 位置3只能看到位置1-3
- 以此类推...

---

## 4. 多头注意力

### 核心思想

将注意力分成多个"头"，每个头独立学习

### 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 为什么需要多头？

**不同的头可以关注不同的信息：**

| 注意力头 | 关注内容 | 示例 |
|---------|---------|------|
| 头1 | 语法结构 | 主语-动词关系 |
| 头2 | 语义相似 | 同义词关联 |
| 头3 | 位置关系 | 邻近词依赖 |

### 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 四个线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        output, attn = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output), attn
```

---

## 5. 位置编码

### 问题

Transformer没有循环结构，无法感知序列中元素的位置

### 解决方案

添加位置编码（Positional Encoding）

### 公式

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 特点

- **周期性**：可以处理任意长度的序列
- **唯一性**：每个位置都有唯一的编码
- **相对位置**：可以通过线性变换表示相对位置

### 可视化

- 热力图：显示不同位置和维度的编码值
- 波形图：显示前几个维度的正弦波形

---

## 6. Transformer架构

### 编码器（Encoder）

**每一层包含：**
1. 多头自注意力
2. 残差连接 + LayerNorm
3. 前馈网络
4. 残差连接 + LayerNorm

### 解码器（Decoder）

**每一层包含：**
1. 遮蔽多头自注意力
2. 残差连接 + LayerNorm
3. 交叉注意力（Encoder-Decoder Attention）
4. 残差连接 + LayerNorm
5. 前馈网络
6. 残差连接 + LayerNorm

### 完整结构

```
输入 → 嵌入 + 位置编码 → [编码器层 × N] → 编码器输出
                                    ↓
输入 → 嵌入 + 位置编码 → [解码器层 × N] → 线性层 → Softmax → 输出
```

---

## 7. 代码实现详解

### 完整Transformer模型

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, 
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码
        enc_output = self.encode(src, src_mask)
        
        # 解码
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        
        # 输出
        return self.fc_out(dec_output)
```

### 编码器层实现

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + attn_output
        x = self.norm1(x)
        
        # 前馈 + 残差
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x
```

---

## 8. 训练与生成

### 字符级语言模型

**任务：** 给定前N个字符，预测第N+1个字符

**数据集：** 《时间机器》（Time Machine）

**训练流程：**
1. 下载并预处理文本
2. 建立字符到索引的映射
3. 创建训练序列
4. 训练Transformer模型
5. 监控训练损失

### 文本生成

**温度采样（Temperature Sampling）：**

```python
def generate_text(model, start_text, temperature=1.0):
    model.eval()
    input_ids = text_to_indices(start_text)
    
    with torch.no_grad():
        for _ in range(max_len):
            # 前向传播
            output = model(input_ids)
            
            # 温度采样
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个字符
            next_char_idx = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_char_idx], dim=1)
    
    return indices_to_text(input_ids)
```

### 温度参数的影响

| 温度 | 效果 | 示例 |
|------|------|------|
| 0.5 | 较保守 | 确定性强，少见词少 |
| 1.0 | 平衡 | 正常生成 |
| 1.5 | 较随机 | 更多样化，可能有错误 |

---

## 9. 可视化展示

### 本代码包含的可视化

1. **注意力权重热力图**
   - 显示每个查询对所有键的关注程度
   - 颜色越深表示权重越大

2. **多头注意力对比**
   - 展示不同注意力头学到的不同模式
   - 每个头可能关注不同类型的关系

3. **位置编码可视化**
   - 热力图：不同位置和维度的编码值
   - 波形图：正弦余弦函数的周期性

4. **训练损失曲线**
   - 展示模型学习过程
   - 损失逐渐下降表示训练有效

---

## 10. 关键概念总结

### 注意力机制
- Q、K、V三个向量的交互
- 缩放因子√d_k防止梯度消失

### 多头注意力
- 并行计算多个注意力
- 每个头关注不同的表示子空间

### 位置编码
- 正弦余弦函数编码位置
- 解决序列顺序问题

### Transformer
- 编码器-解码器架构
- 残差连接和LayerNorm
- 平行计算，高效训练

---

## 11. 实际应用

### 领域
- **自然语言处理**：机器翻译、文本生成、问答系统
- **计算机视觉**：图像分类、目标检测、图像描述
- **语音处理**：语音识别、语音合成

### 代表模型
- BERT（双向编码器）
- GPT（生成式预训练）
- ViT（Vision Transformer）

---

## 12. 学习建议

### 理论学习
1. 理解Q、K、V的含义
2. 掌握缩放点积注意力的公式
3. 理解多头注意力的作用
4. 学习位置编码的原理

### 实践建议
1. 手动实现注意力计算
2. 观察不同头的可视化
3. 调整超参数观察效果
4. 尝试不同的文本生成任务

### 推荐资源
- 《动手学习深度学习》第十、十一章
- Attention is All You Need（原始论文）
- Jay Alammar的博客（可视化解释）

---

## 总结

本代码实现了：
1. ✅ 缩放点积注意力
2. ✅ 遮蔽注意力
3. ✅ 多头注意力
4. ✅ 位置编码
5. ✅ Transformer编码器
6. ✅ Transformer解码器
7. ✅ 字符级语言模型
8. ✅ 文本生成
9. ✅ 丰富的可视化

---

# 谢谢观看！

## 联系方式

如有问题，欢迎讨论！

---

## 代码运行指南

### 环境要求
- Python 3.7+
- PyTorch 1.8+
- Matplotlib
- NumPy

### 运行方法
```bash
python attention_transformer.py
```

### 输出
- 注意力权重可视化
- 位置编码可视化
- 训练损失曲线
- 生成的文本示例

---

## 思考题

1. 为什么需要多头注意力？一个头不够吗？
2. 位置编码为什么使用正弦余弦函数？有什么优点？
3. 残差连接的作用是什么？为什么需要LayerNorm？
4. 温度参数如何影响文本生成的质量？

---

## 参考资料

1. Vaswani et al. "Attention Is All You Need" (2017)
2. 《动手学习深度学习》第10、11章
3. Lilian Weng的博客
4. Jay Alammar的图解指南

---

# 演示结束

感谢聆听！
