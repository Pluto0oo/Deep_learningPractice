import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义英德翻译数据集
translation_data = [
    ("a man in a blue shirt is standing", "ein mann in einem blauen hemd steht"),
    ("two young white men are seen", "zwei junge weiße männer sind zu sehen"),
    ("a group of children are playing in the leaves", "eine gruppe von kindern spielt in den blättern"),
    ("a woman is riding a horse", "eine frau reitet ein pferd"),
    ("a man is riding a bike on the beach", "ein mann fährt ein fahrrad am strand"),
    ("two people are sitting at a table", "zwei personen sitzen an einem tisch"),
    ("a dog is running in the park", "ein hund läuft im park"),
    ("a cat is sleeping on the couch", "eine katze schläft auf dem sofa"),
    ("a boy is playing soccer", "ein junge spielt fußball"),
    ("a girl is reading a book", "ein mädchen liest ein buch"),
]

print(f"数据集大小: {len(translation_data)} 个句子对")

# 构建词汇表
def build_vocab(sentences):
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    for sent in sentences:
        words = sent.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss').split()
        for word in words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_sentences = [pair[0] for pair in translation_data]
tgt_sentences = [pair[1] for pair in translation_data]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)

print(f"源语言词汇表大小: {len(src_vocab)}")
print(f"目标语言词汇表大小: {len(tgt_vocab)}")

# 定义注意力提取器
class AttentionExtractor(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attention_weights = None
    
    def forward(self, query, key, value):
        output, self.attention_weights = self.multihead_attn(query, key, value)
        return output

# 创建模型组件
d_model = 64
num_heads = 4

src_embedding = nn.Embedding(len(src_vocab), d_model).to(device)
tgt_embedding = nn.Embedding(len(tgt_vocab), d_model).to(device)
pos_encoding = nn.Parameter(torch.randn(1, 100, d_model).to(device))

attention_extractor = AttentionExtractor(d_model, num_heads).to(device)
fc_out = nn.Linear(d_model, len(tgt_vocab)).to(device)

# 优化器
params = list(src_embedding.parameters()) + list(tgt_embedding.parameters()) + \
         [pos_encoding] + list(attention_extractor.parameters()) + list(fc_out.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 训练
print("\n开始训练...")
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for src_sent, tgt_sent in translation_data:
        # 准备输入
        src_words = src_sent.lower().split()
        tgt_words = tgt_sent.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss').split()
        
        src_ids = [src_vocab.get(w, 3) for w in src_words]
        tgt_ids = [1] + [tgt_vocab.get(w, 3) for w in tgt_words] + [2]
        
        src_tensor = torch.tensor([src_ids]).to(device)
        tgt_tensor = torch.tensor([tgt_ids[:-1]]).to(device)
        tgt_output = torch.tensor([tgt_ids[1:]]).to(device)
        
        # 前向传播
        src_emb = src_embedding(src_tensor) * math.sqrt(d_model)
        src_emb = src_emb + pos_encoding[:, :src_tensor.size(1), :]
        
        tgt_emb = tgt_embedding(tgt_tensor) * math.sqrt(d_model)
        tgt_emb = tgt_emb + pos_encoding[:, :tgt_tensor.size(1), :]
        
        # 计算交叉注意力
        attn_output = attention_extractor(tgt_emb, src_emb, src_emb)
        
        # 输出
        logits = fc_out(attn_output)
        loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(translation_data):.4f}")

print("训练完成！")

# 提取注意力权重
print("\n提取注意力权重...")

# 测试句子
test_src = "a man in a blue shirt is standing"
test_tgt = "ein mann in einem blauen hemd steht"

src_words = test_src.lower().split()
tgt_words = test_tgt.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss').split()

print(f"源语言: {' '.join(src_words)}")
print(f"目标语言: {' '.join(tgt_words)}")

src_ids = [src_vocab.get(w, 3) for w in src_words]
tgt_ids = [1] + [tgt_vocab.get(w, 3) for w in tgt_words]

src_tensor = torch.tensor([src_ids]).to(device)
tgt_tensor = torch.tensor([tgt_ids]).to(device)

# 提取注意力
attention_extractor.eval()
with torch.no_grad():
    src_emb = src_embedding(src_tensor) * math.sqrt(d_model)
    src_emb = src_emb + pos_encoding[:, :src_tensor.size(1), :]
    
    tgt_emb = tgt_embedding(tgt_tensor) * math.sqrt(d_model)
    tgt_emb = tgt_emb + pos_encoding[:, :tgt_tensor.size(1), :]
    
    _ = attention_extractor(tgt_emb, src_emb, src_emb)
    
    raw_attn_weights = attention_extractor.attention_weights
    print(f"注意力权重形状: {raw_attn_weights.shape}")
    
    # 处理形状
    if len(raw_attn_weights.shape) == 4:
        avg_attn_weights = raw_attn_weights[0].mean(dim=0).cpu().numpy()
    elif len(raw_attn_weights.shape) == 3:
        avg_attn_weights = raw_attn_weights[0].cpu().numpy()
    else:
        raise ValueError(f"意外的形状: {raw_attn_weights.shape}")
    
    print(f"处理后形状: {avg_attn_weights.shape}")
    print(f"\n注意力矩阵:")
    print(avg_attn_weights)

# 可视化
print("\n生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 左上：热力图
ax1 = axes[0, 0]
im1 = ax1.imshow(avg_attn_weights, cmap='Blues', vmin=0, vmax=1)
ax1.set_xticks(range(len(src_words)))
ax1.set_yticks(range(len(tgt_words)))
ax1.set_xticklabels(src_words, fontsize=10, rotation=45, ha='right')
ax1.set_yticklabels(tgt_words, fontsize=10)
ax1.set_xlabel('Source (English)')
ax1.set_ylabel('Target (German)')
ax1.set_title('Attention Weight Heatmap')

for i in range(len(tgt_words)):
    for j in range(len(src_words)):
        color = 'white' if avg_attn_weights[i, j] > 0.4 else 'black'
        ax1.text(j, i, f'{avg_attn_weights[i, j]:.2f}', 
                ha='center', va='center', color=color, fontsize=8)

plt.colorbar(im1, ax=ax1)

# 右上：柱状图
ax2 = axes[0, 1]
colors = plt.cm.Set1(np.linspace(0, 1, len(tgt_words)))
for i in range(len(tgt_words)):
    ax2.bar([x + i*0.8/len(tgt_words) for x in range(len(src_words))], 
           avg_attn_weights[i], 
           width=0.8/len(tgt_words), 
           label=f'{tgt_words[i]}', 
           color=colors[i])

ax2.set_xlabel('Source Position')
ax2.set_ylabel('Attention Weight')
ax2.set_title('Attention Distribution per Target Word')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

# 左下：源词被关注程度
ax3 = axes[1, 0]
src_attention = avg_attn_weights.mean(axis=0)
bars = ax3.bar(src_words, src_attention, color='#1f77b4', alpha=0.8)
ax3.set_xlabel('Source Words')
ax3.set_ylabel('Average Attention')
ax3.set_title('How much each source word is attended to')
ax3.tick_params(axis='x', rotation=45)

for bar, val in zip(bars, src_attention):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# 右下：对齐可视化
ax4 = axes[1, 1]
ax4.set_xlim(-0.5, max(len(src_words), len(tgt_words)) - 0.5)
ax4.set_ylim(-0.5, 1.5)

for i, word in enumerate(src_words):
    ax4.text(i, 1.2, word, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.plot(i, 1.0, 'bo', markersize=8)

for i, word in enumerate(tgt_words):
    ax4.text(i, 0.8, word, ha='center', va='top', fontsize=9, fontweight='bold')
    ax4.plot(i, 1.0, 'ro', markersize=8)
    attended_src = avg_attn_weights[i].argmax()
    weight = avg_attn_weights[i, attended_src]
    ax4.plot([i, attended_src], [1.0, 1.0], 'g-', linewidth=weight * 5, alpha=0.6)

ax4.set_title('Attention Alignment\n(Line thickness = attention strength)')
ax4.axis('off')

plt.tight_layout()
output_dir = 'exercise_plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'exercise_8_translation_attention.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print("\n完成！输出保存为 exercise_8_translation_attention.pdf")
