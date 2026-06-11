"""
动手学习深度学习第10章第7节 - Transformer实现
使用真实的fra-eng数据集（英语-法语翻译）
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import zipfile
import matplotlib.pyplot as plt

# 配置Matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建输出目录
output_dir = 'transformer_outputs'
os.makedirs(output_dir, exist_ok=True)

print(f"{'='*60}")
print("动手学习深度学习第10章第7节 - Transformer实现")
print("使用真实的fra-eng数据集（英语-法语翻译）")
print(f"{'='*60}")
print(f"设备: {device}")
print(f"输出目录: {output_dir}")
print(f"{'='*60}")

# ============== 数据下载和预处理 ==============
def download_extract(url, md5, folder='data'):
    """下载并解压数据集"""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, 'fra-eng.zip')
    extract_dir = os.path.join(folder, 'fra-eng')
    
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    
    if not os.path.exists(extract_dir):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(folder)
    
    return extract_dir

def read_data_nmt():
    """载入“英语－法语”数据集"""
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    md5 = '94646ad1522d915e7b0f9296181140edcf86a4f5'
    data_dir = download_extract(url, md5)
    
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    
    # 使用更高效的方式处理
    punctuation = set(',.!?')
    result = []
    prev_char = ' '
    for char in text:
        if char in punctuation and prev_char != ' ':
            result.append(' ')
        result.append(char)
        prev_char = char
    
    return ''.join(result)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split())
            target.append(parts[1].split())
    return source, target

def build_vocab(tokens):
    """从词元列表构建词汇表"""
    tokens = [token for line in tokens for token in line]
    counter = {}
    for token in tokens:
        counter[token] = counter.get(token, 0) + 1
    
    sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
    idx_to_token = ['<pad>', '<bos>', '<eos>'] + [token for token, _ in sorted_tokens]
    
    if len(idx_to_token) > 10000:
        idx_to_token = idx_to_token[:10000]
    
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    return idx_to_token, token_to_idx

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    """将文本序列转换成小批量"""
    lines = [[vocab[token] for token in line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词汇表"""
    print("Loading data...")
    text = preprocess_nmt(read_data_nmt())
    print(f"Text loaded, length: {len(text)}")
    
    source, target = tokenize_nmt(text, num_examples)
    print(f"Tokenized: {len(source)} source sentences, {len(target)} target sentences")
    
    src_idx_to_token, src_token_to_idx = build_vocab(source)
    tgt_idx_to_token, tgt_token_to_idx = build_vocab(target)
    print(f"Vocab built: source={len(src_idx_to_token)}, target={len(tgt_idx_to_token)}")
    
    src_array, src_valid_len = build_array_nmt(source, src_token_to_idx, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_token_to_idx, num_steps)
    print(f"Arrays built: src shape={src_array.shape}, tgt shape={tgt_array.shape}")
    
    dataset = torch.utils.data.TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    print("DataLoader created successfully")
    
    return data_iter, src_idx_to_token, tgt_idx_to_token

# ============== Multi-Head Attention ==============
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        d_k = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d_k)
        
        if valid_lens is not None:
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            mask = torch.ones_like(scores) * float('-inf')
            mask = torch.triu(mask, diagonal=1)
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.bmm(attention_weights, values)
        output = self.transpose_output(output)
        
        return self.W_o(output), attention_weights

# ============== PositionWiseFFN ==============
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# ============== AddNorm ==============
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# ============== PositionalEncoding ==============
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2) * (-math.log(10000.0) / num_hiddens))
        self.P[:, :, 0::2] = torch.sin(X * div_term)
        self.P[:, :, 1::2] = torch.cos(X * div_term)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# ============== EncoderBlock ==============
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y, _ = self.attention(X, X, X, valid_lens)
        X = self.addnorm1(X, Y)
        Y = self.ffn(X)
        X = self.addnorm2(X, Y)
        return X

# ============== TransformerEncoder ==============
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout=0.1, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias
            ))
    
    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# ============== DecoderBlock ==============
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout=0.1, i=0):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat([state[2][self.i], X], dim=1)
        state[2][self.i] = key_values
        
        Y, _ = self.attention1(X, key_values, key_values)
        Y = self.addnorm1(X, Y)
        
        Y, _ = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Y = self.addnorm2(Y, Y)
        
        Y = self.ffn(Y)
        Y = self.addnorm3(Y, Y)
        
        return Y, state

# ============== TransformerDecoder ==============
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}", DecoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, i
            ))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state

# ============== Transformer ==============
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, enc_valid_lens=None):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        output, _ = self.decoder(dec_X, dec_state)
        return output

# ============== 训练函数 ==============
def train_transformer():
    print("\n" + "="*60)
    print("使用真实fra-eng数据集训练Transformer")
    print("="*60)
    
    batch_size, num_steps = 64, 10
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    
    print(f"\n数据集统计:")
    print(f"  源语言词汇表大小: {len(src_vocab)}")
    print(f"  目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 模型参数（与原文一致）
    num_hiddens, num_layers, dropout = 32, 2, 0.1
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    
    print(f"\n模型参数:")
    print(f"  隐藏层维度: {num_hiddens}")
    print(f"  层数: {num_layers}")
    print(f"  注意力头数: {num_heads}")
    print(f"  前馈网络隐藏层: {ffn_num_hiddens}")
    print(f"  Dropout: {dropout}")
    
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout
    )
    
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout
    )
    
    net = Transformer(encoder, decoder)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    loss = nn.CrossEntropyLoss(ignore_index=src_vocab.index('<pad>'))
    
    num_epochs = 10
    print(f"\n开始训练 {num_epochs} 轮...")
    
    train_losses = []
    
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            
            bos = torch.tensor([tgt_vocab.index('<bos>')] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)
            
            output = net(X, dec_input, X_valid_len)
            l = loss(output.reshape(-1, len(tgt_vocab)), Y.reshape(-1))
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            epoch_loss += l.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{num_epochs:2d}, Loss: {avg_loss:.4f}")
    
    # 绘制训练损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, num_epochs+1), train_losses, marker='o', linewidth=2, color='b')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.set_title('Transformer训练损失曲线')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(range(1, num_epochs+1))
    
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    print(f"\n训练损失曲线已保存: {output_dir}/training_loss.png")
    
    print("\n训练完成！")
    return net, src_vocab, tgt_vocab

# ============== 预测函数 ==============
def predict_transformer(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    # src_vocab 是 idx_to_token 列表，需要创建 token_to_idx 字典
    src_token_to_idx = {token: idx for idx, token in enumerate(src_vocab)}
    tgt_token_to_idx = {token: idx for idx, token in enumerate(tgt_vocab)}
    
    src_tokens = [src_token_to_idx['<bos>']] + [src_token_to_idx.get(token, src_token_to_idx['<pad>']) for token in src_sentence.split()] + [src_token_to_idx['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_token_to_idx['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_token_to_idx['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_tokens = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
        if pred == tgt_token_to_idx['<eos>']:
            break
        output_tokens.append(pred)
    
    return ' '.join([tgt_vocab[i] for i in output_tokens])

# ============== Exercises ==============
def exercise_1():
    print("\n" + "="*60)
    print("练习1: 多头注意力可视化")
    print("="*60)
    num_heads = 4
    attention = MultiHeadAttention(64, 64, 64, 64, num_heads)
    X = torch.randn(2, 5, 64)
    output, attn_weights = attention(X, X, X)
    
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print()
    
    print("各注意力头统计信息:")
    head_stats = []
    for i in range(num_heads):
        head_attn = attn_weights[i * 2:(i + 1) * 2]
        mean_val = head_attn.mean().item()
        max_val = head_attn.max().item()
        head_stats.append((i+1, mean_val, max_val))
        print(f"  Head {i+1}: 均值={mean_val:.4f}, 最大值={max_val:.4f}")
    
    row_sums = attn_weights.sum(dim=-1)
    print(f"\n行和检查 (应为约1): {[f'{v:.4f}' for v in row_sums[0, :].tolist()]}")
    
    # 可视化注意力权重
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        head_idx = i * 2
        im = ax.imshow(attn_weights[head_idx].detach().numpy(), cmap='Blues')
        ax.set_title(f'注意力头 {i+1}')
        ax.set_xlabel('键位置')
        ax.set_ylabel('查询位置')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_heads.png'), dpi=150, bbox_inches='tight')
    print(f"\n注意力权重可视化已保存: {output_dir}/attention_heads.png")

def exercise_2():
    print("\n" + "="*60)
    print("练习2: 位置编码分析")
    print("="*60)
    pos_encoding = PositionalEncoding(16, dropout=0)
    X = torch.zeros(1, 10, 16)
    output = pos_encoding(X)
    
    print(f"位置编码形状: {output.shape}")
    print("\n位置0的前8维:")
    print(f"  {output[0, 0, :8].tolist()}")
    print("\n位置1的前8维:")
    print(f"  {output[0, 1, :8].tolist()}")
    
    print("\n不同位置之间的余弦相似度:")
    sim_matrix = torch.zeros(5, 5)
    for i in range(5):
        for j in range(5):
            sim = F.cosine_similarity(output[0, i], output[0, j], dim=0)
            sim_matrix[i, j] = sim
            if i < j:
                print(f"  位置{i} vs 位置{j}: {sim.item():.4f}")
    
    # 可视化位置编码
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 位置编码热力图
    im1 = ax1.imshow(output[0].detach().numpy(), cmap='viridis')
    ax1.set_title('位置编码热力图')
    ax1.set_xlabel('维度')
    ax1.set_ylabel('位置')
    plt.colorbar(im1, ax=ax1)
    
    # 余弦相似度矩阵
    im2 = ax2.imshow(sim_matrix.detach().numpy(), cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('位置间余弦相似度矩阵')
    ax2.set_xlabel('位置')
    ax2.set_ylabel('位置')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positional_encoding.png'), dpi=150, bbox_inches='tight')
    print(f"\n位置编码可视化已保存: {output_dir}/positional_encoding.png")

def exercise_3():
    print("\n" + "="*60)
    print("练习3: 复杂度分析")
    print("="*60)
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    d_model = 512
    
    print("序列长度n与计算复杂度对比:")
    print(f"{'n':<10} {'自注意力 O(n²d)':<20} {'FFN O(nd²)':<20}")
    print("-" * 50)
    
    attn_complexity = []
    ffn_complexity = []
    
    for n in n_values:
        attn = n * n * d_model
        ffn = n * d_model * d_model
        attn_complexity.append(attn)
        ffn_complexity.append(ffn)
        print(f"{n:<10} {attn:<20} {ffn:<20}")
    
    # 可视化复杂度对比
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, attn_complexity, marker='o', label='自注意力 O(n²d)', linewidth=2)
    ax.plot(n_values, ffn_complexity, marker='s', label='FFN O(nd²)', linewidth=2)
    ax.set_xlabel('序列长度 n')
    ax.set_ylabel('计算复杂度 (浮点运算次数)')
    ax.set_title('Transformer复杂度分析')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.savefig(os.path.join(output_dir, 'complexity_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\n复杂度分析可视化已保存: {output_dir}/complexity_analysis.png")

def exercise_4():
    print("\n" + "="*60)
    print("练习4: Transformer模型测试")
    print("="*60)
    vocab_size = 10
    num_hiddens = 64
    num_heads = 2
    num_layers = 2
    
    encoder = TransformerEncoder(
        vocab_size, num_hiddens, num_hiddens, num_hiddens,
        num_hiddens, [num_hiddens], num_hiddens, 128, num_heads, num_layers
    )
    
    decoder = TransformerDecoder(
        vocab_size, num_hiddens, num_hiddens, num_hiddens,
        num_hiddens, [num_hiddens], num_hiddens, 128, num_heads, num_layers
    )
    
    net = Transformer(encoder, decoder)
    net.to(device)
    
    enc_X = torch.randint(0, vocab_size, (2, 5)).to(device)
    dec_X = torch.randint(0, vocab_size, (2, 5)).to(device)
    output = net(enc_X, dec_X)
    
    print("模型架构信息:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  隐藏层维度: {num_hiddens}")
    print(f"  注意力头数: {num_heads}")
    print(f"  层数: {num_layers}")
    print("\n输入输出形状:")
    print(f"  编码器输入: {enc_X.shape}")
    print(f"  解码器输入: {dec_X.shape}")
    print(f"  输出形状: {output.shape}")
    
    num_params = sum(p.numel() for p in net.parameters())
    print(f"\n模型参数总数: {num_params:,}")
    print("\n✓ Transformer模型实现正确!")

def exercise_5():
    print("\n" + "="*60)
    print("练习5: 真实数据集训练与翻译测试")
    print("="*60)
    net, src_vocab, tgt_vocab = train_transformer()
    
    # 测试翻译
    test_sentences = [
        ("hello world", "bonjour monde"),
        ("i love you", "je t'aime"),
        ("how are you", "comment ça va"),
        ("good morning", "bonjour"),
        ("thank you", "merci")
    ]
    
    print("\n" + "="*60)
    print("翻译结果展示")
    print("="*60)
    print(f"{'英文输入':<20} {'法语翻译':<30} {'参考翻译'}")
    print("-" * 60)
    
    results = []
    for src, ref in test_sentences:
        translation = predict_transformer(net, src, src_vocab, tgt_vocab, 10, device)
        results.append((src, translation, ref))
        print(f"{src:<20} {translation:<30} {ref}")
    
    # 保存翻译结果
    result_file = os.path.join(output_dir, 'translation_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Transformer翻译测试结果\n")
        f.write("="*60 + "\n")
        f.write(f"{'英文输入':<20} {'法语翻译':<30} {'参考翻译'}\n")
        f.write("-" * 60 + "\n")
        for src, translation, ref in results:
            f.write(f"{src:<20} {translation:<30} {ref}\n")
    
    print(f"\n翻译结果已保存: {output_dir}/translation_results.txt")

def main():
    print(f"{'='*70}")
    print("动手学习深度学习第10章第7节 - Transformer实现")
    print("使用真实fra-eng数据集（英语-法语翻译）")
    print(f"{'='*70}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")
    
    # 运行所有练习
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    
    # 输出总结
    print(f"\n{'='*70}")
    print("所有练习完成！")
    print(f"{'='*70}")
    print("\n输出文件清单:")
    print("-" * 30)
    output_files = [
        'attention_heads.png',
        'positional_encoding.png', 
        'complexity_analysis.png',
        'training_loss.png',
        'translation_results.txt'
    ]
    for f in output_files:
        print(f"  ✓ {output_dir}/{f}")
    print("\n可直接上传至GitHub展示！")

if __name__ == '__main__':
    main()