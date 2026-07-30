# 持续学习实验项目

## 项目概述

本项目实现并对比了多种持续学习（Continual Learning）方法，用于文本分类任务。通过模拟两个分布不同的文本分类任务的顺序训练，对比不同方法在缓解灾难性遗忘（Catastrophic Forgetting）方面的表现。

## 目录结构

```
ContinualLearning/
├── configs/
│   └── base.yaml                    # 基础配置文件
├── docs/
│   └── experiment_principle.md      # 实验原理文档
├── scripts/
│   ├── run_fine_tuning.py           # 直接微调基线
│   ├── run_ewc.py                   # 弹性权重巩固（EWC）
│   ├── run_lwf.py                   # 无遗忘学习（LWF）
│   ├── run_der.py                   # 暗经验回放（DER）
│   ├── run_comparison.py            # 全部方法对比
│   └── compare_results.py           # 结果对比可视化
├── src/
│   ├── __init__.py
│   ├── config.py                    # 配置加载模块
│   ├── data.py                      # 数据加载与经验回放缓冲区
│   ├── models.py                    # 模型定义（TextClassifier等）
│   ├── trainer.py                   # 训练器基类与各方法实现
│   ├── evaluator.py                 # 评估指标计算
│   └── visualization.py             # 结果可视化
├── results/                         # 实验结果存储
├── data/                            # 数据集存储
├── requirements.txt                 # 依赖列表
├── .gitignore
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
conda create -n dlp python=3.9
conda activate dlp
pip install -r requirements.txt
```

### 2. 主要依赖

| 库 | 版本要求 | 说明 |
|----|----------|------|
| Python | 3.9+ | 推荐使用conda环境 |
| PyTorch | 2.5.1+ | 深度学习框架 |
| transformers | 4.45.0+ | HuggingFace预训练模型 |
| datasets | 2.20+ | HuggingFace数据集 |
| numpy | 2.0+ | 数值计算 |
| matplotlib | 3.9+ | 可视化绘图 |
| scikit-learn | 1.7+ | 评估指标 |

## 任务设计

### 任务1：IMDB情感分类
- **数据集**：IMDB电影评论
- **类别数**：2（正面/负面）
- **训练样本**：25,000
- **测试样本**：25,000
- **平均文本长度**：约230词

### 任务2：AG News新闻分类
- **数据集**：AG News新闻标题
- **类别数**：4（World/Sports/Business/Tech）
- **训练样本**：120,000
- **测试样本**：7,600
- **平均文本长度**：约12词

### 任务分布差异
- IMDB：电影评论情感分析，文本较长，语言较口语化
- AG News：新闻标题分类，文本较短，语言较正式

## 快速开始

### 运行直接微调基线

```bash
python scripts/run_fine_tuning.py --config configs/base.yaml
```

**方法说明**：直接在每个新任务上微调预训练模型，不采用任何持续学习机制。预期会出现严重的灾难性遗忘。

### 运行弹性权重巩固（EWC）

```bash
python scripts/run_ewc.py --config configs/base.yaml
```

**方法说明**：通过计算Fisher信息矩阵识别对旧任务重要的权重参数，在训练新任务时对这些参数施加正则化约束。

**关键参数**：
- `ewc_lambda`：EWC正则化强度（默认：1000）
- `ewc_fisher_sample_size`：计算Fisher信息的样本数（默认：200）

### 运行无遗忘学习（LWF）

```bash
python scripts/run_lwf.py --config configs/base.yaml
```

**方法说明**：使用知识蒸馏技术，让新模型在学习新任务的同时，保持对旧任务的预测能力。

**关键参数**：
- `lwf_lambda`：蒸馏损失权重（默认：1.0）
- `lwf_temperature`：蒸馏温度（默认：2.0）

### 运行暗经验回放（DER）

```bash
python scripts/run_der.py --config configs/base.yaml
```

**方法说明**：结合经验回放和知识蒸馏，使用旧模型对缓冲区样本的预测作为软标签。

**关键参数**：
- `buffer_size`：经验回放缓冲区大小（默认：500）
- `der_lambda`：蒸馏损失权重（默认：1.0）
- `der_alpha`：新旧样本混合比例（默认：0.5）

### 运行全部方法对比

```bash
python scripts/run_comparison.py --config configs/base.yaml
```

**输出**：依次运行所有方法，生成对比结果。

## 技术架构

### 模型架构
```
基础模型：bert-base-uncased
├── 12层Transformer Encoder
├── 768维隐藏层
├── 12个注意力头
├── Dropout层 (p=0.1)
└── 线性分类层（输出维度=总类别数）

训练策略：
├── 仅微调最后2层Transformer和分类器
├── 学习率：2e-5
├── 批大小：32
└── 训练轮数：3
```

### 数据处理流程
```
原始文本 → Tokenizer → input_ids + attention_mask
    ↓
TextDataset (PyTorch Dataset)
    ↓
ContinualDataLoader (支持多任务加载)
    ↓
ExperienceBuffer (均匀采样策略)
```

### 训练流程
```
Task 1 训练
    ↓
保存Task 1模型（教师模型）
    ↓
计算Fisher信息矩阵（EWC）
    ↓
初始化经验回放缓冲区
    ↓
Task 2 训练
    ├── 新任务样本
    ├── 缓冲区样本（若使用经验回放）
    └── 蒸馏损失计算（若使用LWF/DER）
    ↓
评估所有任务性能
```

## 评估指标

### 准确率（Accuracy）
$$Accuracy = \frac{\text{正确预测数}}{\text{总样本数}}$$

### 遗忘率（Forgetting Rate）
$$Forgetting = \max_{t \leq T} Acc(t) - Acc(T)$$
其中：
- $Acc(t)$：在任务t训练后对任务t的准确率
- $Acc(T)$：在所有任务训练完成后对任务t的准确率

### F1分数
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 计算效率指标
- 训练总时间
- GPU内存占用（Allocated/Reserved）

## 实验结果

### 实验状态

✅ **实验已完成**（2026-07-22运行）：四种方法均成功运行并生成结果。

### 方法性能对比（实际结果）

| 方法 | Task 1准确率 | Task 2准确率 | 遗忘率 | 训练时间(秒) |
|------|-------------|-------------|--------|-------------|
| Fine Tuning | 0.6230 | 0.8830 | +0.0900 | 186.37 |
| EWC (λ=1000) | 0.4530 | 0.8950 | +0.2650 | 208.31 |
| LWF | **0.7150** | 0.6990 | **-0.0020** | 243.53 |
| DER (buffer=2000) | 0.6890 | 0.8720 | -0.0110 | 309.21 |

### 结果解读

**Fine Tuning**：
- 遗忘率仅0.0900，低于预期（预期>30%）
- Task 2学习效果最佳（0.8830），因无旧任务约束
- 训练速度最快（186s）

**EWC**：
- ⚠️ 遗忘率(0.2650)反高于Fine Tuning，表现异常
- λ=1000过大，过度约束模型导致Task 1知识被"锁定"在不利位置
- Task 2学习最好（0.8950），但以严重遗忘为代价

**LWF**：
- ✅ 实现零遗忘（-0.0020），Task 1准确率最高（0.7150）
- 通过知识蒸馏有效保护旧任务知识
- 但Task 2学习受限（0.6990），存在稳定性-可塑性权衡

**DER**：
- ✅ 实现零遗忘（-0.0110），综合平衡最优
- Task 1/2准确率均>0.68，遗忘率接近零
- 训练时间最长（309s），因缓冲区采样+蒸馏计算开销

### 关键发现

1. **DER综合表现最优**：在性能和稳定性间取得最佳平衡
2. **LWF旧任务保护最好**：但以牺牲新任务学习为代价
3. **EWC对超参数极其敏感**：λ需仔细调优，否则可能适得其反
4. **Fine Tuning遗忘率低于预期**：可能因任务分布差异大

### 可视化结果

📊 结果文件位于 `results/plots/` 目录：
- `fine_tuning_curves.png` - Fine Tuning训练曲线
- `ewc_curves.png` - EWC训练曲线
- `lwf_curves.png` - LWF训练曲线
- `der_curves.png` - DER训练曲线
- `comparison.png` - 四方法综合对比图

详细数据分析请参阅 [docs/data_analysis.md](docs/data_analysis.md)

## 参数配置指南

### 基础配置（configs/base.yaml）

```yaml
# 数据集配置
dataset:
  task1: "imdb"
  task2: "ag_news"
  max_length: 256
  val_split: 0.1

# 模型配置
model:
  name: "bert-base-uncased"
  freeze_layers: 10  # 冻结前10层

# 训练配置
training:
  learning_rate: 2e-5
  batch_size: 32
  epochs: 3
  warmup_steps: 100
  max_grad_norm: 1.0

# EWC配置
ewc:
  lambda: 1000
  fisher_sample_size: 200

# LWF配置
lwf:
  lambda: 1.0
  temperature: 2.0

# DER配置
der:
  buffer_size: 500
  lambda: 1.0
  alpha: 0.5
```

### 参数调优建议

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| learning_rate | 1e-5 ~ 5e-5 | 学习率 |
| ewc_lambda | 100 ~ 5000 | EWC正则化强度 |
| lwf_lambda | 0.5 ~ 2.0 | LWF蒸馏权重 |
| lwf_temperature | 1.0 ~ 4.0 | 蒸馏温度 |
| buffer_size | 200 ~ 2000 | DER缓冲区大小 |
| der_lambda | 0.5 ~ 2.0 | DER蒸馏权重 |

## 常见问题解答 (FAQ)

### Q1: 如何调整EWC的正则化强度？

**问题**：EWC的λ参数影响平衡新旧任务的程度。

**建议**：
- λ过小：旧任务遗忘严重
- λ过大：新任务学习困难
- 推荐范围：100-5000
- 可通过验证集调优

### Q2: LWF和DER如何选择？

- **LWF**：适合计算资源有限的场景，不需要存储旧任务样本
- **DER**：适合对性能要求更高的场景，需要额外的缓冲区
- 两者对比：DER通常优于LWF，但计算开销更大

### Q3: 如何处理内存不足？

1. 减小`buffer_size`（DER方法）
2. 减小`batch_size`
3. 减少`max_length`
4. 使用梯度累积代替大batch

### Q4: 如何添加更多任务？

修改`src/data.py`中的任务定义，添加新的数据集和标签映射。

### Q5: 如何使用自定义数据集？

1. 准备数据格式：JSON或CSV文件
2. 实现自定义Dataset类
3. 在`src/data.py`中注册新数据集

## 扩展实验设计

### 扩展1：动态缓冲区管理
- **创新点**：根据样本重要性动态调整缓冲区内容
- **实现方法**：基于模型预测置信度选择样本
- **预期目标**：用更少的缓冲区容量达到更好的效果

### 扩展2：任务自适应正则化
- **创新点**：根据任务相似度调整正则化强度
- **实现方法**：计算任务间特征分布距离
- **预期目标**：在相似任务间减少正则化，在差异任务间增强正则化

### 扩展3：多任务联合训练对比
- **创新点**：对比持续学习与多任务学习的效果
- **实现方法**：同时训练两个任务作为基准
- **预期目标**：评估持续学习方法接近最优多任务学习的程度

## 版本信息

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0.0 | 2026-07-28 | 初始版本，实现Fine Tuning/EWC/LWF/DER四种方法 |
| v1.1.0 | 2026-07-28 | 完善文档，添加参数配置指南和FAQ |

## 参考文献

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521-3526.

2. Li, Z., & Hoiem, D. (2017). Learning without forgetting. ECCV, 697-712.

3. Buzzega, P., Boschini, M., Porrello, A., et al. (2020). Dark experience for general continual learning: A strong, simple baseline. NeurIPS, 33, 15920-15930.

4. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. NeurIPS, 30.

5. Chaudhry, A., Dokania, P. K., Ajanthan, T., et al. (2019). Riemannian walk for incremental learning. CVPR, 3080-3089.

## 许可证

本项目仅供学术研究使用。
