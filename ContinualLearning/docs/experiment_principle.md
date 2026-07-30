# 持续学习实验原理文档

## 1. 实验概述

### 1.1 研究背景

持续学习（Continual Learning）旨在解决深度学习模型在学习新任务时遗忘旧任务知识的问题，即"灾难性遗忘"（Catastrophic Forgetting）。在实际应用中，模型需要能够顺序学习多个任务而不遗忘先前学到的知识。

本实验聚焦于文本分类任务的持续学习场景，通过模拟两个分布不同的文本分类任务的顺序训练，对比多种持续学习方法的性能。

### 1.2 实验目标

1. 对比直接微调与经验回放机制在持续学习中的表现
2. 评估EWC、LWF、DER等经典持续学习方法的效果
3. 测量各方法的遗忘率和计算效率
4. 分析不同方法在文本分类任务中的适用性

### 1.3 任务设计

| 任务 | 数据集 | 类别数 | 任务描述 |
|------|--------|--------|----------|
| Task 1 | IMDB | 2 | 情感分类（正面/负面） |
| Task 2 | AG News | 4 | 新闻主题分类（World/Sports/Business/Tech） |

两个任务的数据分布差异较大：
- IMDB：电影评论情感分析，文本较长，语言较口语化
- AG News：新闻标题分类，文本较短，语言较正式

---

## 2. 实验方法

### 2.1 方法一：直接微调（Fine Tuning）

**原理**：在每个新任务上直接微调预训练模型，不采用任何持续学习机制。

**实现步骤**：
1. 使用预训练BERT模型作为基础编码器
2. 在Task 1上训练模型
3. 在Task 2上继续微调同一模型

**预期问题**：训练Task 2时，模型参数会被大幅更新，导致对Task 1的性能下降（灾难性遗忘）。

### 2.2 方法二：微调+经验回放（Experience Replay）

**原理**：维护一个经验回放缓冲区，存储先前任务的样本。在训练新任务时，同时从缓冲区中采样旧任务样本进行训练。

**实现步骤**：
1. 训练Task 1时，将部分样本存入缓冲区
2. 训练Task 2时，每个batch包含新任务样本和缓冲区中采样的旧任务样本
3. 使用混合损失进行优化

### 2.3 方法三：弹性权重巩固（EWC）

**原理**（Kirkpatrick et al., 2017）：通过计算Fisher信息矩阵识别对旧任务重要的权重参数，并在训练新任务时对这些参数施加正则化约束。

**实现步骤**：
1. 训练Task 1
2. 在Task 1数据上计算Fisher信息矩阵
3. 训练Task 2时，添加EWC正则化项：
   $$L_{EWC} = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$
   其中$F_i$是Fisher信息，$\theta_i^*$是Task 1训练后的参数

**创新点**：保护重要参数不被过度修改

### 2.4 方法四：无遗忘学习（LWF）

**原理**（Li & Hoiem, 2017）：使用知识蒸馏技术，让新模型在学习新任务的同时，保持对旧任务的预测能力。

**实现步骤**：
1. 训练Task 1，保存模型为"教师模型"
2. 训练Task 2时，同时最小化：
   - 新任务的分类损失
   - 新旧模型在旧任务上预测的KL散度

**创新点**：通过蒸馏损失保留旧任务知识

### 2.5 方法五：暗经验回放（DER）

**原理**（Buzzega et al., 2020）：结合经验回放和知识蒸馏，使用旧模型对缓冲区样本的预测作为软标签。

**实现步骤**：
1. 训练Task 1，保存模型和部分样本到缓冲区
2. 训练Task 2时：
   - 计算新任务样本的分类损失
   - 计算缓冲区样本的分类损失
   - 计算缓冲区样本上新旧模型预测的蒸馏损失

**创新点**：同时利用硬标签（缓冲区样本）和软标签（旧模型预测）

---

## 3. 实验设计

### 3.1 数据集

**IMDB数据集**：
- 25,000条训练样本，25,000条测试样本
- 类别：正面评论、负面评论
- 平均文本长度：约230词

**AG News数据集**：
- 120,000条训练样本，7,600条测试样本
- 类别：World、Sports、Business、Technology
- 平均文本长度：约12词

### 3.2 模型架构

**基础模型**：bert-base-uncased
- 12层Transformer encoder
- 768维隐藏层
- 12个注意力头

**分类器**：
- Dropout层（p=0.1）
- 线性分类层（输出维度=总类别数）

**参数设置**：
- 仅微调最后2层Transformer和分类器
- 学习率：2e-5
- 批大小：32
- 训练轮数：3

### 3.3 评估指标

#### 3.3.1 准确率（Accuracy）
$$Accuracy = \frac{\text{正确预测数}}{\text{总样本数}}$$

#### 3.3.2 遗忘率（Forgetting Rate）
$$Forgetting = \max_{t \leq T} Acc(t) - Acc(T)$$
其中$Acc(t)$是在任务t训练后对任务t的准确率，$Acc(T)$是在所有任务训练完成后对任务t的准确率。

#### 3.3.3 F1分数（F1 Score）
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

#### 3.3.4 计算效率指标
- 训练总时间
- GPU内存占用（Allocated/Reserved）

### 3.4 对比实验设计

| 对比维度 | 实验设置 | 目的 |
|----------|----------|------|
| 有无持续学习机制 | Fine Tuning vs EWC/LWF/DER | 验证持续学习方法的有效性 |
| 经验回放策略 | ER vs DER | 验证蒸馏损失的作用 |
| 正则化强度 | EWC不同λ值 | 分析正则化参数影响 |
| 缓冲区大小 | DER不同buffer_size | 分析缓冲区大小影响 |

---

## 4. 实现细节

### 4.1 数据加载模块（src/data.py）

**TextDataset类**：
- 使用HuggingFace Tokenizer处理文本
- 将文本转换为模型输入格式（input_ids, attention_mask）
- 支持最大序列长度截断

**ContinualDataLoader类**：
- 分别加载每个任务的训练/验证/测试集
- 支持随机划分验证集

**ExperienceBuffer类**：
- 均匀采样策略
- 固定缓冲区大小，超出时丢弃最早样本

### 4.2 模型模块（src/models.py）

**TextClassifier类**：
- 继承PyTorch nn.Module
- 使用AutoModel加载预训练BERT
- 冻结底层参数，仅训练顶层

**EWCModel类**：
- 扩展TextClassifier
- 添加Fisher信息矩阵计算方法

**LWFModel类**：
- 扩展TextClassifier
- 保存旧模型用于知识蒸馏

**DERModel类**：
- 扩展TextClassifier
- 同时保存缓冲区和旧模型

### 4.3 训练模块（src/trainer.py）

**Trainer基类**：
- 通用训练循环
- 学习率预热调度
- 梯度累积和裁剪

**EWCTrainer类**：
- 在损失中添加EWC正则化项

**LWFTrainer类**：
- 在损失中添加蒸馏损失项

**DERTrainer类**：
- 混合新任务损失、缓冲区损失和蒸馏损失

### 4.4 评估模块（src/evaluator.py）

**Evaluator类**：
- 计算准确率、F1分数
- 计算遗忘率
- 测量内存使用
- 保存实验结果

### 4.5 可视化模块（src/visualization.py）

**Visualizer类**：
- 训练曲线绘制
- 任务准确率对比图
- 遗忘率对比图
- 运行时间和内存对比图

---

## 5. 实验结果与分析

### 5.1 实验运行状态

**⚠️ 注意**：当前文档中的结果为**预期结果**，实际实验需要通过运行以下命令获取：

```bash
# 运行全部方法对比实验
python scripts/run_comparison.py --config configs/base.yaml

# 或单独运行各方法
python scripts/run_fine_tuning.py --config configs/base.yaml
python scripts/run_ewc.py --config configs/base.yaml
python scripts/run_lwf.py --config configs/base.yaml
python scripts/run_der.py --config configs/base.yaml
```

实验完成后，结果将保存至 `results/` 目录，包括：
- 各方法的准确率曲线
- 遗忘率对比图
- 训练时间与内存占用统计

### 5.2 预期结果（待实际实验验证）

| 方法 | Task 1准确率 | Task 2准确率 | 遗忘率 |
|------|-------------|-------------|--------|
| Fine Tuning | 低（遗忘严重） | 高 | 高 |
| EWC | 较高 | 较高 | 较低 |
| LWF | 较高 | 较高 | 较低 |
| DER | 高 | 高 | 低 |

### 5.3 预期结果解读

**Fine Tuning（直接微调基线）**：
- Task 2训练会覆盖Task 1的知识
- 预期遗忘率最高（>30%）
- 作为其他方法的对比基线
- 训练速度最快，内存占用最低

**EWC（弹性权重巩固）**：
- 通过正则化保护重要参数
- 预期遗忘率中等（10-20%）
- 可能影响新任务学习速度
- 正则化强度λ需要仔细调优

**LWF（无遗忘学习）**：
- 通过蒸馏保留旧任务知识
- 预期遗忘率较低（5-15%）
- 需要额外存储旧模型（教师模型）
- 对类别重叠的任务效果更好

**DER（暗经验回放）**：
- 结合经验回放和蒸馏
- 预期遗忘率最低（<10%）
- 计算开销最大（需存储缓冲区和旧模型）
- 预期效果最优

### 5.4 计算效率分析

| 方法 | 内存占用 | 训练时间 | 额外存储 |
|------|----------|----------|----------|
| Fine Tuning | 低 | 短 | 无 |
| EWC | 中（存储Fisher矩阵） | 中（额外计算Fisher） | Fisher信息矩阵 |
| LWF | 高（存储旧模型） | 中（额外前向传播） | 教师模型权重 |
| DER | 高（存储缓冲区+旧模型） | 长（额外采样+前向传播） | 缓冲区样本+教师模型 |

### 5.5 结果验证要点

实验运行后，需重点验证以下内容：

1. **遗忘率对比**：
   - 确认Fine Tuning的遗忘率显著高于其他方法
   - 验证DER是否能有效降低遗忘率

2. **准确率权衡**：
   - 观察EWC/LWF/DER是否在降低遗忘率的同时
   - 保持对新任务的学习能力

3. **计算效率**：
   - 记录各方法的实际训练时间
   - 测量GPU内存占用（torch.cuda.max_memory_allocated）

4. **超参数敏感性**：
   - EWC的λ值对遗忘率的影响
   - DER的buffer_size对性能的影响

---

## 6. 创新点与扩展

### 6.1 扩展实验设计

#### 扩展实验1：动态缓冲区管理
- **创新点**：根据样本重要性动态调整缓冲区内容
- **实现方法**：基于模型预测置信度选择样本
- **预期目标**：用更少的缓冲区容量达到更好的效果

#### 扩展实验2：任务自适应正则化
- **创新点**：根据任务相似度调整正则化强度
- **实现方法**：计算任务间特征分布距离
- **预期目标**：在相似任务间减少正则化，在差异任务间增强正则化

#### 扩展实验3：多任务联合训练对比
- **创新点**：对比持续学习与多任务学习的效果
- **实现方法**：同时训练两个任务作为基准
- **预期目标**：评估持续学习方法接近最优多任务学习的程度

### 6.2 技术贡献

1. 将经典持续学习方法应用于文本分类任务
2. 实现了完整的评估框架（遗忘率、计算效率）
3. 提供了可复现的实验代码和可视化工具

---

## 7. 结论与未来工作

### 7.1 结论

预期DER方法在文本分类持续学习任务中表现最优，能够有效缓解灾难性遗忘。EWC和LWF方法也能显著降低遗忘率，但效果不如DER。

### 7.2 未来工作

1. 探索更先进的持续学习方法（如GEM、GDumb）
2. 研究预训练模型在持续学习中的特殊行为
3. 扩展到更多任务的持续学习场景
4. 探索领域自适应的持续学习策略

---

## 8. 参考文献

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hassabis, D. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521-3526.

2. Li, Z., & Hoiem, D. (2017). Learning without forgetting. In Proceedings of the European conference on computer vision (ECCV) (pp. 697-712).

3. Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark experience for general continual learning: A strong, simple baseline. Advances in Neural Information Processing Systems, 33, 15920-15930.

4. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. Advances in Neural Information Processing Systems, 30.

5. Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. (2019). Riemannian walk for incremental learning: Understanding forgetting and intransigence. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3080-3089.