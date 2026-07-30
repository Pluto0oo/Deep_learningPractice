# 联邦学习实验项目

## 项目概述

本项目基于Flower框架实现联邦学习（Federated Learning）算法，研究在数据异质性（Non-IID）条件下的模型训练问题。主要实现FedAvg和FedProx两种经典算法，并在SVHN数据集上进行实验验证。

## 目录结构

```
FederatedLearning/
├── src/                        # 核心源代码
│   ├── __init__.py
│   ├── models.py               # 模型定义（ConvNet, ResNet-18）
│   ├── data.py                 # 数据集处理与划分（IID/Dirichlet）
│   ├── client.py               # Flower客户端实现
│   ├── server.py               # Flower服务器实现
│   ├── config.py               # 配置管理
│   ├── results.py              # 结果保存与可视化
│   └── utils.py                # 工具函数
├── configs/                    # 配置文件
│   ├── base.yaml               # 基础配置
│   ├── fedavg_convnet.yaml     # FedAvg + ConvNet配置
│   ├── fedavg_resnet.yaml      # FedAvg + ResNet-18配置
│   └── fedprox_convnet.yaml    # FedProx + ConvNet配置
├── scripts/                    # 运行脚本
│   ├── run_federated.py        # 联邦训练主脚本
│   └── download_cifar.py       # 数据集下载脚本
├── tests/                      # 单元测试
│   ├── test_config.py
│   ├── test_data.py
│   └── test_models.py
├── docs/                       # 文档
│   └── experiment_principle.md # 实验原理与结果分析
├── results/                    # 实验结果
├── data/                       # 数据集存储
├── requirements.txt            # 依赖列表
├── .gitignore
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 主要依赖

| 库 | 版本要求 | 说明 |
|----|----------|------|
| Python | 3.9+ | 推荐使用conda环境 |
| PyTorch | 2.4.0+ | 深度学习框架 |
| Flower | 1.5.0+ | 联邦学习框架 |
| torchvision | 0.19.0+ | 视觉数据处理 |
| numpy | 2.0+ | 数值计算 |
| pandas | 2.0+ | 数据处理 |
| matplotlib | 3.9+ | 可视化绘图 |

## 快速开始

### 运行FedAvg实验

```bash
python scripts/run_federated.py configs/fedavg_convnet.yaml
```

### 运行FedProx实验（扩展实验）

```bash
python scripts/run_federated.py configs/fedprox_convnet.yaml
```

### 运行单元测试

```bash
pytest tests/ -v
```

## 实验设计

### 数据集

#### SVHN数据集
| 属性 | 描述 |
|------|------|
| 图像尺寸 | 32×32 彩色图像 |
| 类别数 | 10个数字类别（0-9） |
| 训练样本 | 73,257张 |
| 测试样本 | 26,032张 |
| 数据来源 | Google街景门牌号图像 |

#### CIFAR-10数据集（可选）
| 属性 | 描述 |
|------|------|
| 图像尺寸 | 32×32 彩色图像 |
| 类别数 | 10个物体类别 |
| 训练样本 | 50,000张 |
| 测试样本 | 10,000张 |

### 数据划分策略

#### IID划分
- 随机打乱所有数据
- 均匀分配给各客户端
- 各客户端数据来自相同分布

#### Dirichlet分布划分
- 模拟真实场景中的数据异质性
- 浓度参数α控制异质性程度
- α→∞：接近IID分布
- α→0：数据高度异质

### 模型架构

#### ConvNet
```
ConvNet架构：
├── Conv2d(3, 32, kernel_size=3, padding=1) + BN + ReLU
├── MaxPool2d(2)
├── Conv2d(32, 64, kernel_size=3, padding=1) + BN + ReLU
├── MaxPool2d(2)
├── Conv2d(64, 128, kernel_size=3, padding=1) + BN + ReLU
├── MaxPool2d(2)
├── Flatten
├── Linear(128×4×4, 256) + ReLU + Dropout(0.5)
└── Linear(256, 10)

参数量：约1.2M
```

#### ResNet-18
```
ResNet-18架构：
├── Conv2d(3, 64, kernel_size=7, stride=2, padding=3) + BN + ReLU
├── MaxPool2d(3, stride=2)
├── 4个残差块组（每组2个残差块）
├── AdaptiveAvgPool2d(1)
└── Linear(512, 10)

参数量：约11.7M
```

### 实验参数

#### FedAvg配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 算法 | FedAvg | 联邦平均算法 |
| 数据集 | SVHN | 使用SVHN数据集 |
| 数据划分 | IID | 独立同分布划分 |
| 客户端数量 | 3 | 参与客户端数 |
| 通信轮次 | 50 | 联邦训练轮数 |
| 每轮epoch | 5 | 本地训练轮数 |
| 批大小 | 32 | 本地批大小 |
| 优化器 | SGD | 随机梯度下降 |
| 学习率 | 0.01 | 学习率 |
| 动量 | 0.9 | SGD动量 |
| 权重衰减 | 1e-4 | L2正则化 |

#### FedProx配置（扩展实验）
| 参数 | 值 | 说明 |
|------|-----|------|
| 算法 | FedProx | 近端惩罚算法 |
| 数据集 | SVHN/CIFAR-10 | 数据集选择 |
| 数据划分 | Dirichlet(α=0.5) | 非IID划分 |
| 客户端数量 | 3 | 参与客户端数 |
| 通信轮次 | 50 | 联邦训练轮数 |
| 近端惩罚μ | 0.01 | 近端惩罚系数 |
| 其他参数 | 同FedAvg | 与FedAvg相同 |

## 技术架构

### FedAvg算法原理

```
FedAvg算法流程：
1. 初始化：服务器初始化全局模型 w₀
2. 客户端训练：
   - 每轮随机选择K个客户端
   - 每个客户端k基于本地数据D_k训练模型
   - 本地训练E个epoch
3. 参数上传：客户端将本地更新后的模型参数w_k上传到服务器
4. 模型聚合：
   - 服务器根据客户端数据量加权平均
   - w_{t+1} = Σ (n_k/n) × w_k^t
5. 全局更新：服务器更新全局模型并广播给所有客户端
```

### FedProx算法改进

```
FedProx在FedAvg基础上引入近端惩罚项：

客户端目标函数：
min_w  1/n_k × Σ ℓ(h(w; x_i), y_i) + μ/2 × ‖w - w^t‖²

其中：
- μ：近端惩罚系数
- w^t：全局模型参数
- μ越大，客户端模型越接近全局模型
- μ越小，客户端模型探索自由度越大
```

### Flower框架集成

```
Flower架构：
├── Server端
│   ├── strategy.py (FedAvg/FedProx策略)
│   ├── 管理客户端选择
│   └── 执行模型聚合
├── Client端
│   ├── client.py (NumPyClient实现)
│   ├── 本地数据加载
│   └── 本地训练与评估
└── 通信协议
    ├── gRPC通信
    └── 安全聚合支持
```

## 实验结果

### FedAvg实验结果（SVHN + IID + ConvNet）

#### 训练曲线

| 通信轮次 | 测试准确率 | 测试损失 |
|----------|------------|----------|
| 1 | 89.03% | 0.4003 |
| 5 | 92.79% | 0.4417 |
| 10 | 92.84% | 0.5999 |
| 15 | 92.93% | 0.5417 |
| 20 | 92.99% | 0.4995 |
| 25 | 92.80% | 0.4643 |
| 30 | 92.91% | 0.5029 |
| 35 | 92.82% | 0.4680 |
| 40 | 93.07% | 0.4713 |
| 45 | 93.04% | 0.4813 |
| 47 | 93.22% | 0.4915 |
| 50 | 93.07% | 0.4431 |

#### 关键统计
- **最终准确率**：93.07%（第50轮）
- **最佳准确率**：93.22%（第47轮）
- **平均准确率**：91.17%
- **收敛速度**：第1轮即达到89.03%

### 结果分析

#### 收敛性分析
```
训练阶段划分：
1. 快速收敛阶段 (0-5轮)：
   - 准确率从10.08%快速提升至92.79%
   - 损失从2.30快速下降至0.44
   - 模型快速学习基本特征

2. 微调阶段 (5-20轮)：
   - 准确率从92.79%缓慢提升至92.99%
   - 损失在0.44-0.60之间波动
   - 收敛速度减缓，进行精细调整

3. 稳定阶段 (20-50轮)：
   - 准确率在92.8%-93.2%区间波动
   - 损失稳定在0.44-0.55
   - 训练趋于稳定，接近最优
```

#### 关键发现
1. **FedAvg在SVHN上表现优异**：93.07%的准确率表明模型有效学习到数字特征
2. **收敛速度快**：仅1轮通信就达到89%以上准确率
3. **训练稳定**：5轮后准确率波动仅0.4%（92.8%-93.2%）
4. **损失与准确率负相关**：损失下降与准确率提升趋势一致

### 待完成实验

| 实验 | 状态 | 说明 |
|------|------|------|
| FedAvg + ResNet-18 | 待完成 | 更复杂模型的性能对比 |
| FedProx + ConvNet (IID) | 待完成 | FedProx在IID下与FedAvg对比 |
| FedProx + ConvNet (Non-IID) | 待完成 | FedProx在异质性数据下的表现 |
| 不同α值的Dirichlet划分 | 待完成 | 数据异质性对性能的影响 |
| 不同客户端数量 | 待完成 | 客户端数量对收敛的影响 |

## 参数配置指南

### 配置文件说明

#### base.yaml（基础配置）
```yaml
# 数据集配置
dataset:
  name: "svhn"           # 数据集名称
  data_dir: "./data"     # 数据存储目录
  download: false        # 是否自动下载

# 数据划分配置
partition:
  method: "iid"          # 划分方法：iid / dirichlet
  alpha: 0.5             # Dirichlet浓度参数（仅dirichlet时使用）
  num_clients: 3         # 客户端数量

# 模型配置
model:
  name: "convnet"        # 模型名称：convnet / resnet18
  num_classes: 10        # 分类数

# 联邦训练配置
federated:
  num_rounds: 50         # 通信轮数
  num_clients: 3         # 客户端数量
  fraction_fit: 1.0      # 参与训练的客户端比例
  min_fit_clients: 3     # 最少参与客户端数

# 本地训练配置
training:
  local_epochs: 5        # 本地训练轮数
  batch_size: 32         # 批大小
  learning_rate: 0.01    # 学习率
  momentum: 0.9          # SGD动量
  weight_decay: 1e-4     # 权重衰减

# FedProx配置（仅FedProx时使用）
fedprox:
  mu: 0.01               # 近端惩罚系数

# 输出配置
output:
  results_dir: "./results"
  save_checkpoint: true
  plot_curves: true
```

### 参数调优建议

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| learning_rate | 0.001 ~ 0.1 | 学习率 |
| local_epochs | 3 ~ 10 | 本地训练轮数 |
| batch_size | 16 ~ 64 | 批大小 |
| num_rounds | 30 ~ 100 | 通信轮数 |
| mu (FedProx) | 0.001 ~ 1.0 | 近端惩罚系数 |
| alpha (Dirichlet) | 0.1 ~ 5.0 | 数据异质性程度 |

## 联邦学习在医疗场景的应用

### 适合的医疗场景

#### 1. 疾病诊断模型训练
- **场景**：多家医院联合训练疾病诊断模型
- **优势**：联合数据提升泛化能力，保护患者隐私
- **示例**：肿瘤检测、糖尿病视网膜病变诊断

#### 2. 药物研发
- **场景**：制药公司与多家医院合作
- **优势**：跨机构数据协作，保护商业机密
- **示例**：药物反应预测、临床试验分析

#### 3. 医疗影像分析
- **场景**：多机构联合训练影像分析模型
- **优势**：降低影像传输成本，保护患者影像隐私
- **示例**：医学影像分割、影像报告生成

#### 4. 电子健康记录分析
- **场景**：基于EHR进行疾病风险预测
- **优势**：在保护隐私的前提下进行大数据分析
- **示例**：慢性病管理、公共卫生监测

### 技术挑战

#### 数据异质性
- **分布漂移**：不同医院患者群体差异
- **特征缺失**：不同医院记录的特征不同
- **数据量差异**：大医院与小医院数据量差异

**解决方案**：
- FedProx、SCAFFOLD等算法
- 自适应聚合策略
- 元学习方法

#### 通信效率
- **带宽限制**：模型参数传输需要较大带宽
- **延迟要求**：实时诊断场景对延迟敏感

**解决方案**：
- 模型压缩（量化、剪枝）
- 异步联邦学习
- 边缘计算部署

#### 隐私保护
- **安全聚合**：密码学技术实现隐私保护
- **差分隐私**：在模型参数中添加噪声
- **同态加密**：支持加密状态下的计算

## 常见问题解答 (FAQ)

### Q1: 如何修改客户端数量？

修改配置文件中的`federated.num_clients`参数：
```yaml
federated:
  num_clients: 5  # 改为5个客户端
```

### Q2: 如何切换FedAvg和FedProx？

使用不同的配置文件：
```bash
# FedAvg
python scripts/run_federated.py configs/fedavg_convnet.yaml

# FedProx
python scripts/run_federated.py configs/fedprox_convnet.yaml
```

### Q3: 如何使用Dirichlet划分？

```yaml
partition:
  method: "dirichlet"
  alpha: 0.5  # 调整异质性程度
```

### Q4: 如何处理数据集下载失败？

1. 手动下载数据集到`data/`目录
2. 或修改配置设置`download: true`
3. CIFAR-10可手动下载tar.gz文件

### Q5: 如何查看实验结果？

```bash
# 查看结果目录
ls results/fedavg_convnet_*/

# 运行对比脚本
python scripts/compare_results.py
```

### Q6: GPU内存不足怎么办？

1. 减小`batch_size`（如从32改为16）
2. 减小`local_epochs`
3. 使用更小的模型（ConvNet替代ResNet-18）

### Q7: 如何扩展到更多数据集？

1. 在`src/data.py`中添加新数据集类
2. 实现数据划分逻辑
3. 在配置文件中添加对应配置

## 扩展实验设计

### 扩展1：FedProx在异质性数据下的性能

**研究问题**：
- FedProx如何缓解数据异质性带来的客户端漂移？
- 近端惩罚系数μ如何影响模型性能？

**实验设计**：
| 参数 | 取值 |
|------|------|
| 算法 | FedAvg, FedProx |
| 数据划分 | IID, Dirichlet(α=0.1, 0.5, 1.0, 5.0) |
| μ值 | 0.001, 0.01, 0.1, 1.0 |

**预期结果**：
1. α越小（数据越异质），FedProx优势越明显
2. 存在最优μ值
3. 在IID数据下，两者性能相近

### 扩展2：模型架构影响

**研究问题**：
- 不同模型架构在联邦学习下的性能表现
- 模型复杂度与通信效率的权衡

**实验设计**：
| 模型 | 参数量 | 特点 |
|------|--------|------|
| ConvNet | ~1.2M | 轻量级，适合移动端 |
| ResNet-18 | ~11.7M | 复杂模型，性能更好 |
| MobileNet | ~3.2M | 轻量级，平衡性能与效率 |

### 扩展3：客户端数量影响

**研究问题**：
- 客户端数量对收敛速度的影响
- 最优客户端参与比例

**实验设计**：
| 客户端数量 | 参与比例 | 预期效果 |
|------------|----------|----------|
| 3 | 1.0 | 基线设置 |
| 5 | 0.5, 1.0 | 中等规模 |
| 10 | 0.3, 0.5, 1.0 | 大规模 |
| 20 | 0.1, 0.3 | 超大规模 |

## 版本信息

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0.0 | 2026-07-28 | 初始版本，实现FedAvg算法及基础实验 |
| v1.1.0 | 2026-07-28 | 完善文档，添加参数配置指南和扩展实验设计 |

## 参考文献

1. McMahan B, Moore E, Ramage D, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, 2017: 1273-1282.

2. Li T, Sahu A K, Zaheer M, et al. Federated Optimization in Heterogeneous Networks. arXiv:1812.06127, 2018.

3. Kairouz P, McMahan H B, Avent B, et al. Advances and Open Problems in Federated Learning. arXiv:1912.04977, 2019.

4. Karimireddy S P, Kale S, Mohri M, et al. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. arXiv:1910.06378, 2019.

5. Yang Q, Liu Y, Chen T, et al. Federated Machine Learning: Concept and Applications. ACM TIST, 2019, 10(2): 1-19.

## 许可证

本项目仅供学术研究使用。
