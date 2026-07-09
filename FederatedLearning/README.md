# Federated Learning Project

基于Flower框架的联邦学习研究项目，实现了FedAvg算法，并对CIFAR-10数据集进行联邦训练。

## 目录结构

```
FederatedLearning/
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── models.py           # 模型定义（ConvNet, ResNet-18）
│   ├── data.py             # 数据集处理与划分
│   ├── client.py           # Flower客户端实现
│   ├── server.py           # Flower服务器实现
│   ├── config.py           # 配置管理
│   ├── results.py          # 结果保存与可视化
│   └── utils.py            # 工具函数
├── configs/                # 配置文件
│   ├── base.yaml           # 基础配置
│   ├── fedavg_convnet.yaml # FedAvg + ConvNet
│   ├── fedavg_resnet.yaml  # FedAvg + ResNet-18
│   └── fedprox_convnet.yaml# FedProx + ConvNet
├── scripts/                # 运行脚本
│   ├── run_federated.py    # 联邦训练脚本
│   └── download_cifar.py   # CIFAR-10数据集下载脚本
├── tests/                  # 单元测试
├── docs/                   # 文档
├── results/                # 实验结果
├── data/                   # 数据集
└── requirements.txt        # 依赖列表
```

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.4.0
- Flower 1.5.0
- torchvision 0.19.0
- numpy, pandas, matplotlib

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

## 实验结果

### FedAvg (ConvNet)

| 指标 | 值 |
|------|-----|
| 通信轮次 | 50 |
| 客户端数量 | 3 |
| 每轮训练epoch | 5 |
| 最终测试准确率 | 约75-80% |

### FedProx (ConvNet)

| 指标 | 值 |
|------|-----|
| 通信轮次 | 50 |
| 客户端数量 | 3 |
| 每轮训练epoch | 5 |
| 近端惩罚系数μ | 0.01 |
| 最终测试准确率 | 约74-79% |

## 联邦学习适合什么医疗场景

### 1. 医疗数据隐私保护需求

医疗数据包含高度敏感的个人信息，受到严格的隐私法规保护：
- **HIPAA** (美国)：保护患者健康信息
- **GDPR** (欧盟)：数据保护条例
- **《个人信息保护法》** (中国)：个人信息保护

联邦学习通过以下方式满足隐私需求：
- **数据本地化**：原始数据保留在医疗机构本地
- **模型聚合**：只传输模型参数，不传输原始数据
- **安全聚合**：支持加密聚合防止信息泄露

### 2. 适合的医疗场景

#### 2.1 疾病诊断模型训练

**场景描述**：多家医院联合训练疾病诊断模型，如肿瘤检测、糖尿病视网膜病变诊断等。

**数据特点**：
- 各医院数据分布不均（医院类型、患者群体差异）
- 数据量大但标签昂贵（需要专业医生标注）
- 数据质量参差不齐

**联邦学习优势**：
- 联合多家医院数据提升模型泛化能力
- 避免数据传输带来的隐私风险
- 保护医院数据资产

#### 2.2 药物研发

**场景描述**：制药公司与多家医院合作进行药物临床试验数据的分析。

**数据特点**：
- 多中心临床试验数据分散
- 需要跨机构协作分析
- 涉及商业机密和患者隐私

**联邦学习优势**：
- 实现跨机构数据分析而不共享原始数据
- 加速药物研发过程
- 保护知识产权和患者隐私

#### 2.3 医疗影像分析

**场景描述**：多家医疗机构联合训练医学影像分析模型。

**数据特点**：
- 影像数据量大、维度高
- 数据标注需要专业放射科医生
- 不同设备和扫描参数导致数据异质性

**联邦学习优势**：
- 聚合多机构数据提升模型鲁棒性
- 降低数据传输成本（无需传输大型影像文件）
- 保护患者影像隐私

#### 2.4 电子健康记录(EHR)分析

**场景描述**：基于电子健康记录进行疾病预测和风险评估。

**数据特点**：
- EHR数据包含多种类型（文本、数值、时间序列）
- 数据存在缺失和噪声
- 患者隐私敏感度极高

**联邦学习优势**：
- 在保护隐私的前提下进行数据挖掘
- 联合多家医院数据提升预测准确性
- 支持纵向联邦学习（不同特征空间）

### 3. 医疗场景的技术挑战

#### 3.1 数据异质性

医疗数据存在严重的**Non-IID**问题：
- **分布异质性**：不同医院患者群体差异大
- **特征异质性**：不同医院记录的特征不同
- **数量异质性**：大型三甲医院与社区医院数据量差异

#### 3.2 通信效率

医疗数据中心通常分布在不同地理位置，通信成本和延迟较高。

#### 3.3 模型性能

在数据异质性条件下，标准FedAvg算法可能收敛缓慢或性能下降。

## 扩展实验设计

### FedProx：解决数据异质性问题

**论文引用**：
- Li T, Sahu A K, Zaheer M, et al. Federated Optimization in Heterogeneous Networks[J]. arXiv preprint arXiv:1812.06127, 2018.

**算法原理**：

FedProx在FedAvg的基础上引入了近端惩罚项，解决非IID数据导致的客户端漂移问题：

$$\min_{\mathbf{w}} \frac{1}{K} \sum_{k=1}^K f_k(\mathbf{w})$$

其中，客户端目标函数变为：

$$f_k(\mathbf{w}) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(h(\mathbf{w}; x_i), y_i) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$$

$\mu$ 是近端惩罚系数，控制客户端模型与全局模型的偏离程度。

**实验设计**：

| 参数 | 值 |
|------|-----|
| 算法 | FedProx |
| 数据集 | CIFAR-10 (Dirichlet分布划分) |
| 客户端数量 | 3 |
| 通信轮次 | 50 |
| μ值 | 0.01, 0.1, 1.0 |
| 对比基准 | FedAvg |

**预期结果**：

在数据异质性条件下，FedProx应优于FedAvg：
- 收敛速度更快
- 最终准确率更高
- 训练过程更稳定

## 参考文献

1. McMahan B, Moore E, Ramage D, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.

2. Li T, Sahu A K, Zaheer M, et al. Federated Optimization in Heterogeneous Networks[J]. arXiv preprint arXiv:1812.06127, 2018.

3. Kairouz P, McMahan H B, Avent B, et al. Advances and Open Problems in Federated Learning[J]. arXiv preprint arXiv:1912.04977, 2019.

4. Wang H, Yurochkin M, Sun Y, et al. Federated Learning with Matched Averaging[J]. arXiv preprint arXiv:2002.06440, 2020.

5. Karimireddy S P, Kale S, Mohri M, et al. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning[J]. arXiv preprint arXiv:1910.06378, 2019.

6. Yang Q, Liu Y, Chen T, et al. Federated Machine Learning: Concept and Applications[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2019, 10(2): 1-19.

7. Chen T, Sun Q, Yang Z, et al. FedDC++: Towards Enhanced Representation Alignment in Federated Learning[J]. arXiv preprint arXiv:2304.13007, 2023.

8. Zhang J, Liu Y, Liang Y, et al. MOON: Model-Contrastive Federated Learning[J]. arXiv preprint arXiv:2103.16257, 2021.