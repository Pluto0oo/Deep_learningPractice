# ActiveLearning

主动学习研究的机器学习实验管理框架。

## 功能特性

- **实验管理**：支持唯一的实验ID、重复实验和详细日志记录
- **配置系统**：基于YAML的实验、数据、模型和训练配置
- **结果追踪**：自动保存指标、模型权重和实验摘要
- **对比工具**：支持多实验对比并生成报告
- **结果汇总**：汇总所有实验结果并生成综合报告

## 项目结构

```
ActiveLearning/
├── configs/          # 配置文件（YAML格式）
├── data/             # 数据目录
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后的数据
├── src/              # 源代码
│   ├── utils.py      # 工具函数
│   ├── data_loader.py # 数据加载工具
│   ├── model.py      # 模型定义
│   ├── trainer.py    # 训练逻辑
│   └── evaluator.py  # 评估逻辑
├── scripts/          # 实验脚本
│   ├── run_experiment.py      # 运行单个实验
│   ├── run_comparison.py      # 运行并对比多个实验
│   ├── aggregate_results.py   # 汇总所有实验结果
│   ├── generate_report.py     # 生成综合报告
│   └── analyze_results.py     # 分析和可视化结果
├── notebooks/        # Jupyter笔记本（用于数据分析）
├── results/          # 实验结果（按exp_id分类）
├── logs/             # 日志文件（按exp_id分类）
├── reports/          # 生成的报告
└── tests/            # 测试文件
```

## 安装

### Anaconda 环境

本项目使用 Anaconda 环境 `dl-gpu`。

**环境详情：**
- Python 3.10
- PyTorch 2.12.0（CUDA 12.8）
- NumPy 2.2.6、Pandas 2.3.3
- scikit-learn 1.7.1、matplotlib 3.10.9、seaborn 0.13.2
- tqdm 4.67.3、pyyaml 6.0.3

```bash
# 激活环境
conda activate dl-gpu

# 安装缺失的包
pip install -r requirements.txt
```

## 使用方法

### 运行单个实验

```bash
python scripts/run_experiment.py --config configs/example_experiment.yaml
```

### 使用自定义实验ID

```bash
python scripts/run_experiment.py --config configs/example_experiment.yaml --exp_id my_experiment_001
```

### 对比多个实验

```bash
python scripts/run_comparison.py --configs configs/base.yaml configs/example_experiment.yaml --output reports/comparison.md
```

### 汇总结果

```bash
python scripts/aggregate_results.py --results_dir results --output_dir reports
```

### 生成报告

```bash
python scripts/generate_report.py --results_dir results --output reports/final_report.md
```

### 分析结果

```bash
python scripts/analyze_results.py --results_dir results --output_dir reports/plots
```

## 配置说明

配置文件使用 YAML 格式，包含以下部分：

```yaml
experiment:
  name: experiment_name    # 实验名称
  seed: 42                 # 随机种子（用于复现）
  repeat_times: 3          # 重复实验次数

data:
  path: data/processed     # 数据路径
  test_size: 0.2           # 测试集比例
  val_size: 0.1            # 验证集比例

model:
  type: SimpleMLP          # 模型类型
  input_dim: 20            # 输入维度
  hidden_dim: 128          # 隐藏层维度
  num_layers: 2            # 隐藏层数量
  output_dim: 10           # 输出维度（类别数）
  dropout_rate: 0.5        # Dropout率

training:
  epochs: 100              # 训练轮数
  batch_size: 32           # 批次大小
  loss: cross_entropy      # 损失函数
  optimizer: adam          # 优化器
  learning_rate: 0.001     # 学习率
  weight_decay: 0.0        # 权重衰减
  scheduler: cosine        # 学习率调度器
  save_best: true          # 是否保存最佳模型

evaluation:
  metrics:                 # 评估指标
    - accuracy
    - precision
    - recall
    - f1
```

## 结果结构

每个实验的结果保存在 `results/exp_id/` 目录下：

```
results/exp_id/
├── config_used.yaml   # 实验使用的配置
├── metrics.csv        # 每个epoch的指标（CSV格式）
├── metrics.json       # 最终指标（JSON格式）
├── summary.md         # 自动生成的摘要（Markdown格式）
├── plots/             # 生成的图表
└── checkpoints/       # 模型权重
    ├── best_model.pt  # 最佳模型（按验证准确率）
    └── final_model.pt # 最终模型

如果 repeat_times > 1：
└── repeats/
    ├── repeat_000/    # 第0次重复的结果
    ├── repeat_001/    # 第1次重复的结果
    └── stats/         # 所有重复实验的统计汇总
```

## 测试

```bash
pytest tests/ -v
```

## 许可证

MIT License