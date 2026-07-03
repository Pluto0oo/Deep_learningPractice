# 实验结果报告

**生成时间**: 2026-07-03 17:20:53

## 概述

本报告总结了共 4 个主动学习实验的结果。

## 实验摘要

| 实验ID | 名称 | 模型 | 随机种子 | 重复次数 |
|--------|------|-------|----------|----------|
| `active_learning_imdb_20260701_115303` | active_learning_imdb | LogisticRegression | 42 | 3 |
| `active_learning_imdb_20260703_110537` | active_learning_imdb | LogisticRegression | 42 | 3 |
| `active_learning_imdb_20260703_111225` | active_learning_imdb | LogisticRegression | 42 | 3 |
| `active_learning_imdb_20260703_171937` | active_learning_imdb | LogisticRegression | 42 | 1 |

## 性能指标

### 采样效率对比（准确率）

| 实验ID | 采样策略 | 10%标注 | 30%标注 | 50%标注 |
|--------|----------|---------|---------|---------|
| `active_learning_imdb_20260701_115303` | Random | 0.7167 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260701_115303` | Entropy | 0.7420 | 0.8877 | 0.9450 |
| `active_learning_imdb_20260703_110537` | Random | 0.7167 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260703_110537` | Entropy | 0.7420 | 0.8877 | 0.9450 |
| `active_learning_imdb_20260703_110537` | Margin | 0.7420 | 0.8877 | 0.9450 |
| `active_learning_imdb_20260703_111225` | Random | 0.7167 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260703_111225` | Entropy | 0.7420 | 0.8877 | 0.9450 |
| `active_learning_imdb_20260703_111225` | Margin | 0.7420 | 0.8877 | 0.9450 |
| `active_learning_imdb_20260703_171937` | Random | 0.8551 | 0.8735 | 0.8831 |
| `active_learning_imdb_20260703_171937` | Entropy | 0.8627 | 0.8725 | 0.8873 |
| `active_learning_imdb_20260703_171937` | Margin | 0.8627 | 0.8725 | 0.8873 |
| `active_learning_imdb_20260703_171937` | Uncertainty | 0.8627 | 0.8725 | 0.8873 |

### F1分数

| 实验ID | 采样策略 | 10%标注 | 30%标注 | 50%标注 |
|--------|----------|---------|---------|---------|
| `active_learning_imdb_20260701_115303` | Random | 0.7164 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260701_115303` | Entropy | 0.7418 | 0.8876 | 0.9450 |
| `active_learning_imdb_20260703_110537` | Random | 0.7164 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260703_110537` | Entropy | 0.7418 | 0.8876 | 0.9450 |
| `active_learning_imdb_20260703_110537` | Margin | 0.7418 | 0.8876 | 0.9450 |
| `active_learning_imdb_20260703_111225` | Random | 0.7164 | 0.8553 | 0.9080 |
| `active_learning_imdb_20260703_111225` | Entropy | 0.7418 | 0.8876 | 0.9450 |
| `active_learning_imdb_20260703_111225` | Margin | 0.7418 | 0.8876 | 0.9450 |
| `active_learning_imdb_20260703_171937` | Random | 0.8585 | 0.8757 | 0.8844 |
| `active_learning_imdb_20260703_171937` | Entropy | 0.8585 | 0.8770 | 0.8881 |
| `active_learning_imdb_20260703_171937` | Margin | 0.8585 | 0.8770 | 0.8881 |
| `active_learning_imdb_20260703_171937` | Uncertainty | 0.8585 | 0.8770 | 0.8881 |

## 最佳性能

- **50%标注时最佳准确率**: `active_learning_imdb_20260701_115303` (active_learning_imdb) - 0.9080

## 配置对比

### 模型架构

- **LogisticRegression**: 4 个实验

### 实验设置

- **随机种子**: 42
- **重复次数**: 1, 3
