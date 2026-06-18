# cv_finetune - 实验报告

## 一、实验概述

**实验名称**: cv_finetune  
**实验时间**: 2026-06-18 19:11:47

---

## 二、实验配置

```json
{
  "model": "SimpleCNN (Finetune)",
  "task": "MNIST -> MNIST-M",
  "method": "Fine-tuning",
  "dataset": {
    "source": "MNIST",
    "target": "MNIST-M",
    "train_samples": 10000,
    "test_samples": 10000
  },
  "hyperparameters": {
    "batch_size": 64,
    "pretrain_epochs": 10,
    "finetune_epochs": 5,
    "pretrain_lr": 0.001,
    "finetune_lr": 0.0001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss"
  }
}
```

---

## 三、系统信息

| 项目 | 值 |
|------|------|
| 操作系统 | Windows |
| Python版本 | 3.10.20 |
| CUDA可用 | True |
| CUDA设备 | NVIDIA GeForce RTX 5060 Laptop GPU |

---

## 四、训练日志

### 4.1 训练曲线摘要

| 轮次 | 损失 | 准确率 |
|------|------|--------|
| 1 | 0.520271 | 0.8359 |
| 2 | 0.114835 | 0.9629 |
| 3 | 0.075444 | 0.9772 |
| 4 | 0.055266 | 0.9818 |
| 5 | 0.042290 | 0.9869 |
| 6 | 0.029837 | 0.9904 |
| 7 | 0.024656 | 0.9922 |
| 8 | 0.015372 | 0.9955 |
| 9 | 0.012212 | 0.9959 |
| 10 | 0.009862 | 0.9968 |
| 11 | 1.553938 | 0.5390 |
| 12 | 0.584796 | 0.8568 |
| 13 | 0.325473 | 0.9167 |
| 14 | 0.227379 | 0.9424 |
| 15 | 0.178710 | 0.9526 |

---

## 五、评估指标

### 5.1 target_test数据集

| 指标 | 值 |
|------|------|
| accuracy | 0.95 |
| correct | 9500 |
| total | 10000 |

---

## 六、模型信息

### 6.1 finetune_model

- **保存路径**: results\cv_finetune_20260618_191130\models\finetune_model.pth
- **参数数量**: 390,986
- **可训练参数**: 390,986

---

## 七、实验结果文件

| 类型 | 路径 |
|------|------|
| 配置文件 | `results\cv_finetune_20260618_191130\config.json` |
| 训练日志 | `results\cv_finetune_20260618_191130\logs\training.log` |
| 评估指标 | `results\cv_finetune_20260618_191130\metrics` |
| 模型文件 | `results\cv_finetune_20260618_191130\models` |
| 可视化结果 | `results\cv_finetune_20260618_191130\visualizations` |

---

*报告生成时间: 2026-06-18 19:11:47*
