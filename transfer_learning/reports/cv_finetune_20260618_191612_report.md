# cv_finetune - 实验报告

## 一、实验概述

**实验名称**: cv_finetune  
**实验时间**: 2026-06-18 19:16:43

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
| 1 | 0.512879 | 0.8424 |
| 2 | 0.131383 | 0.9589 |
| 3 | 0.077755 | 0.9769 |
| 4 | 0.053411 | 0.9829 |
| 5 | 0.037845 | 0.9878 |
| 6 | 0.028249 | 0.9915 |
| 7 | 0.020391 | 0.9936 |
| 8 | 0.024134 | 0.9914 |
| 9 | 0.013613 | 0.9956 |
| 10 | 0.020686 | 0.9933 |
| 11 | 1.450261 | 0.5717 |
| 12 | 0.410735 | 0.8993 |
| 13 | 0.214064 | 0.9427 |
| 14 | 0.153875 | 0.9575 |
| 15 | 0.121195 | 0.9684 |

---

## 五、评估指标

### 5.1 target_test数据集

| 指标 | 值 |
|------|------|
| accuracy | 0.9612 |
| correct | 9612 |
| total | 10000 |

---

## 六、模型信息

### 6.1 finetune_model

- **保存路径**: results\cv_finetune_20260618_191612\models\finetune_model.pth
- **参数数量**: 390,986
- **可训练参数**: 390,986

---

## 七、实验结果文件

| 类型 | 路径 |
|------|------|
| 配置文件 | `results\cv_finetune_20260618_191612\config.json` |
| 训练日志 | `results\cv_finetune_20260618_191612\logs\training.log` |
| 评估指标 | `results\cv_finetune_20260618_191612\metrics` |
| 模型文件 | `results\cv_finetune_20260618_191612\models` |
| 可视化结果 | `results\cv_finetune_20260618_191612\visualizations` |

---

*报告生成时间: 2026-06-18 19:16:43*
