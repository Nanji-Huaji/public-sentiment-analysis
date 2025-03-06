# 基于自然语言处理的立场检测-舆情分析系统
使用微调的BERT实现立场检测。



## 当前目录结构
```
public-sentiment-analysis/
│
├── README.md                   # 项目说明文件
│
├── data/                       # 数据文件夹
│
├── models/                     # 模型文件夹
│   └── fine_tuned_model/       # 微调后的模型
│
├── src/                        # 源代码文件夹（后端）
│   ├── dataset.py              # 实现torch.utils.data.Dataset
│   ├── model.py                # 定义模型架构和加载预训练模型
│   └── train.py                # 训练脚本，包含训练循环、损失计算、优化等
│
├── frontend/                   # 前端文件夹
│
├── scripts/                    # 脚本文件夹
│
└── logs/                       # 日志文件夹

```