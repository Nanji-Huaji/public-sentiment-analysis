# 基于自然语言处理的立场检测-舆情分析系统
使用微调的mBERT实现立场检测。



## 当前目录结构
```
public-sentiment-analysis/
│
├── README.md                   # 项目说明文件
│
├── data/                       # 数据文件夹
|   ├── raw                     # 找到的数据集
|   └── processed               # 处理过的数据集                    
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

## 实现步骤
### 处理数据集
我们采用VAST数据集（英文）、Weibo-SD数据集（中文）、NLPCC-2016数据集（中文）微调mBERT。

这些数据集中，前者是`.csv`格式，后两者是`.json`格式，需要处理成`.csv`格式便于调用。
