# 基于自然语言处理的立场检测-舆情分析系统
使用微调的BERT实现立场检测。

## 当前目录结构
```
Stance_Detection_Web_App/
│
├── README.md                   # 项目说明文件
│
├── data/                       # 数据文件夹
│   ├── processed/              # 预处理后的数据
│   │   ├── train.csv           # 训练集
│   │   ├── val.csv             # 验证集
│   │   └── test.csv            # 测试集
│   └── raw/                    # 原始 VAST 数据
│
├── models/                     # 模型文件夹
│   ├── pretrained_model/       # 预训练的 BERT 模型
│   └── fine_tuned_model/       # 微调后的模型
│
├── src/                        # 源代码文件夹（后端）
│   ├── data_processing.py      # 数据预处理脚本
│   ├── model.py                # 定义模型架构和加载预训练模型
│   ├── train.py                # 训练脚本，包含训练循环、损失计算、优化等
│   ├── evaluate.py             # 评估脚本，用于在验证集和测试集上评估模型
│   ├── utils.py                # 工具函数，例如日志记录、性能指标计算等
│   └── api.py                  # 后端 API 路由和逻辑
│
├── frontend/                   # 前端文件夹
│   ├── index.html              # 主页面
│   ├── css/                    # 样式表文件夹
│   │   └── style.css           # 样式表文件
│   ├── js/                     # JavaScript 文件夹
│   │   └── app.js              # 主 JavaScript 文件
│   └── static/                 # 静态资源文件夹（如图片、字体等）
│       ├── images/             # 图片文件夹
│       └── fonts/              # 字体文件夹
│
├── scripts/                    # 脚本文件夹
│   ├── train.sh                # 用于训练模型的 shell 脚本
│   └── evaluate.sh             # 用于评估模型的 shell 脚本
│
├── logs/                       # 日志文件夹
│   └── training.log            # 训练日志文件
│
├── config.json                 # 配置文件，包含超参数设置等
│
└── package.json                # 项目依赖管理文件（如果使用 Node.js）
```