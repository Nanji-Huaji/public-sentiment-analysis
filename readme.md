# 基于自然语言处理的立场检测-舆情分析系统

## 项目简介

本项目是一个基于自然语言处理技术的舆情分析系统，使用微调的多语言BERT（mBERT）模型实现立场检测功能。系统可以分析文本对特定目标的情感立场（支持、反对或中立），并提供相应的置信度，适用于社交媒体舆情监测、公共事件分析等场景。

## 主要功能

1. **立场检测**：分析文本对特定目标的情感态度（支持、反对或中立）
2. **舆情监测**：爬取社交媒体平台的相关帖子，进行立场分析
3. **数据可视化**：展示舆情分析结果，包括立场分布统计图表
4. **大语言模型增强**：可选择使用GPT、DeepSeek等大语言模型对舆情数据进行深度分析和总结

## 技术架构

- **后端**：基于PyTorch实现的mBERT微调模型
- **前端**：使用Streamlit构建的交互式Web界面
- **数据收集**：社交媒体平台爬虫（支持微博、小红书、贴吧等）
- **模型增强**：集成OpenAI API，支持GPT-3、GPT-3.5、GPT-4o等大语言模型

## 数据集

本项目使用了多个中文立场检测数据集，包括：

- **微博立场检测数据集(Weibo-SD)**：包含五个热点事件的微博评论，如"唐山打人"、"胡鑫宇失踪"等
- **C-Stance数据集**：中文立场检测数据集
- **VAST数据集**：用于模型泛化能力训练

## 目录结构

```
public-sentiment-analysis/
│
├── README.md                   # 项目说明文件
│
├── data/                       # 数据文件夹
|   ├── raw                     # 原始数据集
|   ├── processed               # 处理过的数据集
|   ├── csv_data                # CSV格式数据
|   └── analysis                # 分析结果数据
│
├── models/                     # 模型文件夹
│   └── checkpoint-*/           # 微调后的各个模型检查点
│       └──produce              # 用于实际调用检查点的路径
│
├── src/                        # 源代码文件夹（后端）
│   ├── dataset.py              # 数据集类实现 
│   ├── model.py                # 模型架构和预训练模型加载
│   ├── train.py                # 训练脚本（包含训练循环、损失计算、优化等）
│   ├── eval.py                 # 评估脚本
│   ├── inference.py            # 推理脚本
│   ├── logger.py               # 日志工具
│   ├── prompt.py               # 提示词模板
│   ├── incremental.py          # 增量学习相关
│   └── MediaCrawler.py         # 社交媒体爬虫
│
├── frontend/                   # 前端文件夹
│   ├── myapp.py                # Streamlit应用主文件
│   ├── utils.py                # 前端工具函数
│   └── data/                   # 前端所需数据
│
│   
├── tests/                      # 测试文件夹
│   └── test_llm_inference.py   # LLM推理测试
│
├── logs/                       # 日志文件夹
│
├── requirements.txt            # 项目依赖
├── Dockerfile                  # Docker配置
├── dataratio.py                # 数据比例分析工具
├── process.py                  # 数据处理脚本
└── plot_f1_by_epoch.py         # 绘制F1分数随训练周期变化图表
```

## 使用方法

### 环境配置

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 数据准备

```bash
python process.py
```

### 训练模型

```bash
python src/incremental.py
```

### 启动前端界面

```bash
cd frontend
streamlit run myapp.py
```

## 结果展示

系统能够实现：

1. 对输入文本进行立场检测
2. 对社交媒体数据进行爬取和分析
3. 可视化展示立场分布比例
4. 使用大语言模型生成深度舆情分析报告
