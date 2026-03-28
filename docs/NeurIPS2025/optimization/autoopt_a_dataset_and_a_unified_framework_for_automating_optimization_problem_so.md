# AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving

**会议**: NeurIPS 2025  
**arXiv**: [2510.21436](https://arxiv.org/abs/2510.21436)  
**代码**: https://github.com/Shobhit1201/AutoOpt  
**领域**: 优化自动化  
**关键词**: 图像OCR, LaTeX转PYOMO, 双层优化求解, 数学规划, 手写识别

## 一句话总结
AutoOpt 构建了首个优化问题图像到代码的端到端框架——11554 张优化公式图像（手写+印刷）的 AutoOpt-11k 数据集 + M1 混合编码器（ResNet+Swin→mBART）图像转 LaTeX（BLEU 96.70）+ M2 DeepSeek-Coder LaTeX 转 PYOMO + M3 双层分解求解器，框架级成功率 94.20%。

## 研究背景与动机

1. **领域现状**：数学优化公式常以图像形式存在——白板、论文扫描、手写笔记。OCR 和 LLM 在通用文档上表现好但数学公式识别仍难。
2. **现有痛点**：(a) 无专门的优化问题图像数据集——现有 OCR 数据集不覆盖目标函数/约束的特定结构；(b) LaTeX 识别到可执行代码的管线断裂——需要理解优化语义转为 PYOMO；(c) 复杂非凸/多层优化缺乏统一的自动求解方法。
3. **核心矛盾**：将优化问题从图像到解需要三个不同能力——视觉识别、语义理解、求解算法——但没有系统同时解决。
4. **本文要解决什么？** 构建图像→LaTeX→PYOMO→求解的完整自动化管线。
5. **切入角度**：分三个模块各个击破——M1 用 CNN+Transformer 混合编码器处理图像→LaTeX；M2 用微调 DeepSeek-Coder 处理 LaTeX→PYOMO；M3 用双层分解处理复杂优化。
6. **核心 idea 一句话**：AutoOpt-11k 数据集 + ResNet-Swin-mBART 图像→LaTeX + DeepSeek-Coder LaTeX→PYOMO + 双层优化分解 = 图像到优化解的全自动管线。

## 方法详解

### 整体框架
优化公式图像 → **M1**: ResNet-101 + Swin Transformer 混合编码 → mBART 解码生成 LaTeX → **M2**: 微调 DeepSeek-Coder-1.3B 将 LaTeX 转为 PYOMO 代码 → **M3**: BOBD 双层分解求解

### 关键设计

1. **M1 混合编码器（ResNet+Swin→mBART）**:
   - 做什么：将优化公式图像转为 LaTeX
   - 核心思路：ResNet-101 提取局部特征 $\mathbf{f}_{ResNet} = \alpha \cdot \text{LN}(\text{Proj}(F))$，前置拼接到 Swin Transformer 的 patch embedding，mBART 解码生成 LaTeX。从 NOUGAT 权重迁移学习
   - 设计动机：CNN 捕获局部笔画，Swin 捕获全局结构——互补。消融显示 CNN+Transformer (DL3) BLEU 96.70 vs Transformer only (DL2) 95.51 vs CNN only (DL1) 16.10

2. **M2 LaTeX→PYOMO（DeepSeek-Coder）**:
   - 做什么：将 LaTeX 数学公式翻译为可执行的 PYOMO 优化代码
   - 核心思路：在 1018 个数学程序的 80% 上微调 DeepSeek-Coder-1.3B，BLEU 88.25
   - 设计动机：LaTeX 是数学表示，PYOMO 是编程接口——需要理解变量声明、目标函数、约束映射

3. **M3 BOBD 双层分解**:
   - 做什么：处理复杂非凸/多层优化问题
   - 核心思路：ML 分类变量为上/下层 → 上层用遗传算法优化不可微变量 → 下层用 CVX 求解凸子问题
   - 设计动机：统一处理线性/非线性/凸/非凸/多层目标/随机优化

### 损失函数 / 训练策略
- M1: NOUGAT 迁移学习，图像 768×1024 + 对比增强
- M2: DeepSeek-Coder 微调
- M3: 遗传算法 + CVX 求解器

## 实验关键数据

### 主实验

| 模块 | 指标 | AutoOpt | Nougat | GPT-4o | Gemini 2.0 |
|------|------|---------|--------|--------|-----------|
| M1 | BLEU | **96.70** | 95.51 | — | — |
| M1 | CER | **0.0286** | 0.0440 | 0.1017 | 0.1338 |
| M2 | BLEU | **88.25** | — | — | — |
| 框架 | 成功率 | **94.20%** | — | — | — |

### 消融实验

| M1 架构 | BLEU | CER |
|---------|------|-----|
| DL1 (CNN only) | 16.10 | 0.8812 |
| DL2 (Transformer only) | 95.51 | 0.0440 |
| **DL3 (CNN+Transformer)** | **96.70** | **0.0286** |

### 关键发现
- 混合编码器比纯 Transformer 好 1.2 BLEU / 35% CER 改善——CNN 特征对数学符号识别关键
- 手写 CER 0.0412 vs 印刷 0.0176——手写仍是主要挑战
- 框架整体可靠性下界估计 89.12%——M1 和 M2 的误差乘积
- 500 个 OOD 测试达 94.20% 成功——泛化性强

## 亮点与洞察
- **11554 张数据集**（含 5070 手写）是首个优化问题 OCR 基准
- **端到端管线**从图像→解的全自动化在优化社区是首次
- **CNN+Transformer 混合**在数学 OCR 上的优势被量化

## 局限性 / 可改进方向
- 仅处理单页公式——多页跨页未解决
- 大规模优化问题（十亿变量）未测试
- 标注者一致性有差异（BLEU 0.82-0.86）
- 模糊/不完整的优化问题定义未评估

## 相关工作与启发
- **vs NOUGAT**: 通用文档 OCR，AutoOpt 专注数学优化识别
- **vs GPT-4o/Gemini**: 通用 VLM 在数学 OCR 上 CER 3-5× 更差

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个图像→优化解的完整管线
- 实验充分度: ⭐⭐⭐⭐ 11K 数据集 + 消融 + OOD 测试
- 写作质量: ⭐⭐⭐⭐ 管线描述清晰
- 价值: ⭐⭐⭐⭐ 优化问题自动化的重要基础工作

### 方法补充说明
- **数据集构成细节**：11554 张图像中 5070 手写 + 6484 印刷；10838 单目标 + 159 多目标 + 399 多层 + 158 随机优化；2130 线性 + 9122 非线性；2580 凸 + 3574 非凸——覆盖广泛
- **M1 图像预处理**：768×1024 中心填充 + 对比度增强 + 锐化滤波——针对手写公式中常见的低对比度和模糊问题
- **NOUGAT 迁移学习的效果**：从通用文档 OCR 预训练权重出发微调到数学公式域，CER 从 0.044 降到 0.029——迁移学习关键
- **框架可靠性估计方法**：$(1-	ext{CER}_{M1}) 	imes (1-	ext{CER}_{M2}) = (1-0.0286) 	imes (1-0.0825) = 89.12\%$ 作为下界
- **与 Wolfram Alpha/Mathematica 的区别**：这些工具需要文本输入，AutoOpt 直接从图像端到端求解

- **评分详细说明**：AutoOpt开创了优化问题OCR到求解的完整工作流，数据和代码均已公开，但仅处理单页公式和中等规模问题
