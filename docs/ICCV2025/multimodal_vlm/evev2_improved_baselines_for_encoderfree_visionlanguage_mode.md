# EVEv2: Improved Baselines for Encoder-Free Vision-Language Models

**会议**: ICCV 2025 (Highlight)  
**arXiv**: [2502.06788](https://arxiv.org/abs/2502.06788)  
**代码**: [https://github.com/baaivision/EVE](https://github.com/baaivision/EVE)  
**领域**: 多模态VLM / 无编码器架构  
**关键词**: encoder-free VLM, Divide-and-Conquer, 模态稀疏, decoder-only, 视觉感知从零学习  

## 一句话总结
系统性地探索无视觉编码器VLM的最优架构和训练策略，提出Divide-and-Conquer架构将transformer完全分解为模态专用组件（attention/FFN/LayerNorm各模态独立），在仅100M公开数据下超越所有encoder-free同类并接近encoder-based VLM性能。

## 背景与动机
主流VLM（如LLaVA/InternVL）依赖预训练视觉编码器（如CLIP-ViT），但编码器带来分辨率/宽高比的惯性偏置、复杂的多组件协调需求、以及独立scaling的困难。Encoder-free VLM（如Fuyu/EVEv1）直接让一个统一的decoder-only模型从零学习视觉感知，结构更简洁。但挑战在于：(1) 从零学习视觉感知需要大量数据和计算；(2) 视觉和语言在同一个模型中可能产生表示干扰——简单的权重共享或MoE解耦不够充分。

## 核心问题
如何让encoder-free VLM高效地从零学习视觉感知,同时最小化视觉和语言模态之间的表示干扰？

## 方法详解

### 整体框架
EVEv2.0基于Qwen2.5-7B LLM，用简单的双层卷积做patch embedding（stride 16+2），然后将visual tokens和text tokens一起送入一个完全模态稀疏的decoder-only transformer。四阶段训练：(1)预对齐patch embedding → (2)冻结LLM训练视觉层 → (3)全模型QA微调 → (4)指令微调。

### 关键设计
1. **Divide-and-Conquer（完全模态解耦）架构**：不同于EVEv1的密集模型、EVEv1.2的重参数化、EVEv1.5的MoE解耦FFN，EVEv2.0在每一层的**所有组件**上引入模态分组：Query/Key/Value矩阵、输出投影、LayerNorm、FFN都有独立的视觉版和文本版参数。总参数量2×7B但每个token的活跃FLOPs等于7B dense。关键发现是LayerNorm的模态干扰最严重（LLM→VLM的权重变化最大），必须完全解耦。

2. **DenseFusion++标注引擎**：用LLaVA-1.6(7B)融合多个视觉专家（Tag/Detection/OCR等），学习GPT-4V的融合策略，生成超详细的图像描述。比前代LLaVA-1.5(13B)+Emu2(17B)组合更好，且单节点8卡A100每天可标注70万张。这使得EVEv2.0只需100M公开数据就能达到强效果。

3. **渐进式四阶段训练**：Stage1仅训练patch embedding（对齐初始化）→ Stage2.1冻结LLM训练视觉层学习视觉感知（低分辨率→高分辨率渐进）→ Stage2.2全参数QA微调增强多模态对齐→ Stage3指令微调。视觉层从LLM权重初始化，保证训练开始时文本能力不损失。

### 损失函数 / 训练策略
- 标准交叉熵自回归loss
- 数据规模：44M Datacomp + 15M LAION + 11M SA-1B + 7M OpenImages（预训练）；15M multi-task（QA）；7.3M指令微调
- 16节点128卡A100训练
- 分辨率从800×800渐进到1600×1600，最大2500 patch tokens

## 实验关键数据
| 模型 | 类型 | 参数 | MMMU | MMBench | TextVQA | ChartQA | AI2D | OCRBench |
|------|------|------|------|---------|---------|---------|------|----------|
| LLaVA-1.5 | encoder | 7B | 35.3 | 64.3 | 46.1 | 18.2 | 54.8 | 318 |
| LLaVA-1.6 | encoder | 7B | 35.1 | 67.4 | 64.9 | 54.8 | 66.6 | 532 |
| Cambrian | encoder | 7B | 42.7 | 75.9 | 71.7 | 73.3 | 73.0 | 614 |
| Fuyu | enc-free | 8B | 27.9 | 10.7 | - | - | 64.5 | 366 |
| EVEv1 | enc-free | 7B | 32.6 | 52.3 | 56.8 | 59.1 | 61.0 | 398 |
| Mono-InternVL | enc-free | 1.8B | 33.7 | 65.5 | 72.6 | 73.7 | 68.6 | 767 |
| **EVEv2.0** | **enc-free** | **7B** | **39.3** | **66.3** | **71.1** | **73.9** | **74.8** | **702** |

- 超越所有encoder-free方法（除Mono-InternVL在部分指标上更强，但其用13x更多数据）
- 接近LLaVA-1.6和Cambrian等encoder-based方法
- ScienceQA达96.2%，超越大部分encoder-based方法
- 数据效率：100M数据 vs Mono-InternVL的1.3B数据

### 消融实验要点
- DaC > MoE > ReP > Dense：完全解耦在24M数据下比MoE高1.4%，且差距随数据增多而扩大
- DenseFusion++标注引擎 > LLaVA-1.5+Emu2标注 > Raw web captions
- 多源数据混合（Datacomp+LAION+SA1B+OpenImages）远优于单源
- LayerNorm是最需要解耦的模块（仅解耦LN即可获得明显提升）
- 推理速度：EVEv2.0 TTFT仅比EVEv1.0多13%，TPS相同（35 tok/s）

## 亮点
- **ICCV Highlight，系统性研究的典范**：不是追求SOTA而是系统性地回答"encoder-free VLM的最优路径是什么"
- **Divide-and-Conquer架构**：完全模态解耦是encoder-free VLM的关键突破——简单但有效，通过量化分析（LLM vs VLM权重变化）给出充分动机
- **DenseFusion++标注引擎**的高效性：7B模型标注质量超越13B+17B组合，可规模化
- **数据效率出色**：100M公开数据达到需要1.3B数据的Mono-InternVL的可比水平
- **透明和可复现**：所有数据公开、代码开源、训练细节详尽

## 局限性 / 可改进方向
- 因计算资源限制，未充分探索更大模型（>7B）和更多数据的scaling
- 知识密集型任务（MMMU）仍落后于encoder-based方法
- 文档理解任务（DocVQA等）有一定差距
- 2×7B参数的存储需求高于标准7B模型
- 尚未扩展到音频/视频模态

## 与相关工作的对比
- **vs. EVEv1**：EVEv1用单个dense模型+视觉监督；EVEv2完全解耦+DenseFusion++，无需视觉监督，性能大幅提升
- **vs. Mono-InternVL**：Mono-InternVL仅解耦FFN（MoE），EVEv2完全解耦所有组件；但Mono-InternVL用13x更多数据
- **vs. Scaling Language-Free Visual Repr**：Web-SSL证明SSL可以匹配CLIP；EVEv2证明全零学习的视觉encoder也可以匹配预训练encoder——两者方向互补
- **vs. FALCON**：FALCON在encoder内部用register压缩高分辨率token；EVEv2从根本上去掉encoder

## 启发与关联
- **idea潜力**：完全模态解耦的思路可以扩展到三模态（视觉/文本/音频）的native multimodal model
- Divide-and-Conquer与Dynamic-DINO的MoE方法可以结合——在decoder-only架构中使用模态感知的细粒度专家
- DenseFusion++标注引擎的思路对数据工程领域有重要参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 完全模态解耦是encoder-free VLM的范式性突破，LayerNorm必须解耦的发现很深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 13个benchmark、4种架构变体（v1.0/1.2/1.5/2.0）、数据源/标注引擎/训练策略逐项消融
- 写作质量: ⭐⭐⭐⭐⭐ 系统性研究的写作标杆，Figure 2的权重变化量化分析和Figure 5的scaling对比极具说服力
- 价值: ⭐⭐⭐⭐⭐ Highlight当之无愧，为encoder-free VLM方向确立了清晰的技术路线图
