---
title: >-
  [论文解读] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model
description: >-
  [CVPR 2025][预训练][scaling law] 首次在人类动作生成领域系统验证缩放律，提出包含Motion FSQ-VAE（解决codebook collapse）、260小时MotionUnion数据集和文本前缀自回归Transformer的可扩展系统ScaMo，发现归一化测试损失与FLOPs的对数律以及词汇参数/模型参数/数据量与FLOPs的幂律关系，并在$1\times 10^{18}$FLOPs预算下成功预测最优配置。
tags:
  - "CVPR 2025"
  - "预训练"
  - "scaling law"
  - "autoregressive motion generation"
  - "FSQ-VAE"
  - "Transformer"
  - "vocabulary scaling"
---

# ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model

**会议**: CVPR 2025  
**arXiv**: [2412.14559](https://arxiv.org/abs/2412.14559)  
**代码**: [github.com/shunlinlu/ScaMo](https://github.com/shunlinlu/ScaMo)  
**领域**: 动作生成 / 缩放律  
**关键词**: scaling law, autoregressive motion generation, FSQ-VAE, text-prefix transformer, vocabulary scaling

## 一句话总结
首次在人类动作生成领域系统验证缩放律，提出包含Motion FSQ-VAE（解决codebook collapse）、260小时MotionUnion数据集和文本前缀自回归Transformer的可扩展系统ScaMo，发现归一化测试损失与FLOPs的对数律以及词汇参数/模型参数/数据量与FLOPs的幂律关系，并在$1\times 10^{18}$FLOPs预算下成功预测最优配置。

## 研究背景与动机
**领域现状**：缩放律(scaling law)在NLP（GPT系列）和图像生成领域已广泛验证，能精确预测给定计算预算下的最优模型大小和数据需求。但在人类动作生成领域，缩放律几乎完全未被探索。

**现有痛点**：(1) **数据规模不足**——最大的Motion-X仅98k序列，且concurrent work中50%+是静态帧（单帧重复64次），数据质量差；(2) **词汇表无法扩展**——传统VQ-VAE在增大codebook时发生codebook collapse（利用率骤降至23%），性能反而劣化；(3) **模型架构缺乏可扩展性**——直接用预训练LLM扩展词汇表会损害生成性能，且受限于现有LLM的固定尺寸选择。

**核心矛盾**：如何构建一个从tokenizer到生成模型都可系统扩展的动作生成框架，以验证缩放律的存在？

**切入角度**：分别解决三个障碍——FSQ解决codebook collapse、MotionUnion解决数据规模、text-prefix design解决文本编码后自由扩展模型。

## 方法详解

### 整体框架
ScaMo由两个核心组件组成：(1) **Motion FSQ-VAE**将连续动作序列编码为离散token；(2) **文本前缀自回归Transformer**以冻结T5-XL的词级嵌入作为前缀、以因果注意力生成motion token。系统支持从44M到3B的模型规模和$2^8$到$2^{16}$的词汇表大小，在MotionUnion（150k序列, 30M帧, 260小时）上训练。

### 关键设计
1. **Motion FSQ-VAE（有限标量量化）**:
    - 功能：用简单的round操作替代VQ的argmin匹配，从根本上解决codebook collapse问题
    - 核心思路：传统VQ中 $\hat{\mathbf{z}} = \arg\min_{\mathbf{e}_k} \|\mathbf{z} - \mathbf{e}_k\|_2^2$ 的argmin操作导致优化器偏好特定codebook条目而忽略其他。FSQ改为 $\hat{\mathbf{z}} = \text{round}(f(\mathbf{z}))$（$f$为sigmoid），每个通道量化为$L$个整数，codebook大小$|\mathcal{C}| = \prod_{i=1}^d L_i$。训练目标仅需重构损失 $\mathcal{L} = \|\mathbf{m} - \text{Dec}(f(z) + \text{sg}(\text{round}(f(z)) - f(z)))\|_2^2$，无需EMA或codebook reset等tricks
    - 设计动机：VQ的argmin是codebook collapse的根源——空间匹配导致训练倾向于使用小部分条目。FSQ的round操作是均匀的，每个条目被同等概率选中，从而在$2^{16}$规模codebook上仍保持96%利用率（VQ仅23%）

2. **文本前缀自回归Transformer（Text-Prefix AR）**:
    - 功能：用冻结的T5-XL编码器生成词级embeddings作为prefix，motion token部分用因果注意力自回归生成
    - 核心思路：文本token用双向注意力（互相可见），motion token用因果注意力（仅可见前文），motion token可attend所有text token。训练目标为motion token的交叉熵损失 $\mathcal{L} = -\sum_{t=1}^n \log p(\hat{m}_t | m_{<t}, S, V)$。模型规模从44M到3B（8-48层, 512-3200维），架构类LLaMA（RMSNorm + prefix attention + FFN）
    - 设计动机：之前的方法(MotionGPT等)直接扩展LLM词汇表，损害语言压缩能力且受限于预训练LLM固定尺寸。Text-prefix设计解耦了文本编码（冻结T5-XL）和动作生成（可自由扩展），实验FID从0.226降至0.104

3. **MotionUnion数据集**:
    - 功能：构建260小时、150k序列的大规模文本-动作数据集
    - 核心思路：整合Motion-X、CombatMotion、100-Style和内部数据，统一retarget到SMPL骨架，采用HumanML3D的motion representation pipeline处理。内部数据用GPT-4生成文本标注。不含静态帧等质量问题
    - 设计动机：现有数据集规模不足以观察scaling behavior，且concurrent work的数据集50%+是静态帧（质量差）

### 损失函数 / 训练策略
FSQ-VAE仅用重构损失训练（无codebook loss/EMA/reset）。自回归模型用motion token的交叉熵损失训练，text prefix部分不参与loss计算。为公平评估不同词汇表大小，使用归一化损失 $\mathcal{L}_u = -\frac{1}{T}\sum_{t=1}^T \log \frac{p(m_t|m_{<t},S,V)}{p(m_t|S,V)}$。缩放律验证训练44M-3B模型×$2^8$-$2^{16}$词汇表的全矩阵实验。

## 实验关键数据

### 主实验：FSQ vs VQ在不同codebook大小下的对比

| 指标 | VQ ($2^{10}$) | FSQ ($2^{10}$) | VQ ($2^{16}$) | FSQ ($2^{16}$) |
|---|---|---|---|---|
| 重构L1↓ | 0.031 | **0.030** | 0.034(不稳定) | **0.022** |
| MPJPE↓ | 0.072 | **0.070** | 0.156(崩塌) | **0.089** |
| Codebook利用率↑ | 89% | **95%** | 23%(崩塌) | **96%** |
| Entropy↑ | 6.1 | **6.5** | 4.2(崩塌) | **7.8** |

### 缩放律验证

| 关系 | 公式 | $R^2$ |
|---|---|---|
| 词汇参数 $N_v$ vs FLOPs $C$ | $N_v = 10^{-5.29} \cdot C^{0.75}$ | 0.95 |
| 非词汇参数 $N_{nv}$ vs FLOPs $C$ | $N_{nv} = 10^{-0.52} \cdot C^{0.57}$ | 0.93 |
| 数据量 $D$ vs FLOPs $C$ | $D = 10^{-0.05} \cdot C^{0.43}$ | 0.91 |
| $N_v$ vs $N_{nv}$ | $N_v = 10^{-5.604} \cdot N_{nv}^{1.467}$ | **0.95** |
| 归一化损失 vs FLOPs | $\mathcal{L}_u = -1.062 \times \log_{10}(C) + 13.839$ | 0.97 |

### 预测验证（$C = 1 \times 10^{18}$）

| 项目 | 缩放律预测 | 实际值 |
|---|---|---|
| 最优模型大小 | 3B | 3B |
| 最优词汇表大小 | $2^{16}$ | $2^{16}$ |
| 预测归一化损失 | ~-4.3 | **精确对齐** |

### 架构消融（343M模型，MotionUnion训练）

| Text编码器 | Prefix设计 | FID↓ | Matching Score↑ | Top1 R-P↑ |
|---|---|---|---|---|
| CLIP | ✗ | 0.226 | 3.422 | 0.402 |
| **T5-XL** | **✓** | **0.104** | **3.021** | **0.510** |

### 关键发现
- FSQ在大codebook上完全碾压VQ——利用率96% vs 23%，MPJPE几乎减半
- 词汇参数应比模型参数更快扩展：$N_v \propto N_{nv}^{1.467}$（$\gamma > 1$是新发现）
- 模型参数应比数据更快扩展：$N_{nv}/D \propto C^{1.325} > C$（与NLP中Chinchilla的结论一致方向）
- 归一化损失与FLOPs精确遵循对数律，$R^2=0.97$
- T5-XL prefix设计相比CLIP非prefix，FID降低54%

## 亮点与洞察
- 首次在动作生成领域验证缩放律存在，为该领域的大规模训练提供理论指导
- FSQ从根本上解决了codebook collapse——不是"修补"而是"去除"argmin操作
- "$N_v \propto N_{nv}^{1.467}$"的发现是新颖的——词汇表应比模型参数更快增长，与NLP直觉不同
- 缩放律在$10^{18}$FLOPs预算下的预测精确度验证了理论的可靠性
- ScaMo-3B可处理抽象和复杂长句子输入，展示了缩放带来的涌现能力

## 局限与展望
- MotionUnion数据集规模（150k序列）相比NLP/CV仍然很小，可能限制缩放律的外推适用范围
- 仅验证了decoder-only自回归范式，diffusion-based动作生成的缩放律未探索
- FSQ未扩展到group FSQ或residual FSQ（作者留作future work）
- 生成评估主要在HumanML3D benchmark上，缺乏对更多下游应用（动画制作、机器人控制）的验证
- 内部数据不开源，MotionUnion的完整可复现性受限

## 相关工作与启发
- **vs T2M-GPT**：使用相同的VQ-VAE架构但codebook无法扩展，ScaMo用FSQ解决了这一根本限制
- **vs MotionGPT/LLM-based方法**：直接扩展LLM词汇表损害性能且受限于固定模型尺寸，text-prefix设计解耦了两者
- **vs Chinchilla (Hoffmann 2022)**：ScaMo发现的缩放律方向一致（模型比数据增长更快），但额外引入了词汇表维度的幂律
- **启发**：词汇表大小作为独立变量参与缩放律是新视角，可能适用于其他codebook-based生成模型（VQ-VAE图像/视频生成）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次验证动作生成缩放律，FSQ解决codebook collapse优雅有效
- 实验充分度: ⭐⭐⭐⭐⭐ 44M-3B全矩阵实验、多种codebook大小、缩放律拟合与预测验证
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，RQ驱动的实验组织方式好
- 价值: ⭐⭐⭐⭐⭐ 为动作生成社区提供了可操作的理论工具（给出计算预算→查缩放律→确定配置）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](../../ECCV2024/llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)

</div>

<!-- RELATED:END -->
