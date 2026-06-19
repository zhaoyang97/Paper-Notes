---
title: >-
  [论文解读] Byte Latent Transformer: Patches Scale Better Than Tokens
description: >-
  [ACL 2025][预训练][byte-level LLM] 提出 Byte Latent Transformer (BLT)，一种无分词器的字节级 LLM 架构，通过基于熵的动态分组将字节聚合为可变长度 patch，首次在 8B 规模上匹配 token-based 模型性能，同时解锁了通过同时增大 patch 和模型尺寸来提升推理效率的新 scaling 维度。
tags:
  - "ACL 2025"
  - "预训练"
  - "byte-level LLM"
  - "tokenizer-free"
  - "dynamic patching"
  - "图像分割"
  - "scaling laws"
---

# Byte Latent Transformer: Patches Scale Better Than Tokens

**会议**: ACL 2025  
**arXiv**: [2412.09871](https://arxiv.org/abs/2412.09871)  
**代码**: [https://github.com/facebookresearch/blt](https://github.com/facebookresearch/blt)  
**领域**: 其他  
**关键词**: byte-level LLM, tokenizer-free, dynamic patching, entropy-based segmentation, scaling laws

## 一句话总结

提出 Byte Latent Transformer (BLT)，一种无分词器的字节级 LLM 架构，通过基于熵的动态分组将字节聚合为可变长度 patch，首次在 8B 规模上匹配 token-based 模型性能，同时解锁了通过同时增大 patch 和模型尺寸来提升推理效率的新 scaling 维度。

## 研究背景与动机

**领域现状**：几乎所有现代 LLM 都使用 BPE 等分词器将字节序列转化为固定词表中的 token。分词是整个训练流程中唯一的非端到端启发式预处理步骤，已成为"默认选择"。

**现有痛点**：固定词表的分词方式带来多重缺陷——(1) 同一个词在不同上下文中可能被分成不同 token，造成不一致性；(2) 对输入噪声（拼写错误、大小写变化）极度敏感；(3) 缺乏正字法知识（不知道词由哪些字符组成）；(4) 多语言不公平（低资源语言的 token 更长、计算更贵）；(5) 领域/模态敏感性。之前的字节级模型（如 MegaByte）因序列过长导致计算量暴增，无法在大规模上竞争。

**核心矛盾**：字节级建模消除了分词的所有问题，但直接在字节上运行 Transformer 的计算成本由大型 FFN 层主导（非注意力），序列长度增加直接导致成本线性增长。关键洞察：大部分字节的预测是简单的（如一个单词后面的字节），不需要大型 Transformer 的全部算力。

**本文目标** 如何在字节级别高效训练 LLM——保持字节级建模优势的同时让效率和性能匹配 token 模型。

**切入角度**：根据下一字节预测的熵动态分配计算——信息密度高的位置分配更多计算，低的分配更少。比 BPE 按压缩启发式分配更合理。

**核心 idea**：用小型字节级语言模型的熵估计动态分割字节为可变长度 patch，大型 Transformer 只在 patch 级别运行，轻量级模型处理 patch 内部字节。

## 方法详解

### 整体框架

BLT 由三部分组成：(1) **Local Encoder**——轻量级 Transformer（层数 $l_\mathcal{E} \ll l_\mathcal{G}$），将输入字节编码为 patch 表示，通过交叉注意力将字节池化为 patch；(2) **Latent Global Transformer**——大型自回归 Transformer，在 patch 表示上操作，消耗绝大部分 FLOP；(3) **Local Decoder**——轻量级 Transformer，将 patch 表示解码回字节序列。Pipeline：输入字节 → entropy patching 动态分组 → Local Encoder 编码为 patch → Global Transformer 处理 → Local Decoder 解码为字节。

### 关键设计

1. **Entropy Patching（基于熵的动态分组）**:

    - 功能：根据数据复杂度动态分配计算资源
    - 核心思路：训练一个 100M 参数的字节级语言模型，计算每个字节位置的下一字节熵 $H(x_i) = \sum_{v \in \mathcal{V}} p_e(x_i=v|\mathbf{x}_{<i}) \log p_e(x_i=v|\mathbf{x}_{<i})$。当熵超过全局阈值 $\theta_g$ 时开始新 patch。例如 "George R.R. Martin" 中 "G" 的熵高（不确定下一字符）因此成为新 patch 起点，而 "eorge" 的熵低归入同一 patch。另有近似单调约束：$H(x_t) - H(x_{t-1}) > \theta_r$ 时划分。通过调整阈值可以任意控制平均 patch size
    - 设计动机：BPE 根据频率统计压缩，与预测难度不一定相关。Entropy patching 让模型在预测困难处（如新句子开头）投入完整 Transformer 计算，在容易处（如单词内部）几乎零成本通过

2. **Hash N-gram Embeddings**:

    - 功能：为字节位置注入局部上下文信息
    - 核心思路：对每个字节位置 $i$ 构建 3-gram 到 8-gram 的字节片段，通过多项式 rolling hash 映射到 500K 大小的嵌入表，加到字节嵌入上：$e_i = x_i + \sum_{n=3}^{8} E_n^{hash}(\text{Hash}(g_{i,n}))$
    - 设计动机：单个字节（0-255）信息量极少，n-gram 嵌入让每个位置"看到"前几个字节的模式（常见前缀、后缀），弥补字节级模型缺乏子词信息的短板

3. **Encoder-Decoder 交叉注意力**:

    - 功能：在字节和 patch 表示之间高效传递信息
    - 核心思路：Encoder 中 patch 作 query、字节作 key/value（Perceiver 风格），将字节信息池化到 patch；Decoder 中反转，字节作 query、patch 作 key/value。query 由 max-pooling 初始化。每个 patch query 只 attend 到对应 patch 内的字节。patch 维度 $h_\mathcal{G}$ 由多个 $h_\mathcal{E}$ 的 head 拼接
    - 设计动机：交叉注意力比全局自注意力高效，且天然适合字节→patch 的尺度变换。掩码策略确保因果性

### 损失函数 / 训练策略

标准字节级自回归交叉熵损失。Local Decoder 输出 256 维 logits（字节表大小）。AdamW 优化器（$\beta_1=0.9, \beta_2=0.95$），学习率 4e-4，cosine 衰减到 0，2000 步 warmup，weight decay 0.1。在 Llama 2 数据（2T tokens）做 scaling law，在 BLT-1T 高质量数据做完整训练。batch size 保持 16M bytes/batch，通过 pack patches 避免 padding。

## 实验关键数据

### 主实验（8B 模型下游任务评估）

| 模型 | Arc-E | Arc-C | HellaSwag | PIQA | MMLU | MBPP | HumanEval | 平均 |
|------|-------|-------|-----------|------|------|------|-----------|------|
| Llama 3 (1T tokens) | 77.6 | 53.3 | 79.1 | 80.7 | 58.1 | 40.2 | 31.1 | 60.0 |
| BLT-Space (6T bytes) | 75.4 | 49.8 | 79.6 | 81.1 | 54.8 | 37.6 | 27.4 | 58.0 |
| BLT-Entropy (4.5T bytes) | **79.6** | 52.1 | **80.6** | 80.6 | 57.4 | **41.8** | **35.4** | **61.1** |

（FLOP-matched。BLT-Entropy 在 7/7 项中均约等于或胜过 Llama 3）

### 鲁棒性和字符级任务

| 任务 | Llama 3 (1T) | Llama 3.1 (16T) | BLT (1T) |
|------|-------------|----------------|----------|
| HellaSwag Noise Avg | 56.9 | 64.3 | **64.3** |
| CUTE 字符理解 | 27.5 | 20.0 | **54.1** |
| Spelling | 1.1 | - | **99.9** |
| Spelling Inverse | 30.1 | 3.6 | **99.9** |
| Contains Char | 0.0 | 0.0 | **55.9** |
| Substitute Char | 0.4 | 1.2 | **48.7** |

### 关键发现

- BLT-Entropy 在同等训练 FLOP 下平均性能超过 Llama 3（61.1 vs 60.0），代码任务提升显著（HumanEval 35.4 vs 31.1）
- **字符级理解碾压 token 模型**：拼写准确率 99.9% vs 1.1%，字符包含判断 55.9% vs 0.0%。token 模型从根本上无法访问 token 内部字符
- 噪声鲁棒性提升 8 个百分点，甚至匹配 16 倍数据量的 Llama 3.1——字节感知不是"更多数据能弥补的"
- **固定推理 FLOP 的 scaling**：BLT 可同时增大 patch size 和模型参数而维持推理成本不变。Patch size 8 模型在约 2.5x compute-optimal 数据量后超越 BPE 模型
- 低资源语言翻译提升显著：亚美尼亚语 1.7→6.3，格鲁吉亚语 1.7→7.4，孟加拉语 4.7→12.7（BLEU），对非拉丁字母语言优势尤其明显
- BLT-Space（patch size 6）性能略低于 Llama 3 但推理 FLOP 省约 30%，提供了性能-效率的灵活权衡

## 亮点与洞察

- **Entropy patching 优雅解决计算分配**：用小模型的"不确定性"引导大模型的计算分配，让简单字节零成本通过、困难字节获得完整处理。"花钱花在刀刃上"的完美体现
- **新 scaling 维度**：token 模型增大词汇表受限于 embedding 层增长。BLT 的 patch size 可任意增大而不影响参数量，解锁了"更大模型 + 更大 patch = 更好性能 + 不变推理成本"的独特路径。随着模型规模增长，Local Encoder/Decoder 的 FLOP 占比逐渐缩小，大 patch size 的优势更凸显
- **字节级建模的"免费午餐"**：拼写、字符操作、噪声鲁棒性、多语言——token 模型需大量数据才可能弥补的能力，在 BLT 中天然内建

## 局限与展望

- Entropy model 增加预处理开销（虽可用小模型或查表优化）
- 推理时需逐步判断 patch 边界（incremental patching），实现工程复杂度高于 BPE
- 目前只在 8B 验证，70B+ 规模是否延续趋势需验证
- BLT-Space 在同 FLOP 下性能低于 Llama 3，说明太大的 patch size 会丢信息——最优 patch size 需随规模调整
- 缺乏指令微调和 RLHF 场景的探索

## 相关工作与启发

- **vs MegaByte (Yu et al., 2023)**: 固定 stride 分组，不考虑复杂度，缺乏 n-gram 嵌入和交叉注意力。BLT 的每项创新直接解决了 MegaByte 的一个短板
- **vs SpaceByte (Slagle, 2024)**: 按空格分组比固定 stride 好，但无法处理非空格语言（中/日文），也无法调节 patch size。BLT 的 entropy patching 是通用可调的替代
- **vs Llama 3**: BLT 匹配性能且提供额外的推理效率杠杆和字符级能力。长期 scaling 趋势更好——这是最有说服力的证据

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次大规模证明字节级模型可匹配 token 模型，entropy patching 和 patch scaling 是原创贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 400M-8B 完整 scaling law、下游任务、鲁棒性、多语言、消融实验非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，FLOP 计算透明可复现，图表设计优秀
- 价值: ⭐⭐⭐⭐⭐ 可能改变 LLM 预处理范式，从根本上解决 tokenization 的多个长期痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ICML 2026\] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](../../ICML2026/llm_pretraining/the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[NeurIPS 2025\] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](../../NeurIPS2025/llm_pretraining/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)

</div>

<!-- RELATED:END -->
