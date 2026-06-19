---
title: >-
  [论文解读] Vision-centric Token Compression in Large Language Model
description: >-
  [NeurIPS 2025 Spotlight][模型压缩][token compression] Vist 提出了一种以视觉为核心的慢-快双路径 token 压缩框架，将远端长文本渲染为图像后用轻量视觉编码器压缩，配合概率引导的视觉增强（PVE）训练目标，在 11 个 ICL 基准上以 2.3× 更少的 token 实现同等精度，FLOPs 降低 16%、显存减少 50%。
tags:
  - "NeurIPS 2025 Spotlight"
  - "模型压缩"
  - "token compression"
  - "vision encoder"
  - "long context"
  - "in-context learning"
  - "frequency-based masking"
---

# Vision-centric Token Compression in Large Language Model

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.00791](https://arxiv.org/abs/2502.00791)  
**代码**: [https://github.com/CSU-JPG/VIST](https://github.com/CSU-JPG/VIST)  
**领域**: 模型压缩 / Token压缩  
**关键词**: token compression, vision encoder, long context, in-context learning, frequency-based masking

## 一句话总结

Vist 提出了一种以视觉为核心的慢-快双路径 token 压缩框架，将远端长文本渲染为图像后用轻量视觉编码器压缩，配合概率引导的视觉增强（PVE）训练目标，在 11 个 ICL 基准上以 2.3× 更少的 token 实现同等精度，FLOPs 降低 16%、显存减少 50%。

## 研究背景与动机

大语言模型（LLM）正面临上下文窗口不断增长与参数规模持续膨胀的双重压力。实际场景中长文档理解、多轮问答等任务对长上下文建模提出了刚需，而直接处理超长 token 序列会导致计算和显存成本飙升。现有的 token 压缩方法大多依赖 LLM 自身来计算 token 级别的信息熵以评估重要性（如 LLMLingua 系列），但这导致压缩过程本身就需要重量级的 LLM 参与，成本高昂。

心理语言学研究表明，人类熟练阅读者在快速浏览文本时会跳过约三分之一的高频功能词（如 "the"、"of"），将注意力集中在稀有的内容词上。这种"选择性阅读"策略天然形成了一种快-慢回路：快速视觉通道略过低显著性的远端上下文以维持全局感知，慢速认知通道深入处理近端关键句子。

本文的核心 idea 是：**将远端低相关文本渲染为图像，让冻结的轻量视觉编码器（如 CLIP ViT）充当"快速眼睛"进行粗略扫描，而 LLM 作为"大脑"聚焦近端关键信息进行深层推理。** 这一方案利用了预训练视觉编码器天然具备的 OCR 能力，绕过了传统 text tokenizer 的词表瓶颈和字符级噪声敏感问题。

## 方法详解

### 整体框架

Vist（Vision-centric Token Compression）采用慢-快双路径设计：

1. **快速视觉路径**：将前 $T_e$ 个 text token 均匀渲染为 $M$ 张 RGB 图像（每张 224×224），送入冻结的 CLIP ViT-L/14 视觉编码器提取特征，再通过可训练的 Perceiver Resampler 压缩为固定数量的视觉 token。
2. **慢速认知路径**：剩余的 $T_d$ 个原始 text token 直接送入 LLM 进行精细推理。
3. 压缩后的视觉 token 通过交叉注意力（cross-attention）注入 LLM，与原始 text token 一起进行 next-token prediction。

### 关键设计

1. **文本-图像渲染**：将文本以 10px 字体大小和 Google Noto Sans 字体渲染为 RGB 图像（H=14, W=3584, C=3），等效 224×224 分辨率。空白区域用 mask 排除在注意力和损失计算之外。1024 个 text token 需渲染为 7 张图像。

2. **Perceiver Resampler 压缩**：冻结 ViT-L/14 提取图像特征 $F \in \mathbb{R}^{M \times L \times D}$，通过可学习的 Perceiver Resampler 将每张图像压缩为 N+1 个视觉 token（含 CLS token），默认 N=64。训练时 4096 个text token 渲染为 28 张图像，压缩为 64×28=1792 个视觉 token，压缩比 Δ=2.3。

3. **概率引导的视觉增强（PVE）**：核心训练目标，包含两个关键组件：

    - **文本锚定语义一致性**：对比学习损失，拉近 Resampler 输出的视觉特征 $\hat{F}'$ 与 LLM tokenizer 产生的 text token embedding $\hat{F}^t$ 之间的距离。
    - **基于频率的遮蔽（FM）**：借鉴 Shannon 信息论（$I(y) = -\log_2 P(y)$），利用语料级 token 频率作为语义重要性的代理。高频 token（如 "the"、"with"）承载的信息量少，被优先遮蔽；低频 token（领域特定或语境关键词）被保留。重要性分数 $s_w = \log \frac{|S|}{1+\text{count}(w)}$，遮蔽率 50%，低重要性 token 被遮蔽的概率更高。

### 损失函数 / 训练策略

- 训练目标：next-token prediction loss + PVE 对比损失
- PVE 对比损失公式：$\mathcal{L}_{PVE}^{ij} = -\log \frac{\exp(\langle \hat{F}'_i, \hat{F}^t_j \rangle / \tau)}{\sum_{k=1}^B \exp(\langle \hat{F}'_i, \hat{F}^t_k \rangle / \tau)}$
- 使用 float16 精度 + DeepSpeed Zero-2 + CPU offloading 训练
- 基础 LLM 为 TinyLlama，预训练数据为 RedPajama 11B token（含 ArXiv、Book、C4 等 7 个领域）
- Perceiver Resampler 与 LLM 的 cross-attention 端到端联合训练

## 实验关键数据

### 主实验：长上下文语言建模 (PPL)

| 方法 | $T_e$ | $T_d$ | ArXiv | Book | PG19 | TFLOPs | MEM(GB) |
|------|--------|--------|-------|------|------|--------|---------|
| TinyLlama | - | 4096 | >10³ | >10³ | >10³ | 8.47 | 5.46 |
| CEPE* | 6144 | 2048 | 3.005 | 14.919 | 11.112 | 13.27 | 7.74 |
| Vist | 6144 | 2048 | **2.989** | **14.894** | 12.737 | **11.65** | **4.94** |
| CEPE* | 14336 | 2048 | 3.003 | 14.921 | 10.909 | 23.30 | 13.59 |
| Vist | 14336 | 2048 | **2.965** | **14.815** | 11.933 | **19.52** | **6.75** |

### Open-domain QA（Exact Match）

| 方法 | $k_e$ | $k_d$ | TriviaQA | NQ | PopQA |
|------|--------|--------|----------|------|-------|
| TinyLlama | - | 10 | 21.45 | 8.45 | 10.79 |
| CEPE* | 20 | 10 | 16.56 | 6.75 | 5.78 |
| Vist | 20 | 10 | **25.67**(+9.11) | **8.81**(+2.06) | **11.84**(+6.06) |

### 消融实验

| 配置 | NLUS | NLUI | TriviaQA | NQ | PopQA | 说明 |
|------|------|------|----------|------|-------|------|
| 无遮蔽 | 9.9 | 26.4 | 17.14 | 6.51 | 5.72 | 基线 |
| 随机遮蔽 | 8.3 | 30.2 | 24.88 | 8.35 | 10.19 | 随机遮蔽有帮助但不够 |
| 频率遮蔽(FM) | **15.6** | **40.6** | **25.20** | **8.71** | **11.44** | FM 是关键 |

### 关键发现

- Vist 在 14K token 输入时，比 CEPE* 节省 3.78 TFLOPs 和 6.84GB 显存，吞吐量提升 2.3×
- 在 Open-domain QA 上，Vist 比 CEPE* 平均高出 5.7%（EM），因为 PVE 引导 Resampler 聚焦关键语义，而 CEPE* 在更多 passage 加入时反而引入噪声
- 频率遮蔽策略（50% 遮蔽率）保留了大部分高信息增益（IG）的 token，证明 token 频率是语义重要性的有效代理
- 每张图像 64 个视觉 token 是最优配置，过多 token（如 128）反而引入噪声
- 扩展到 Mistral 7B 同样有效，PPL 优于对应的 CEPE

## 亮点与洞察

- **范式创新**：首次从视觉角度解决 LLM 长文本压缩问题，将文本渲染为图像后用轻量视觉编码器处理，绕过了传统 text tokenizer 的词表瓶颈
- **生物学启发**：慢-快双路径设计灵感来自心理语言学中人类的选择性阅读策略，巧妙地将学术观察转化为工程方案
- **简洁有效的 PVE**：用 token 频率替代昂贵的 LLM 信息熵计算来评估 token 重要性，大幅降低了压缩的计算开销
- 视觉编码器作为"视觉文本 tokenizer"的四大优势：简化 tokenization、缓解词表瓶颈、抗字符噪声、多语言高效

## 局限与展望

- 目前仅在 TinyLlama 和 Mistral 7B 上验证，缺乏在更大规模 LLM（如 70B+）上的实验
- 在高类别多样性任务（如 NLUS、TREC、TREF）上，轻量编码路径与全 LLM 仍有差距
- 文本渲染为图像的方案对非拉丁文字（中文、日文等）效果虽然理论上更好（减少 token 数），但缺乏实验验证
- PG19、Proof 等文学/数学文本上 PPL 略逊于 CEPE*，说明纯文本语义型内容的压缩仍有提升空间

## 相关工作与启发

- 与 CEPE（文本编码器压缩）形成直接对比：Vist 用视觉编码器替代文本编码器，在性能相当甚至更优的同时显著降低显存
- 与 LLMLingua 系列（基于 LLM 信息熵的选择式压缩）互补：Vist 不依赖 LLM 计算 token 重要性
- Pixel（将文本渲染为图像进行预训练）的思路延伸到了长上下文压缩场景
- 启发：轻量视觉编码器在某些场景下可以作为 LLM 的"前端降噪器"，这一思路可能拓展到多模态 RAG 等场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)
- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](../../ICML2025/model_compression/radio_rate-distortion_optimization_for_large_language_model_compression.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](../../ACL2025/model_compression/quantification_of_large_language_model_distillation.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](../../ACL2025/model_compression/aligndistil_token_level_alignment.md)

</div>

<!-- RELATED:END -->
