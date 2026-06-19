---
title: >-
  [论文解读] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity
description: >-
  [ACL 2025][多模态VLM][自适应视觉粒度] 在 LLaVA-NeXT 上增加视觉粒度缩放器（空间金字塔池化获取多级粒度 token）和视觉粒度路由器（基于图像+指令自适应选粒度），并提出 RGLF 训练范式用 LMM 自身的生成概率作为反馈来训练路由器，在 11 个基准上实现"减少 token 反而提升性能"的效果。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "自适应视觉粒度"
  - "视觉token压缩"
  - "MoE路由"
  - "RGLF训练"
  - "LLaVA-NeXT"
---

# AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity

**会议**: ACL 2025  
**arXiv**: [2410.02745](https://arxiv.org/abs/2410.02745)  
**代码**: [GitHub](https://github.com/DeepLearnXMU/AVG-LLaVA)  
**领域**: 多模态VLM  
**关键词**: 自适应视觉粒度, 视觉token压缩, MoE路由, RGLF训练, LLaVA-NeXT

## 一句话总结

在 LLaVA-NeXT 上增加视觉粒度缩放器（空间金字塔池化获取多级粒度 token）和视觉粒度路由器（基于图像+指令自适应选粒度），并提出 RGLF 训练范式用 LMM 自身的生成概率作为反馈来训练路由器，在 11 个基准上实现"减少 token 反而提升性能"的效果。

## 研究背景与动机

**领域现状**：高分辨率 LMM（如 LLaVA-NeXT、Monkey 等）为了处理高分辨率图像，通常将图像切分为多个局部图像分别编码，再与全局图像 token 拼接送入 LLM。这种做法虽然提升了细粒度感知能力，但代价是视觉 token 数量暴增——例如一张 672×672 的图像在 LLaVA-NeXT 中会产生 2880 个视觉 token。

**现有痛点**：这些视觉 token 中存在大量冗余。认球衣号码需要细粒度，但问球衣颜色只需粗粒度——现有方法对所有图像-问题对都使用同样多的视觉 token，既浪费计算资源，冗余信息还可能干扰 LLM 的推理。已有的 token 压缩方法（如 FastV 剪枝、LLaVA-PruMerge 合并）虽然能减少 token 数量，但往往伴随性能下降；LLaVA-M3 虽支持多粒度，但需要手动指定粒度级别。

**核心矛盾**：视觉 token 数量与任务需求之间的不匹配——固定粒度策略无法同时满足"简单问题少用 token 加速推理"和"复杂问题保留细粒度信息"这两个需求。

**本文目标** (1) 如何自动根据输入图像和用户指令选择合适的视觉粒度？(2) 如何有效训练这个粒度选择器——直接用视觉指令微调无法学到粒度区分能力？

**切入角度**：借鉴人类视觉认知——"难题仔细看，简单题扫一眼"。将不同粒度的视觉特征类比为 MoE 中的不同专家，用路由器根据图像和指令动态选择最合适的"专家"（粒度级别）。

**核心 idea**：用 MoE 风格的路由器根据图像+指令自适应选择视觉粒度，并通过 LMM 自身的生成概率反馈来训练路由器的粒度偏好排序。

## 方法详解

### 整体框架

AVG-LLaVA 在 LLaVA-NeXT 的基础上新增两个模块：(1) 视觉粒度缩放器（Visual Granularity Scaler）通过多级池化获得不同粒度的视觉 token；(2) 视觉粒度路由器（Visual Granularity Router）根据图像和指令自适应选择最合适的粒度。整体流程为：图像 → 视觉编码器（CLIP ViT-L/14）→ 视觉粒度缩放器（生成 5 级粒度 token）→ 视觉粒度路由器（选择 1 个粒度）→ 选中粒度的 token 送入 LLM 生成回答。训练分两阶段：先让模型学会处理多粒度视觉信息，再用 RGLF 训练路由器。

### 关键设计

1. **视觉粒度缩放器（Visual Granularity Scaler）**:

    - 功能：将原始视觉 token 变换为多个粒度级别的 token 序列
    - 核心思路：采用空间金字塔池化设计，交替堆叠 1×2 和 2×1 平均池化层。以 CLIP ViT-L/14（输出 24×24 grid）为例，依次获得 24×12（288 token）、12×12（144 token）、12×6（72 token）、6×6（36 token）四级更粗的粒度，加上原始的 24×24（576 token），共 5 级粒度。这种交替池化方式在减半 token 数量的同时尽量保留空间布局信息
    - 设计动机：无需训练参数，纯操作性的模块，利用金字塔结构自然获得由细到粗的多级视觉表示，为后续路由器提供候选粒度

2. **视觉粒度路由器（Visual Granularity Router）**:

    - 功能：根据输入图像和用户指令，从 5 个粒度级别中选择最合适的一个
    - 核心思路：受 MoE 启发，将不同粒度视觉特征视为不同专家。路由器包含三层结构——首先将所有粒度的视觉 token 展平拼接为 $\bar{X}_v = [X_v^1; X_v^2; \ldots; X_v^N]$，同时计算指令 token 与原始粒度视觉 token 的余弦相似度，保留 top-$k$（$k=32$）最相关的指令 token $\bar{X}_{\text{instruct}}$。然后将视觉 token 和过滤后的指令 token 拼接，送入一个 Transformer 层进行跨模态融合。接着 MLP 对每个 token 预测粒度 logits $Z_{\text{out}} \in \mathbb{R}^{L \times N}$。最后一个可学习权重矩阵（Voter）$W \in \mathbb{R}^{1 \times L}$ 对所有 token 的预测进行加权聚合，得到最终 logits $Z_{\text{final}} \in \mathbb{R}^{1 \times N}$，softmax 后选概率最高的粒度
    - 设计动机：不同于传统 MoE 用简单线性层做路由，这里需要同时考虑图像内容和指令语义，因此使用 Transformer 层做跨模态融合。top-$k$ 过滤指令 token 是为了剔除噪声（过多/过少指令 token 都会影响性能，实验显示 $k=32$ 最优）。Voter 层的设计让每个 token 的预测按学习到的重要性加权，而非简单平均

3. **RGLF 训练范式（Ranking Granularity based on LMM Feedback）**:

    - 功能：有效训练路由器学习区分不同粒度的优劣，选择最合适的粒度
    - 核心思路：冻结除路由器外的所有模块。对每个训练样本，分别用 5 个粒度的视觉 token 让 LMM 生成回答，计算各粒度下的答案 log 概率作为反馈信号。按 log 概率降序排列粒度，然后用排序损失 $\mathcal{L}_{\text{rank}} = \sum_{i} \sum_{j>i} \max(0, s_j - s_i + \lambda_{ij})$ 对齐路由器概率与 LMM 偏好，其中 margin $\lambda_{ij}$ 根据两个粒度间的 log 概率差动态调整。同时加上交叉熵损失 $\mathcal{L}_{\text{ce}}$ 让路由器学习直接预测 LMM 最偏好的粒度。总损失为 $\mathcal{L}_2 = \mathcal{L}_{\text{rank}} + \alpha \mathcal{L}_{\text{ce}}$，$\alpha = 0.1$
    - 设计动机：直接用视觉指令微调（通过 Gumbel-Softmax 反向传播）训练路由器效果很差，因为路由器无法从端到端训练中学到不同粒度之间的区分能力。RGLF 的关键洞察是：LMM 本身就知道哪个粒度更适合当前样本（通过生成概率体现），因此可以利用 LMM 的反馈作为"监督信号"来训练路由器。排序损失确保路由器学会粒度间的相对偏好顺序，交叉熵损失确保能选出最优粒度，两者互补

### 损失函数 / 训练策略

**Stage 1 - 多粒度视觉指令微调**：训练视觉编码器、连接器和 LLM，使用 1M 图文对。对每个样本分别用 $N$ 个粒度的视觉 token 做 next-token prediction，损失为所有粒度上的交叉熵均值 $\mathcal{L}_1 = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(x_t | X_v^i, X_{\text{instruct}}, X_{a,<t})$。这一阶段让模型学会在不同粒度下都能理解图像。

**Stage 2 - RGLF 路由器训练**：冻结其他模块，只训练路由器。使用相同的 1M 数据，学习率设为 1e-3（远高于 Stage 1 的 1e-5），总训练开销约 14 小时（8×H800），仅为 Stage 1（65 小时）的 1/5。

## 实验关键数据

### 主实验

| 基准 | 类型 | LLaVA-NeXT | AVG-LLaVA | 提升 |
|------|------|-----------|-----------|------|
| GQA | 通用VQA | 64.2 | 63.0 | -1.2 |
| ScienceQA | 通用VQA | 70.1 | 71.1 | +1.0 |
| VizWiz | 通用VQA | 57.6 | 59.8 | +2.2 |
| TextVQA | 文本VQA | 64.9 | 67.1 | +2.2 |
| ChartQA | 文本VQA | 54.8 | 66.3 | **+11.5** |
| DocVQA | 文本VQA | 74.4 | 74.6 | +0.2 |
| AI2D | 文本VQA | 66.6 | 67.3 | +0.7 |
| MME | 多模态 | 1519.0 | 1557.4 | +38.4 |
| MMB | 多模态 | 67.4 | 69.9 | **+2.5** |
| POPE | 多模态 | 86.5 | 87.4 | +0.9 |
| MMMU | 多模态 | 35.8 | 37.4 | +1.6 |

效率对比（AVG-LLaVA vs LLaVA-NeXT）：

| 基准 | Token 减少比例 | 推理加速 |
|------|-------------|---------|
| AI2D | **85.3%** | **2.53×** |
| MME | 69.3% | 1.19× |
| ScienceQA | 54.9% | 1.41× |
| GQA | 80.0% | 1.14× |
| VizWiz | 26.4% | 1.77× |
| MMMU | 30.0% | 1.87× |

### 消融实验

| 配置 | ScienceQA | ChartQA | MME | MMB |
|------|-----------|---------|-----|-----|
| AVG-LLaVA (完整) | 71.1 | 66.3 | 1557.4 | 69.9 |
| 固定粒度代替自适应 | 70.0 | 66.4 | 1554.5 | 68.7 |
| 随机选择代替路由器 | 69.7 | 56.8 | 1535.7 | 67.9 |
| 路由器仅用图像（去掉指令）| 70.1 | 53.9 | 1525.2 | 69.0 |
| 减少粒度范围 {36,144,576} | 69.8 | 65.3 | 1547.7 | 66.3 |
| 用指令微调训练路由器 | 70.5 | 50.9 | 1514.8 | 68.6 |
| 去掉排序损失 | 70.1 | 64.8 | 1534.6 | 68.6 |
| 去掉交叉熵损失 | 70.2 | 66.3 | 1550.8 | 69.4 |

### 关键发现

- **指令信息对路由至关重要**：去掉指令 token 后 ChartQA 暴跌 12.4 分，说明同一张图对不同问题确实需要不同粒度
- **RGLF 远优于端到端训练**：用 Gumbel-Softmax 做视觉指令微调来训练路由器，ChartQA 只有 50.9（比 RGLF 低 15.4 分），说明路由器很难通过端到端梯度学到粒度区分
- **排序损失比交叉熵损失更关键**：去掉排序损失后 MME 下降 22.8，去掉交叉熵损失仅下降 6.6。排序损失提供了粒度间的相对偏好信号，仅靠 CE 学到的是"最优粒度"但缺乏全局排序能力
- **中间粒度虽然很少被选中，但不能删除**：72 和 288 token 的粒度虽然占比极低，但移除后性能下降，说明它们帮助模型渐进式学习不同粒度的差异
- **文本密集型任务倾向细粒度**：TextVQA/ChartQA/DocVQA 中路由器主要选择最细粒度（576 token），而 AI2D/MMMU 等概念性任务倾向粗粒度（36 token），行为符合直觉

## 亮点与洞察

- **减少冗余 token 反而提升性能**：这是最反直觉的发现。AVG-LLaVA 在 AI2D 上只用 14.7% 的 token 就超过了 LLaVA-NeXT，说明过多的视觉 token 实际上会引入噪声干扰 LLM 推理。这个发现对整个 MLLM 效率研究方向有重要指导意义
- **RGLF 训练范式是核心贡献**：利用 LMM 自身的生成概率作为排序信号来训练路由器，巧妙地避免了"需要人工标注每个样本应该用什么粒度"的问题。这种"用模型自身反馈训练辅助模块"的思路可以迁移到其他需要离散选择的场景（如 MoE 专家选择、检索增强中的文档选择等）
- **动态 margin 设计**：排序损失中的 margin $\lambda_{ij}$ 不是固定的，而是根据 LMM 在不同粒度下的 log 概率差自动计算，让惩罚程度与粒度差异成正比，避免对差异很小的粒度对施加过大的分离力

## 局限与展望

- **粒度级别跳跃过大**：每次池化直接减半 token 数，导致相邻粒度间差距大（576→288→144→72→36）。在文本密集型任务上路由器几乎只选 576，说明没有足够细的中间粒度可选。设计更平滑的粒度缩放网络（如可变步长池化）可能有帮助
- **两阶段训练增加开销**：Stage 1 需要对每个样本用所有 5 个粒度分别前向传播，训练成本约为单粒度的 5 倍。虽然 Stage 2 很快（14 小时），但能否将两阶段合并为交替训练值得探索
- **仅验证了 LLaVA-NeXT 架构**：是否能推广到 InternVL、Qwen-VL 等其他架构？路由器设计是否需要针对不同架构调整？
- **路由器参数量小但推理有额外开销**：虽然仅增加 1.66% 参数量，但需要先生成所有粒度的 token 再选择一个，在推理时所有粒度的缩放仍然要执行

## 相关工作与启发

- **vs FastV/VTW（token 剪枝）**：FastV 在第 2 层 decoder 后按注意力分数剪枝 50% token，VTW 更激进地在某层后删除所有视觉 token。这些方法是"编码后再删"，容易丢失关键信息；AVG-LLaVA 是"编码前就选好粒度"，保留了所选粒度的完整空间结构
- **vs LLaVA-PruMerge（剪枝+合并）**：PruMerge 用 class token 相似度来决定保留/合并哪些 token，是静态策略不考虑指令；AVG-LLaVA 的路由器同时看图像和指令，能根据问题动态调整
- **vs LLaVA-M3 / MQT-LLaVA（多粒度）**：LLaVA-M3 支持多粒度但需要用户手动指定；MQT-LLaVA 用嵌套 dropout 训练兼容多粒度。AVG-LLaVA 的优势在于自动选择+RGLF 训练对齐，无需人工干预
- **RGLF 与 RLHF 的类比**：RGLF 本质上是从 LMM 获取偏好信号来训练路由器，类似 RLHF 从人类偏好训练奖励模型。不同之处在于 RGLF 的偏好信号是客观的 log 概率而非主观标注，更稳定

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应粒度选择思路直觉但有效，RGLF 训练范式是亮点；但粒度缩放器本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个基准全面覆盖，消融实验涵盖 7 个维度，路由可视化和注意力图分析深入
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述层次分明，架构图和可视化丰富
- 价值: ⭐⭐⭐⭐ 对 MLLM 推理效率有直接实用价值，RGLF 范式可迁移；但仅在 7B 规模验证，缺少更大模型的验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[CVPR 2026\] Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM](../../CVPR2026/multimodal_vlm/granulon_awakening_pixel-level_visual_encoders_with_adaptive_multi-granularity_s.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
