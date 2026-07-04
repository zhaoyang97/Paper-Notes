---
title: >-
  [论文解读] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models
description: >-
  [ICML2025][多模态VLM][Theory of Mind] 本文从可解释性角度评估多模态大模型（MLLM）的心智理论（ToM）能力，构建了基于 2D 网格世界的多模态 ToM 数据集 GridToM，并提出一种无需训练的注意力头激活干预方法来显著提升模型的 ToM 表现。 核心问题：大语言模型/多模态大模型是否真正…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "Theory of Mind"
  - "多模态大模型"
  - "可解释性"
  - "注意力干预"
  - "GridToM"
---

# From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models

**会议**: ICML2025  
**arXiv**: [2506.14224](https://arxiv.org/abs/2506.14224)  
**代码**: [项目页](https://annaisavailable.github.io/GridToM)  
**领域**: 多模态VLM  
**关键词**: Theory of Mind, 多模态大模型, 可解释性, 注意力干预, GridToM  
**作者**: Xinyang Li, Siqi Liu, Bochao Zou, Jiansheng Chen, Huimin Ma

## 一句话总结

本文从可解释性角度评估多模态大模型（MLLM）的心智理论（ToM）能力，构建了基于 2D 网格世界的多模态 ToM 数据集 GridToM，并提出一种无需训练的注意力头激活干预方法来显著提升模型的 ToM 表现。

## 研究背景与动机

**核心问题**：大语言模型/多模态大模型是否真正具备心智理论（Theory of Mind, ToM）——即推断他人信念、意图、认知状态的能力？

**现有方法的不足**：

**单模态局限**：大多数 ToM 评估仅基于文本或视频单一模态，缺乏多模态整合评估。

**黑盒评估**：现有方法几乎完全依赖问答任务的输出正确率来判断模型是否具备 ToM，忽略了模型内部是否真正形成了区分不同视角的认知表征。

**感知信息缺失**：真实场景视频数据集（如 MMToM-QA）无法精确控制角色是否"看到"某物体，评估准确性受感知信息质量影响。

**幻觉干扰**：MLLM 存在幻觉现象，模型可能"理解"概念但给出错误回答，QA 正确率不能完全反映 ToM 能力。

**本文动机**：从模型内部机制出发，通过可解释性方法探究 MLLM 是否在中间层形成了区分不同视角信念状态的内部表征，并据此进行针对性干预以增强 ToM 表现。

## 方法详解

### 1. GridToM 数据集构建

基于 Multigrid/Minigrid 库构建的 2D 网格世界数据集：

- **规模**：1,296 个视频-文本对，分辨率 294×420，每段约 40 帧
- **场景**：10×7 网格，3 个房间，2 个智能体
- **配置多样性**：27 种地图 × 2 种初始位置 × 2 种朝向 × 6 种移动序列 × 真/假信念配对
- **任务类型**：初始信念测试、一阶信念测试、二阶信念测试
- **关键设计**：关门时遮挡门后信息，模拟真实感知限制；为每个智能体提供完整的物理视角信息
- **测试集 500 样本**，训练/验证集 148 样本（75%/25% 划分）

与此前数据集的核心区别在于：GridToM 提供可操控的多模态视觉-语言因果故事，包含所有角色的完整感知信息，且 2D 网格世界简化了高层信息，让模型聚焦于核心 ToM 推理。

### 2. 注意力特征提取

模型将视觉输入 $V=\{v_1, v_2, \ldots, v_m\}$ 和文本输入 $X=\{x_1, x_2, \ldots, x_n\}$ 拼接为统一序列 $T = \text{concat}(V, X) \in \mathbb{R}^{(m+n) \times DH}$，经过 $L$ 层 Transformer 处理。

多头注意力近似为：

$$T_{l+1} = T_l + \sum_{h=1}^{H} Attn_l^h(P_l^h T_l) \cdot W_l^o$$

其中 $P_l^h \in \mathbb{R}^{D \times DH}$ 将激活投影到 $D$ 维注意力头空间，$W_l^o$ 为输出投影矩阵。

提取每层每个注意力头在最后一个 token 位置的激活 $X \in \mathbb{R}^{L \times H \times D}$，并关联两类信念标签：
- $Y_p$：主角视角信念正确性
- $Y_o$：全知视角信念正确性

标签设计统一了**视角分离**和**信念推理**两个方面：$Y_p = \{Y_p^{TB} \cap Y_p^{FB}\}$。

### 3. 探针分析（Probing）

对每个注意力头训练独立的逻辑回归二分类探针：

$$f_l^h = \frac{1}{1 + e^{-(x\theta + b)}}$$

其中 $\theta \in \mathbb{R}^D$ 为权重向量，$b \in \mathbb{R}$ 为偏置，通过最小化交叉熵损失优化。

**关键发现**：
- 从中间层到最终层，大量注意力头能准确捕获主角视角的信念状态
- 四种信念组合（TB/FB × 正确/错误）在表征空间中形成无重叠的明确聚类
- 信念表征具有**线性可分性**，说明 MLLM 确实在内部形成了与多视角信息提取和信念推理相关的中间表征

### 4. 注意力干预（Intervention）

选择验证集上灵敏度最高的 Top-$K$ 注意力头，在多头注意力计算后、映射回输出前进行干预：

$$T_{l+1} = T_l + \sum_{h=1}^{H} \left(Attn_l^h(P_l^h T_l) + \alpha \sigma_l^h \theta_l^h \right) \cdot W_l^o$$

- $\sigma_l^h$：激活沿目标方向的标准差
- $\theta_l^h$：干预目标方向，来自探针的权重向量
- $\alpha$：干预强度参数

该方法**完全无需训练**，仅需通过少量样本训练线性探针获取方向，然后在推理时沿该方向偏移激活即可。

## 实验关键数据

### Baseline 评估（GridToM 一阶信念，多模态设置）

| 模型 | TB(%) | FB(%) | Both(%) |
|------|-------|-------|---------|
| Human | 99.9 | 99.9 | 99.8 |
| GPT-4o | 6.2 | 100.0 | 6.2 |
| Doubao-1.5-vision-pro | 16.8 | 100.0 | 16.8 |
| LLaVA-Next-Video-7B | 53.2 | 42.8 | 0.8 |
| Qwen2-VL-7B | 26.6 | 97.0 | 23.6 |

### 干预后提升（多模态，一阶信念）

| 模型 | TB(%) | FB(%) | Both(%) |
|------|-------|-------|---------|
| LLaVA-Next-Video +α | 63.8 (+10.6) | 51.6 (+8.8) | 22.0 (+21.2) |
| Qwen2-VL +α | 60.4 (+33.8) | 97.4 (+0.4) | 31.2 (+7.6) |

### 关键观察

1. **TB/FB 不平衡**：GPT-4o 在 FB 上达 100% 但 TB 仅 6.2%，说明模型过度依赖 FB 模式而非真正理解信念
2. **Both 指标极低**：所有 MLLM 在 Both（同时答对 TB 和 FB）上表现极差，暴露了推理能力的脆弱性
3. **文本 vs 视频**：纯文本条件下 LLM（如 Doubao-1.5-Pro-32k）可达 100%，但多模态/视频条件下大幅下降
4. **二阶信念更难**：所有模型在二阶信念任务上接近随机水平（~50%）
5. **干预有效**：无训练的注意力干预在 Both 指标上带来 +7.6~+21.2% 的显著提升

## 亮点与洞察

1. **从可解释性角度切入 ToM 评估**，不依赖 QA 输出而是分析内部表征，避免了幻觉等因素的干扰
2. **线性可分性发现**：证明 MLLM 内部确实存在编码不同视角信念状态的注意力头，且信念表征是线性可分的
3. **无训练干预方法**：仅需少量探针训练即可在推理时提升 ToM 能力，具有良好的实用性和轻量性
4. **数据集设计精巧**：2D 网格世界避免了高层语义干扰，可精确控制感知信息，使评估更纯粹
5. **Both 指标的引入**：比单独看 TB 或 FB 更能反映模型的真实信念推理能力，避免了偏倚模式带来的虚假高分

## 局限与展望

1. **场景过于简化**：2D 网格世界与真实世界场景差距大，结论能否迁移到复杂视觉场景仍需验证
2. **模型覆盖有限**：干预实验仅在 LLaVA-Next-Video-7B 和 Qwen2-VL-7B 上验证，缺少更大规模模型（如 GPT-4o）的内部分析（闭源无法探测）
3. **干预依赖超参数**：$K$ 和 $\alpha$ 的选择依赖验证集，不同任务/模型可能需要调整
4. **仅限 GridToM 干预**：由于 MMToM-QA 数据集限制，干预实验仅在 GridToM 上完成，实际场景的泛化性存疑
5. **探针本身的局限**：线性探针可能遗漏非线性编码的信念信息，probing 准确率高不能完全等价于"模型理解了 ToM"

## 可复现性要点

- 数据集基于开源 Multigrid/Minigrid 库生成，项目页面提供数据
- 探针为标准逻辑回归，计算开销极低
- 干预方法无需额外训练，仅需前向传播中插入偏移
- 所有模型温度设为 0，输入为 4 关键帧 + 3 中间帧
- 测试集 500 样本，训练/验证集 148 样本（75%/25%）

## 个人点评

这篇工作的核心贡献在于将 ToM 评估从"看输出猜能力"转向"看内部表征验能力"，方法论上更为扎实。GridToM 数据集虽然场景简单，但正因简单才能精确控制变量，是一个好的实验范式。注意力头的线性可分性发现是一个有价值的实证结果，为理解 MLLM 的认知能力提供了新视角。

不过，从实用角度看，2D 网格世界的 ToM 与真实社交场景的 ToM 差距巨大。模型在简单格子世界中能区分视角，不代表在复杂社交视频中也能做到。干预方法虽然轻量有效，但提升幅度有限（Both 指标最高也只到 31.2%），距离人类水平（99.8%）仍有巨大差距。此外，"模型内部存在信念表征"和"模型真正理解信念"之间仍存在解释鸿沟——线性可分的特征也可能是统计相关性的副产物而非因果性理解。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[ICML 2025\] ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs](enhancing_rating-based_reinforcement_learning_to_effectively_leverage_feedback_f.md)
- [\[ICCV 2025\] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)

</div>

<!-- RELATED:END -->
