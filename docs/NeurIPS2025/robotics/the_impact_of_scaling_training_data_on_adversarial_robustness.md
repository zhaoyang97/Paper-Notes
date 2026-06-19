---
title: >-
  [论文解读] The Impact of Scaling Training Data on Adversarial Robustness
description: >-
  [NeurIPS 2025][机器人][对抗鲁棒性] 系统评估 36 个 SOTA 视觉模型在 6 类黑盒攻击下的鲁棒性，发现攻击成功率(ASR)随数据量和模型规模按对数律下降，但 数据质量和模型规模比数据量本身更关键：。 领域现状： DNN 在 CV 任务上取得了巨大成功，但对抗样本仍然是安全关键应用部署的根本挑战…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "对抗鲁棒性"
  - "缩放定律"
  - "黑盒攻击"
  - "数据质量"
  - "视觉模型"
---

# The Impact of Scaling Training Data on Adversarial Robustness

**会议**: NeurIPS 2025  
**arXiv**: [2509.25927](https://arxiv.org/abs/2509.25927)  
**代码**: 无  
**领域**: 音频/语音 (对抗鲁棒性)  
**关键词**: 对抗鲁棒性, 缩放定律, 黑盒攻击, 数据质量, 视觉模型

## 一句话总结

系统评估 36 个 SOTA 视觉模型在 6 类黑盒攻击下的鲁棒性，发现攻击成功率(ASR)随数据量和模型规模按对数律下降，但 **数据质量和模型规模比数据量本身更关键**。

## 研究背景与动机

**领域现状**: DNN 在 CV 任务上取得了巨大成功，但对抗样本仍然是安全关键应用部署的根本挑战。近年涌现了 ViT、DINOv2、CLIP 等不同训练范式的模型，训练数据规模从 120 万到 220 亿张不等。

**现有痛点**: 传统认为更大数据集和更复杂训练目标应带来更强鲁棒性，但实证发现有些小规模精选数据训练的模型反而比数据量大几个量级的模型更鲁棒。

**核心矛盾**: 数据量、数据质量、模型规模、训练范式对鲁棒性的独立贡献未被分离和量化。

**本文目标**: 建立训练数据特征与对抗鲁棒性之间的定量关系（缩放定律）。

**切入角度**: 跨越 36 个模型、6 类语义攻击的大规模系统性评估。

**核心 idea**: 对抗鲁棒性随数据量和模型大小对数下降，但模型规模的影响远大于数据量，而 DINOv2 等高质量策展数据训练的模型可以碾压数据量大百倍的 CLIP 模型。

## 方法详解

### 整体框架

构建了全面的黑盒评估框架：
- **36 个模型**: ViT、ResNet、CLIP、DINOv1/v2、Swin/v2、ConvNeXt、YOLO、ViT-MAE、PaliGemma、BEiT/v2、SigLIP/v2
- **6 类攻击**: Random Perturbations、GeometricMasksV1、GeometricMasksV2、COCO Objects、ImageNet-C、ImageNet-R
- **评估**: ImageNet-1K 验证集

### 关键设计

#### 评估指标
- **Accuracy**: $\text{Acc}(C, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \mathbf{1}[C(x) = y]$
- **攻击成功率 ASR**: $\text{ASR} = \frac{1}{|\mathcal{S}_{\text{correct}}|} \sum_{(x,y) \in \mathcal{S}_{\text{correct}}} \mathbf{1}[C(A(x)) \neq y]$
- 对于无法获取原始干净图片的场景，使用代理数据集近似 ASR，误差为 3.09pp ($\sigma=1.93$pp)

#### 对抗微调实验
三个 ResNet50 变体在 GeometricMasksV2 不同配置下微调：
- v1: 3-4-2 C1 (opacity=64, 50% 对抗样本)
- v2: 3-4-2 C1&C2
- v3: Random C1&C2

#### 人类评估
- GeometricMasksV2 6-7-2 C1，4 个难度级别（opacity 0/64/96/128）
- ImageNette 数据集，6 人参与评估

### 损失函数/训练策略

- CLIP 模型零样本评估，使用 "a photo of a {class name}" prompt
- DINOv1、ViT-MAE、PaliGemma 冻结骨干 + 线性分类头
- 对抗微调：从 ImageNet 预训练权重出发，batch=64，3 epochs

## 实验关键数据

### 主实验 — 缩放定律

| 维度 | 单变量缩放定律 | 含义 |
|-----|-------------|------|
| 数据量 | $\text{ASR} = -3.16 \log_{10}(x) + 55.53$ | 数据量增 10 倍 → ASR 降 ~3.2pp |
| 模型规模 | $\text{ASR} = -13.39 \log_{10}(x) + 141.18$ | 参数增 10 倍 → ASR 降 ~13.4pp |

**双变量缩放定律**（PCA 分离后）：
$$\text{ASR} = -0.46 \log_{10}(x_{\text{data}}) - 12.53 \log_{10}(x_{\text{model}}) + 137.67$$

模型规模的独立贡献远大于数据量！

### 模型排名

| 模型 | 训练数据量 | 总平均 ASR |
|-----|----------|----------|
| DINOv2-G | 142M | **10.3%** (最佳) |
| DINOv2-L | 142M | ~12% |
| Swinv2-L-384 | 14.2M | 16.8% |
| ResNet50 | 1.2M | ~50% (最差) |

### 人类 vs. 模型对比

| 难度 (opacity) | 人类 | DINOv2-B | ResNet-v1 (微调) | ResNet50 |
|---------------|------|---------|----------------|---------|
| 0 (干净) | ~100% | ~99% | ~98% | ~96% |
| 64 | ~97% | ~92% | ~93% | ~65% |
| 128 | ~93% | ~87% | ~87% | ~35% |

### 消融实验 — 对抗微调

- 对结构变化（形状、尺度、旋转）可以泛化 ✅
- 对颜色分布变化无法迁移 ❌ → 几何和色彩不变性是分开学习的

### 关键发现

1. 训练范式（监督/自监督/对比学习）对鲁棒性影响有限——对比学习 27.9% vs 监督 34.3% ASR
2. DINOv2 训练数据仅 142M，但 ASR 远低于训练数据达 22B 的 CLIP 模型
3. CLIP 在未控制数据质量的情况下，规模增大带来的收益有限
4. 人类在所有难度级别上始终优于最佳模型，最佳模型在高难度下仍有 ~13% 的误分类

## 亮点与洞察

- **首次建立视觉模型对抗鲁棒性的双变量缩放定律**，分离了数据量和模型规模的独立贡献
- **"质量 > 数量"的强有力证据**: DINOv2 (142M 高质量数据) >> CLIP (数十亿网络数据)
- **对抗微调的局限性**: 几何鲁棒性可迁移但色彩鲁棒性不可，揭示了视觉特征学习的模块化本质
- **生物 vs 人工视觉的持久差距**: 人类视觉系统的鲁棒性仍是人工模型难以企及的上界

## 局限与展望

1. 缺乏白盒梯度攻击（如 PGD、AutoAttack）的评估
2. 仅关注分类任务，未扩展到检测/分割
3. 训练数据集的文档不够标准化，难以精确控制变量
4. Future: 验证缩放趋势是否在梯度攻击下依然成立

## 相关工作与启发

- **RobustBench** (Croce et al., 2021): 标准化鲁棒性基准
- **DINOv2** (Oquab et al., 2024): 自监督视觉表示学习
- **Bartoldson et al., 2024**: 语言模型鲁棒性缩放研究
- **启发**: 在资源有限时，优先投资数据策展和模型规模，而非单纯增大数据量

## 评分

⭐⭐⭐⭐ (4/5)
大规模系统性评估提供了非常有价值的定量洞察，但攻击类型限于黑盒、任务限于分类，实验覆盖面可进一步扩展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](../../ICLR2026/robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)

</div>

<!-- RELATED:END -->
