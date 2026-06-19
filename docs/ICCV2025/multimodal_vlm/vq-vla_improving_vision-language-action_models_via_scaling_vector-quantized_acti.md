---
title: >-
  [论文解读] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
description: >-
  [ICCV2025][多模态VLM][动作tokenizer] 本文提出基于卷积残差 VQ-VAE 的动作 tokenizer，在比先前方法多 100 倍的训练数据（含大量合成数据）上训练后可零样本迁移到各种下游 VLA 任务，在真实机器人上将长时域任务成功率提升最高 30%，推理速度提升近 3 倍。
tags:
  - "ICCV2025"
  - "多模态VLM"
  - "动作tokenizer"
  - "VQ-VAE"
  - "合成数据缩放"
  - "长时域任务"
  - "机器人操控"
---

# VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers

**会议**: ICCV2025  
**arXiv**: [2507.01016](https://arxiv.org/abs/2507.01016)  
**代码**: [https://github.com/VQ-VLA](https://github.com/VQ-VLA) (待确认)  
**领域**: 机器人 / VLA / 动作表示  
**关键词**: 动作tokenizer, VQ-VAE, 合成数据缩放, 长时域任务, 机器人操控

## 一句话总结
本文提出基于卷积残差 VQ-VAE 的动作 tokenizer，在比先前方法多 100 倍的训练数据（含大量合成数据）上训练后可零样本迁移到各种下游 VLA 任务，在真实机器人上将长时域任务成功率提升最高 30%，推理速度提升近 3 倍。

## 研究背景与动机

**领域现状**：Vision-Language-Action（VLA）模型将视觉语言理解和机器人控制结合在一起，当前主流做法（如 OpenVLA、RT-2）通过将连续动作离散化为 256 个 bin（每维度独立分箱）来适配 LLM 的 token 预测框架。

**现有痛点**：
   - 简单的 per-dimension binning 离散化精度有限，在长时域任务中误差会逐步累积导致失败
   - 每步只预测一个动作的 token，推理速度慢（OpenVLA 仅 4.16 Hz）
   - 动作表示质量受限于训练数据规模和多样性，但扩大 VLA 整体训练成本极高

**核心矛盾**：VLA 模型需要高精度、高效率的动作表示，但简单的 bin 离散化在精度和序列压缩上都有天花板；同时训练整个 VLA 模型的成本过高，需要一条低成本的性能提升路径。

**本文目标**：设计一个可缩放的通用动作 tokenizer，(a) 提高动作表示精度和长时域鲁棒性，(b) 通过动作 chunking 加速推理，(c) 利用合成数据低成本地扩大训练规模。

**切入角度**：作者观察到一个关键现象——**动作轨迹在仿真与真实之间的 domain gap 极小**（不像图像或物理属性），因此可以大量使用合成轨迹来训练 tokenizer 而不损失真实世界性能。

**核心 idea**：用卷积残差 VQ-VAE 做动作 tokenizer，通过渐进式策略在 100 倍规模的混合数据（真实+合成）上训练，实现 VLA 的精度、速度和长时域能力同步提升。

## 方法详解

### 整体框架
VQ-VLA 的 pipeline 分为两阶段：
- **阶段一**：训练一个通用卷积残差 VQ-VAE 动作 tokenizer。输入是长度为 $n$ 的动作序列 $\mathbf{a}_{t:t+n} \in \mathbb{R}^{n \times d}$（7 维：XYZ + Euler 角 + 夹爪），输出为 $N_q$ 个离散 token（对应 $N_q$ 层 RVQ 量化）
- **阶段二**：冻结 VQ-VAE，将其作为 OpenVLA 的动作 tokenizer 替换原始 binning 方案，用 LoRA 微调 OpenVLA

### 关键设计

1. **卷积残差 VQ-VAE 架构**:

    - 功能：将连续动作序列编码为离散 token，解码时恢复动作序列
    - 核心思路：编码器和解码器使用 2D 时序卷积层（而非 MLP），能更好捕捉局部时序关系和层级时序依赖。残差向量量化（RVQ）将隐变量分解为多层量化：$\mathbf{q}(\mathbf{x}) = \sum_{i=1}^{N_q} \mathbf{q}_i(\mathbf{r}_i)$，每层修正前一层的残差
    - 训练损失：$\mathcal{L} = \|\mathbf{a} - \hat{\mathbf{a}}\|_2^2 + \lambda(\|\text{sg}(\mathbf{x}) - \mathbf{q}(\mathbf{x})\|_2^2 + \|\mathbf{x} - \text{sg}(\mathbf{q}(\mathbf{x}))\|_2^2)$，$\lambda=4$
    - 设计动机：2D 时序卷积相比 MLP 在 LIBERO 上成功率从 53.4% 提升到 60%，说明局部时序建模至关重要

2. **时间嵌入 + 动作类型嵌入**:

    - 功能：在动作序列输入编码器前添加两种嵌入
    - 核心思路：正弦时间嵌入（sinusoidal）编码不同频率的时序信息；可学习的动作类型嵌入区分 7 个维度（XYZ、欧拉角、夹爪）的不同语义角色
    - 设计动机：动作向量的 7 个维度含义各异，需要先验信息帮助模型区分处理

3. **渐进式训练策略 + 合成数据缩放**:

    - 功能：从真实数据到合成数据逐步扩大训练规模
    - 核心思路：首先在 Open X-Embodiment（真实但噪声大）上训练，然后逐步加入 LIBERO 和 ManiSkill 的仿真数据（更干净平滑）。三个版本：VQ_O（仅 OpenX）、VQ_{O+L}（+LIBERO）、VQ_{O+L+M}（+ManiSkill）
    - 设计动机：作者发现动作轨迹在 sim-real 之间的 domain gap 极小（VQ_L 纯仿真训练的性能与 VQ_{O+L} 相当），因此可以放心使用大量合成数据

4. **VQ-VAE 与 VLA 的集成**:

    - 功能：用 VQ-VAE 的离散 token 替换 OpenVLA 的 binning token
    - 核心思路：不同 RVQ 层的 token ID 使用不重叠的范围——第 $i$ 层的 token $z_q^i \in [256(i-1), 256i-1]$，避免不同层之间的语义混淆。VLM 直接预测这些 token，损失为标准的 next-token cross-entropy
    - 设计动机：用压缩比为 5 的 action chunking（一步预测 5 个动作），大幅减少推理步数

### 损失函数 / 训练策略
- VQ-VAE 训练：重建损失 + VQ 损失 + commitment 损失，$\lambda=4$
- VLA 微调：标准 next-token prediction cross-entropy，LoRA 微调 400K 步（仿真）或 100K 步（真实）

## 实验关键数据

### 仿真主实验（LIBERO-90）

| 方法 | 训练数据 | LIBERO-90 成功率 |
|------|---------|-----------------|
| OpenVLA baseline | - | 73.53% |
| VQ_M (仅 ManiSkill) | ManiSkill | 14.38% |
| VQ_{M+R} (ManiSkill+RLBench) | ManiSkill+RLBench | **80.98%** |

VQ_{M+R} 比 baseline 提升 7.45%。数据不足时（VQ_M）性能大幅下降，验证了数据规模的重要性。

### 真实机器人实验

| 任务 | Baseline | VQ_O | VQ_{O+L} | VQ_{O+L+M} |
|------|----------|------|----------|------------|
| Pull tissue | 5% | 20% | 25% | 25% |
| Pick toy (avg 3) | 30% | 46.7% | 43.3% | 50% |
| Flip pot upright | 30% | 45% | 45% | **60%** |
| Put toy in basket | 20% | 35% | 35% | **45%** |
| Put cups in basket (长时域) | 15% | - | - | **50%** |
| Put toy in drawer (长时域) | ~0% | 15% | 10% | **25%** |

### 消融实验

| Action Chunking 方式 | LIBERO-90 | Flip pot | Put in basket |
|---------------------|-----------|----------|---------------|
| Baseline (单步) | 74.76% | 30% | 20% |
| Autoregressive Output | 66.53% | 10% | 0% |
| VQ-based (VQ_{O+L+M}) | **86.61%** | **60%** | **45%** |

自回归式 action chunking 反而大幅下降（出现 shortcut learning 现象——chunk 内多个动作值高度相似），而 VQ-based chunking 表现最优。

### 推理速度

| 方法 | 频率 (Hz) |
|------|----------|
| OpenVLA | 4.16 |
| VQ-VLA | **11.84** |

推理速度提升约 2.85 倍。

### 关键发现
- **合成数据缩放有效**：ManiSkill 数据量是 LIBERO 的 50 倍，加入 ManiSkill 后短时域平均成功率从 37.5% 提升到 46.25%
- **Sim-real domain gap 极小**：纯仿真训练的 VQ_L 性能与 VQ_{O+L} 相当（Flip pot: 55% vs 45%）
- **长时域任务受益最大**：VQ 的 action chunking 减少了累积误差，Put cups in basket 从 15% 提升到 50%
- **嵌入有帮助**：加入时间嵌入和动作类型嵌入后 LIBERO-90 成功率从 85.17% 提升到 86.16%

## 亮点与洞察
- **"动作轨迹的 sim-real gap 极小"是一个深刻发现**：不同于图像、物理属性等模态，动作轨迹的统计分布在仿真和真实间高度一致。这意味着可以用极低成本的仿真数据来提升动作表示质量，为 VLA 提供了一条"偏科补课"的高效路径
- **VQ-based action chunking 优于自回归 chunking**：这揭示了 LLM 的自回归生成在低维连续信号（动作）上容易产生 shortcut learning，而 VQ 的显式压缩-解压缩能更好保留序列内的变化性
- **tokenizer 训练成本极低**：仅需单卡 A100 训练一周，却能为下游 VLA 带来一致性的性能/速度提升。这种"小组件大收益"的思路可以迁移到其他模态的 tokenizer 设计

## 局限与展望
- **仅在 OpenVLA 上验证**：架构通用性声称强但只替换了一种 VLA 的 tokenizer，需要在更多 VLA（如 RT-2、Octo）上验证
- **动作空间限定为 7-DoF SE(3)**：未涉及灵巧手等更高维动作空间
- **缺少与其他动作 tokenizer 的直接对比**：如 cosine transform 方法（FAST）
- **可改进方向**：(a) 结合动作数据的频率特征作为额外条件；(b) 与 VLM 的蒸馏/量化结合进一步加速；(c) 扩展到更大规模仿真数据集（如 RLBench on CoppeliaSim）

## 相关工作与启发
- **vs OpenVLA (binning)**: OpenVLA 将每维动作做 256-bin 离散化，精度受限且每步只预测一个动作。VQ-VLA 用 RVQ 实现更精细的量化，且通过 action chunking 一步预测 5 个动作
- **vs FAST (cosine transform)**: FAST 用余弦变换做动作 tokenization，是另一条技术路线。本文没有直接对比，但两者思路互补
- **vs MiniVLA**: MiniVLA 也关注 VLA 效率，但从模型压缩角度出发。VQ-VLA 从动作表示角度提效，两者可以结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 卷积残差 VQ-VAE + 合成数据缩放的组合有新意，sim-real gap 的发现有价值
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实全覆盖，消融充分，但缺少与 FAST 等方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分符号使用不够一致
- 价值: ⭐⭐⭐⭐ 为 VLA 提供了一条低成本性能提升路径，实用价值高

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](../../CVPR2026/multimodal_vlm/from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](../../CVPR2026/multimodal_vlm/joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](../../CVPR2026/multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)

</div>

<!-- RELATED:END -->
