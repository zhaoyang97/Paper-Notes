---
title: >-
  [论文解读] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation
description: >-
  [ECCV 2024][信号/通信][参数高效微调] 本文提出 PYRA，通过并行生成解耦的自适应调制权重并以 re-activation 策略调节待合并 token 的特征，实现了 Vision Transformer 在下游任务适配时同时兼顾训练效率（仅调 0.4% 参数）和推理效率（约 1.7-3.2 倍加速），性能与不压缩的 PEFT 方法持平甚至更优。
tags:
  - "ECCV 2024"
  - "信号/通信"
  - "参数高效微调"
  - "模型压缩"
  - "令牌合并"
  - "Transformer"
  - "训练推理高效"
---

# PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation

**会议**: ECCV 2024  
**arXiv**: [2403.09192](https://arxiv.org/abs/2403.09192)  
**代码**: [https://github.com/THU-MIG/PYRA](https://github.com/THU-MIG/PYRA)  
**领域**: Signal & Communication (模型高效化)  
**关键词**: 参数高效微调, 模型压缩, 令牌合并, Vision Transformer, 训练推理高效

## 一句话总结
本文提出 PYRA，通过并行生成解耦的自适应调制权重并以 re-activation 策略调节待合并 token 的特征，实现了 Vision Transformer 在下游任务适配时同时兼顾训练效率（仅调 0.4% 参数）和推理效率（约 1.7-3.2 倍加速），性能与不压缩的 PEFT 方法持平甚至更优。

## 研究背景与动机
随着 Vision Transformer 规模增长到数十亿参数，下游任务适配面临两个核心挑战：训练开销（全量微调消耗巨大 GPU 资源）和推理效率（部署时模型计算量过大）。现有方案分别处理这两个问题：PEFT（如 LoRA）冻结骨干仅调少量参数解决训练效率，但不改善推理速度；模型压缩（如剪枝）提升推理效率，但需要大量训练资源进行结构搜索和重训练。二者简单组合并不能同时保证两端效率——在低压缩率下性能会有轻微下降，在高压缩率（>3倍加速）下性能急剧恶化，甚至不如直接在对应吞吐的小模型上微调（作者称之为"Adverse Compression"现象）。核心矛盾在于：PEFT 的有限参数更新能力无法充分感知下游数据分布，而 token 合并过程中的信息丢失在逐层传播中不断累积，最终导致性能崩塌。本文的切入角度：在 token 合并前对待合并 token 进行自适应特征调制，用极少量可学习参数补偿合并带来的信息损失。核心 idea：并行生成 channel 和 token 两个维度的解耦调制权重，通过 sigmoid re-activation 策略稳健地调制 token 特征。

## 方法详解

### 整体框架
PYRA 基于 LoRA + Token Merging（ToMe）的基线构建。在每个 ViT block 的 MHSA 之前执行 token 合并：先按 ToMe 的方式找到待合并的 token 对 $(t_{m_k}, t_{n_k})$，然后在合并（平均池化）之前，通过 PYRA 模块对 $t_{m_k}$ 侧的 token 进行特征调制。PYRA 模块非常轻量：每层仅引入两个可学习向量 $W_r \in \mathbb{R}^{r \times 1}$ 和 $W_D \in \mathbb{R}^{1 \times D}$。LoRA 模块附加在 Q/K/V 投影矩阵上，推理时可合并进骨干不增加额外计算。整个系统端到端训练。

### 关键设计
1. **并行生成解耦权重 (Parallel Yielding Adaptive Weights)**:
    - 功能：为每对待合并 token 生成自适应的调制权重矩阵 $W^l \in \mathbb{R}^{D \times r}$
    - 核心思路：将调制权重矩阵分解为 channel 维度权重 $\delta_D^l \in \mathbb{R}^{D \times 1}$ 和 token 维度权重 $\delta_r^l \in \mathbb{R}^{1 \times r}$ 的外积 $W^l = \delta_D^l \delta_r^l$。先将待合并 token 对求和并做 LayerNorm 得到信息矩阵 $M_\text{info}^l = \text{LayerNorm}(M_s^l + M_t^l) \in \mathbb{R}^{D \times r}$，然后并行计算 $\delta_D^l = M_\text{info}^l W_r^l$（感知 token 分布）和 $\delta_r^l = W_D^l M_\text{info}^l$（感知 channel 分布）
    - 设计动机：直接学习完整矩阵 $W^l$ 既冗余又不具有自适应性（不同图片/不同 token 对应固定权重），解耦为两个方向后参数量极少（$D + r$ 个参数），且通过与 token 信息矩阵的乘法天然获得输入自适应性

2. **Re-Activation token 调制策略**:
    - 功能：将生成的权重稳健地应用于 token 特征调制
    - 核心思路：分两步调制。第一步：将 $\delta_D^l$ 广播后做 sigmoid 激活，与 $M_s^l$ 逐元素相乘得到中间结果 $\hat{M}_s^l = 2\sigma(\hat{\delta}_D^l) \odot M_s^l$。第二步：将 $\delta_r^l$ 广播做 sigmoid 激活后再次调制 $M_s^l \leftarrow M_s^l + (2\sigma(\hat{\delta}_r^l) - 1) \odot \hat{M}_s^l$。残差连接保证了梯度流动，$W_D^l$ 零初始化使训练初始等价于恒等变换
    - 设计动机：（1）sigmoid 将权重约束在合理范围内，避免训练不稳定；（2）两次非线性激活的组合提升了低秩调制矩阵的表达能力；（3）残差+零初始化保证从基线方法平滑过渡

3. **Token Merging 基线选择**:
    - 功能：提供无参数、免训练的推理加速
    - 核心思路：采用 ToMe 的二部图匹配机制，将 token 按余弦相似度配对，选择最相似的 $r$ 对进行平均池化合并。每层减少 $r$ 个 token，逐层叠加实现整体压缩
    - 设计动机：ToMe 不改变模型结构，不引入额外参数，与 PEFT 的存储高效性完全兼容；LoRA 推理时可合并进骨干，两者组合是一个干净且强的基线

### 损失函数 / 训练策略
使用与 LoRA 相同的训练策略，PYRA 模块的生成器和 LoRA 模块一起端到端训练。每层 PYRA 仅增加 $D + r$ 个参数（如 ViT-B 总计约 8.64K）和 $4rD$ FLOPs 额外计算。$W_r$ 用随机高斯初始化，$W_D$ 用零初始化。训练时仅调制 $M_s$（而非 $M_t$），因为 $t_{m_k}$ 是唯一的，而不同 $t_{n_k}$ 可能指向同一目标 token，分开处理保证并行性。

## 实验关键数据

### 主实验
VTAB-1k 基准上低压缩率（~1.7x 加速）性能对比：

| 方法 | 训练参数 | 吞吐量 | ViT-B Avg | ViT-L Avg |
|------|---------|--------|-----------|-----------|
| PEFT (LoRA) | 0.34%/0.39% | 425/130 | 74.76 | 76.52 |
| DiffRate | 0.35%/0.39% | 709/221 | 55.82 | 59.53 |
| ToMe | 0.34%/0.39% | 753/227 | 74.10 | 76.11 |
| **PYRA** | **0.35%/0.40%** | **745/225** | **74.69** | **76.84** |

高压缩率（~3.2x 加速）性能对比：

| 方法 | 训练参数 | 吞吐量 | ViT-B Avg | ViT-L Avg |
|------|---------|--------|-----------|-----------|
| 小模型 PEFT* | 0.34% | 1350/425 | 71.85 | 74.76 |
| ToMe | 0.34%/0.39% | 1381/431 | 70.43 | 74.10 |
| **PYRA** | **0.35%/0.40%** | **1365/427** | **72.06** | **75.66** |

PYRA 在低压缩率下几乎无损（ViT-L 甚至提升 0.32），在高压缩率下消除了 Adverse Compression 现象，超越对应吞吐的小模型。

### 消融实验

| 配置 | Natural | Specialized | Structured | Average | 说明 |
|------|---------|-------------|------------|---------|------|
| Baseline (ToMe+LoRA) | 72.87 | 80.78 | 57.64 | 70.43 | 基线 |
| + 仅 $W_r$ (无激活) | 72.90 | 81.07 | 57.66 | 70.54 | token 维度权重微弱提升 |
| + 仅 $W_r$ (有激活) | 73.18 | 81.69 | 57.65 | 70.84 | 激活函数显著提升 |
| + 仅 $W_D$ (无激活) | 73.09 | 81.13 | 58.43 | 70.88 | channel 维度权重效果更大 |
| + 仅 $W_D$ (有激活) | 73.31 | 82.17 | 58.44 | 71.31 | 激活函数再次提升 |
| + $W_r$ & $W_D$ (无激活) | 73.77 | 81.37 | 58.81 | 71.32 | 并行组合优于单独 |
| **PYRA (全部)** | **73.91** | **82.60** | **59.66** | **72.06** | 所有组件互补 |

### 关键发现
- 并行解耦权重的组合效果优于单独使用任一维度，证明 token 和 channel 两个方向的调制互补
- Sigmoid re-activation 在所有配置下都带来显著提升（+0.3~0.5），是保证训练稳定的关键
- 直接学习完整调制矩阵 $W_{D \times r}$（70.66K 参数，71.49 avg）不如 PYRA（8.64K 参数，72.06 avg），说明自适应生成权重比固定权重更有效
- 与常用的 gated MLP 生成器对比（73.73K 参数，71.06 avg），PYRA 用不到 1/8 的参数取得了更好结果
- 在自监督预训练（MAE）和不同架构（DeiT）上泛化性良好：MAE ViT-L 低压缩率时甚至超越未压缩基线
- 在不同压缩率下 PYRA 一致压平了精度-吞吐量曲线，说明方法对压缩率鲁棒

## 亮点与洞察
- 准确地识别了一个实际痛点：PEFT 只管训练效率不管推理效率，模型压缩只管推理效率不管训练效率，二者简单组合不够
- 设计极其轻量——每层仅增加 $D + r$ 个参数（ViT-B 总计 8.64K），额外 FLOPs 几乎可忽略，体现了"最小侵入"的设计哲学
- 零初始化 + 残差连接的设计保证 PYRA 初始等效于恒等变换，从基线平滑过渡，训练非常稳定
- "Adverse Compression"现象的发现和命名有助于社区理解压缩+微调的失效模式
- 实验覆盖了 4 种骨干 × 2 种压缩率 × 19 个任务，较为全面

## 局限与展望
- 仅验证了图像分类（VTAB-1k），缺乏在检测、分割等密集预测任务上的验证
- Token 调制仅作用于 $M_s$（被合并一侧），$M_t$ 未被调制，潜在信息利用不充分
- 方法与 ToMe 的二部图匹配机制耦合，对其他 token 减少策略（如注意力驱动的 pruning）的适配性未验证
- 高压缩率下与小模型的比较依赖于小模型是否可用，实际场景中小模型的预训练权重通常是存在的
- 仅使用了 LoRA 作为 PEFT 方法，与其他 PEFT（Adapter、Prompt Tuning 等）的兼容性未探索

## 相关工作与启发
- **LoRA**: 低秩自适应微调，本文作为 PEFT 模块的默认选择
- **ToMe**: Token Merging，无参数 token 减少方法，本文作为压缩基线并在其上构建
- **DiffRate**: 搜索每层最优 token 减少率的方法，但需要 ImageNet-21K 搜索且在 PEFT 场景下效果差
- **SSF / AdaptFormer / Consolidator**: 各种 PEFT 方法，本文框架理论上可与它们兼容
- 启发：Token 合并中的信息丢失可以通过极轻量的特征调制有效补偿，为"训练推理双高效"提供了一个可行范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖（Training-Inference Efficient Task Adaptation），解决方案设计精巧但各组件较直觉化
- 实验充分度: ⭐⭐⭐⭐⭐ 4种骨干、2种压缩率、19个任务、详尽的消融和对比，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式规范，图示直观，问题动机阐述好
- 价值: ⭐⭐⭐⭐ 填补了 PEFT+压缩的空白，方法轻量实用，对大模型部署有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[ICLR 2026\] Efficient Message-Passing Transformer for Error Correcting Codes](../../ICLR2026/signal_comm/efficient_message-passing_transformer_for_error_correcting_codes.md)
- [\[CVPR 2025\] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](../../CVPR2025/signal_comm/ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)
- [\[ECCV 2024\] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)

</div>

<!-- RELATED:END -->
