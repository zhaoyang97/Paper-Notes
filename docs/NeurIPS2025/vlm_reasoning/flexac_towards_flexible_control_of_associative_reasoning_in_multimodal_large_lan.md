---
title: >-
  [论文解读] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models
description: >-
  [NeurIPS 2025][多模态VLM][幻觉控制] FlexAC 发现 MLLM 的联想推理行为主要编码在中间层，通过从幻觉响应中提取引导向量并在推理时注入中间层表示，实现忠实性与创造力的灵活调控——幻觉率降低 29%(CHAIR)，创造力提升 5.8×(Creation-MMBench)，且无需训练。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "幻觉控制"
  - "创造力增强"
  - "联想推理"
  - "中间层干预"
  - "引导向量"
---

# FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.11190](https://arxiv.org/abs/2510.11190)  
**代码**: [github.com/ylhz/FlexAC](https://github.com/ylhz/FlexAC)  
**领域**: 多模态VLM  
**关键词**: 幻觉控制, 创造力增强, 联想推理, 中间层干预, 引导向量

## 一句话总结
FlexAC 发现 MLLM 的联想推理行为主要编码在中间层，通过从幻觉响应中提取引导向量并在推理时注入中间层表示，实现忠实性与创造力的灵活调控——幻觉率降低 29%(CHAIR)，创造力提升 5.8×(Creation-MMBench)，且无需训练。

## 研究背景与动机
**领域现状**：多模态 LLM 面临忠实性(低联想)与创造力(高联想)的内在矛盾——事实性任务需抑制联想，创意任务需增强联想。

**现有痛点**：(1) 幻觉缓解方法（如 VCD 的对比解码、Ha-DPO 的偏好优化）在降低幻觉的同时全面压制联想能力，导致创造力下降(VDAT 降 1.78)；(2) 缺乏可调控机制——要么全抑制，要么不处理。

**核心矛盾**：幻觉和创造力可能共享相同的联想机制，只是在不同任务中表现为"有害"或"有益"，现有方法无法区分两者。

**本文目标** (1) 定位 MLLM 中联想行为的产生层次；(2) 设计可控的联想强度调节机制。

**切入角度**：从认知科学的收敛思维(基于事实的联想)和发散思维(非典型联想)出发，假设幻觉和创造力源自共享的联想机制，可通过中间层表示的方向性干预来调控。

**核心idea**：幻觉响应中蕴含联想方向信息——提取"幻觉 - 真实"的表示差作为引导向量，正向注入增强创造力，反向注入降低幻觉。

## 方法详解

### 整体框架
分两阶段：**Phase I (离线)** 构建通用和任务特定的联想引导向量；**Phase II (推理时)** 在中间层注入引导向量并自适应校准强度。全过程无需重新训练。

### 关键设计

1. **中间层联想行为分析与定位**:

    - 功能：确定联想行为在模型哪些层产生
    - 核心思路：收集 1000 张 COCO 图像的真实响应 $f^{(n)}$ 和诱导幻觉响应 $f^{(a)}$，逐层计算余弦距离 $\mathcal{D}_\text{cos}$ 和欧氏距离 $\mathcal{D}_\text{Euc}$。进一步做层干预实验——在第 $m$ 层将联想特征替换为非联想特征 $f_m^{\text{modified}} = f_m^{(n)}$，观察对后续层的影响
    - 关键发现：(1) 浅层(0-9)距离低→共享低级感知；(2) 余弦距离在中间层(10-15)达到峰值→联想方向在此形成；(3) 替换中间层特征后下游差异显著减少→中间层是联想行为的源头而非传播层

2. **幻觉引导的联想引导向量构建 (Phase I)**:

    - 功能：从幻觉响应中提取可用于调控联想的方向向量
    - 核心思路：对每个样本在中间层 $l$ 计算方向差 $v_l = f_l^{(a)} - f_l^{(n)}$，选择余弦距离最大的 Top-K 个样本对取平均得到通用引导向量：
    $\mathcal{I} = \text{Top-K}(\mathcal{D}_\text{cos}(f_{l,i}^{(a)}, f_{l,i}^{(n)})); \quad v_l = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} (f_{l,i}^{(a)} - f_{l,i}^{(n)})$
    - 设计动机：Top-K 选择降低噪声；实验中从 2000 张图随机选 50 张即可构建有效向量

3. **任务特定联想向量 (Directional Integration)**:

    - 功能：为故事创作、隐喻等不同创意任务构建专属引导方向
    - 核心思路：用 GPT-4o 为目标任务生成高联想输出样本，提取其中间层特征与基础模型输出的差异作为任务特定向量 $v_l^{\text{task}}$，推理时与通用向量组合：$f_l^{\text{control}} = f_l + \alpha_\text{gen} \cdot v_l^{\text{gen}} + \alpha_\text{task} \cdot v_l^{\text{task}}$
    - 设计动机：联想推理是多维度的（事件规划 vs 文学创作需要不同的联想方向），单一向量不够

4. **自适应引导强度校准 (SIC)**:

    - 功能：防止过度引导导致语义漂移
    - 核心思路：根据当前表示与引导方向的对齐度自适应调整 $\alpha$：
    $\alpha = \text{sigmoid}\left(\max\left(-\frac{f_l \cdot v_l}{\|f_l\|\|v_l\|}, 0\right)\right)$
      当前表示已与联想方向对齐时 $\alpha$ 小（抑制过度引导），未对齐时 $\alpha$ 大（加强引导）。引导后标准化保持特征尺度：$f_l^{\text{control}} \leftarrow f_l^{\text{control}} \cdot \frac{\|f_l\|}{\|f_l^{\text{control}}\|}$
    - 设计动机：均匀施加引导向量对已具有强联想倾向的输入会造成过度偏移

### 训练策略
完全免训练(training-free)，仅需离线构建引导向量（50 张图 + GPT-4o 生成样本），推理时注入中间层。

## 实验关键数据

### 主实验：幻觉基准

| 模型 | 方法 | CHAIR_S↓ | CHAIR_I↓ | POPE F1↑ |
|------|------|----------|----------|----------|
| Qwen-VL | Regular | 40.6 | 12.5 | 85.6 |
| Qwen-VL | VCD | 42.0 | 11.2 | 86.3 |
| Qwen-VL | **FlexAC** | **19.2** | **5.4** | **87.1** |
| LLaVA-1.5 | Regular | 50.8 | 14.3 | 86.5 |
| LLaVA-1.5 | Ha-DPO | 36.8 | 10.4 | 83.9 |
| LLaVA-1.5 | **FlexAC** | **36.6** | **10.4** | **87.9** |
| DeepSeek-VL2 | Regular | 32.6 | 9.2 | 88.5 |
| DeepSeek-VL2 | **FlexAC** | **28.6** | **8.1** | **88.6** |

### 创造力基准

| 方法 | VDAT (Qwen) | VDAT (LLaVA) | Creation-MMBench Reward |
|------|------------|-------------|------------------------|
| Regular | 84.85 | 86.89 | 0.00 |
| Ha-DPO | — | 85.11↓ | — |
| VCD | 83.69↓ | 86.83 | -3.86↓ |
| **FlexAC** | **86.58**↑ | **88.49**↑ | **10.92**↑ |

FlexAC 是唯一同时提升忠实性和创造力的方法；其他方法在降低幻觉时创造力下降，或两者都不显著改善。

### 消融实验

| 配置 | CHAIR_S↓ | VDAT↑ |
|------|----------|-------|
| FlexAC-P (完整, α=-1) | **19.2** | — |
| FlexAC-C (完整, α=1) | — | **86.58** |
| FlexAC - IS - SIC | 30.4 | 85.05 |
| FlexAC - DI | ~20 | 85.8 |
| Regular | 40.6 | 84.85 |

### 关键发现
- 中间层(Qwen 15-17, LLaVA 11-13, DeepSeek 4-6)是最佳干预点，浅层和深层干预效果微弱
- Instance Selection 和 SIC 对幻觉缓解贡献最大（去掉后 CHAIR 从 19.2 升至 30.4）
- Directional Integration 对创造力提升关键，体现了联想推理的多维度性
- FlexAC 在通用基准(MME/MMMU/MMStar)上不降反升，特别是 OCR 任务因增强了文字-视觉联想而提升

## 亮点与洞察
- **幻觉=联想的统一视角**：将"有害幻觉"和"有益创造力"统一为联想强度的连续谱，用同一机制双向调控，思路新颖且优雅
- **层干预分析方法论**：特征替换实验精准定位联想产生层（而非传播层），这一分析模式可推广到其他模型行为的解剖
- **SIC 的自适应设计**：简单的余弦角度阈值实现样本级的动态强度控制，避免过度引导，实际效果验证了其必要性

## 局限与展望
- 需要白盒访问模型中间层，不适用于 ChatGPT 等黑盒 API 模型
- 引导向量基于 COCO 数据集构建，在领域差异大的场景(如医学)是否需要重新构建有待验证
- VDAT 指标基于 CLIP 嵌入的语义距离，可能不完全反映人类对创造力的感知
- 仅在 7B 级别模型上验证，更大规模模型的中间层动态可能不同

## 相关工作与启发
- **vs VCD (对比解码)**：VCD 在解码层面对比清晰/模糊输入，FlexAC 在表示层面直接操控联想方向，更精准且可双向调控
- **vs Ha-DPO (偏好优化)**：Ha-DPO 需要额外训练且不可逆地抑制联想，FlexAC 免训练且可灵活切换
- **vs CAA (对比激活添加)**：FlexAC 可视为 CAA 在多模态场景的扩展，增加了 SIC 自适应校准和任务特定方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 幻觉-创造力统一视角是全新洞察，免训练双向调控框架实用性强
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖幻觉/创造力/通用 3 类 7 个基准 × 3 个模型，消融全面
- 写作质量: ⭐⭐⭐⭐ 分析-发现-方法的逻辑链清晰，图文并茂
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——免训练、即插即用、效果显著，对 MLLM 部署有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)
- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](../../ACL2025/multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[NeurIPS 2025\] Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)

</div>

<!-- RELATED:END -->
