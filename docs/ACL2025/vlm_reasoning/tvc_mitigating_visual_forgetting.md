---
title: >-
  [论文解读] Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning
description: >-
  [ACL 2025][多模态VLM][视觉遗忘] 发现 MLLM 在长链 CoT 推理中存在严重的视觉遗忘现象——推理过半后移除图像仅导致 ~2% 的准确率下降，表明模型过度依赖自生成文本而忽视视觉证据。提出 TVC (Take-along Visual Conditioning) 策略，在训练阶段通过动态视觉重确认 (DVR) 注入图像回顾机制，推理阶段通过周期性视觉校准 (PVC) 压缩并重注入视觉 token，在 5 个数学推理基准上平均超越 SOTA 3.4 分（43.4 vs 40.0）。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉遗忘"
  - "Take-along Visual Conditioning"
  - "长链 CoT 推理"
  - "动态视觉重确认"
  - "周期性视觉校准"
---

# Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning

**会议**: ACL 2025  
**arXiv**: [2503.13360](https://arxiv.org/abs/2503.13360)  
**代码**: [TVC 项目页](https://sun-hailong.github.io/projects/TVC)  
**领域**: 多模态推理  
**关键词**: 视觉遗忘, Take-along Visual Conditioning, 长链 CoT 推理, 动态视觉重确认, 周期性视觉校准

## 一句话总结

发现 MLLM 在长链 CoT 推理中存在严重的视觉遗忘现象——推理过半后移除图像仅导致 ~2% 的准确率下降，表明模型过度依赖自生成文本而忽视视觉证据。提出 TVC (Take-along Visual Conditioning) 策略，在训练阶段通过动态视觉重确认 (DVR) 注入图像回顾机制，推理阶段通过周期性视觉校准 (PVC) 压缩并重注入视觉 token，在 5 个数学推理基准上平均超越 SOTA 3.4 分（43.4 vs 40.0）。

## 研究背景与动机

**领域现状**：LLM 推理能力已从 CoT 提示发展到产品级方案（o1/DeepSeek-R1），多模态推理也通过数据驱动方法取得进展（Math-LLaVA/MAmmoTH-VL）。

**现有痛点**：(1) 文本 LLM 可通过重复关键术语保持问题上下文，但 MLLM 中视觉输入被限制在初始处理阶段，后续推理步骤无法回看图像；(2) 随着推理链增长，模型对视觉输入的注意力呈指数衰减，导致过度依赖自生成文本（"视觉遗忘"）；(3) 这种衰减引发幻觉和空间关系验证失败。

**核心矛盾**：当前 MLLM 架构中，视觉信息仅在输入时注入一次，而长链推理需要持续的视觉-文本交互。类似人类解题时反复查看图形，模型也需要在推理过程中重新关注视觉输入。

**本文目标** 诊断并缓解 MLLM 长链推理中的视觉遗忘现象。

**切入角度**：通过"渐进式图像移除"实验量化视觉遗忘程度，再设计训练+推理双阶段的视觉再注入方案。

**核心 idea**：在推理链的关键阶段重新注入压缩后的视觉 token，模拟人类"回看图像"行为来保持视觉注意力。

## 方法详解

### 整体框架

TVC 包含训练和推理两个阶段：训练阶段使用迭代蒸馏流水线（QVQ-72B→Qwen2-VL）构建长链推理数据集，并通过 DVR 在训练数据的自我反思点注入视觉 caption 重激活；推理阶段通过 PVC 在推理中途压缩视觉 token（4×4 平均池化）并重置 KV cache 再注入。

### 关键设计

1. **视觉遗忘的量化诊断**
    - **功能**：通过渐进式图像移除实验定量揭示视觉遗忘程度
    - **核心思路**：将推理过程等分为 $K=8$ 个阶段，在不同位置 $k$ 重置 KV cache 以移除图像 token，比较正常推理与移除图像后的准确率差异
    - **关键发现**：(a) 在 $k=4$（推理过半）处移除图像仅降低 2.2% 准确率（40.9 vs 43.1）；(b) 遗忘效应近似指数衰减 $\Delta_{\text{visual}}(k) \propto e^{-k}$；(c) 早期移除（$k=0$）导致 ~20% 下降，说明早期阶段确实利用了视觉信息；(d) 注意力矩阵可视化证实：约 20% token 后图像注意力显著减弱
    - **设计动机**：在提出解决方案前需要定量理解问题的严重程度和模式

2. **Dynamic Visual Reaffirmation (DVR) — 训练阶段**
    - **功能**：在训练数据中注入视觉重激活点，教模型学会回看图像
    - **核心思路**：(a) 使用 QVQ-72B 作为教师模型蒸馏长链 CoT 数据，通过双温度采样（$\tau=0$ 初始采样 + $\tau=1$ 错误纠正，best-of-8）获得 ~200K 高质量样本；(b) 在自我反思间隔（如推理中点 $r_1=0.5L$）手动注入桥接提示（"Let me see the image again"）和视觉 caption 重生成；(c) 训练时同时微调 LLM 参数和跨模态连接器，冻结视觉编码器
    - **数据质量保证**：动态 token 截断（200-8000 token）过滤过长/过短推理链；反思词修剪（上限 25 个反思标记）减少无效元认知循环
    - **设计动机**：QVQ 模型本身缺乏在推理中迭代引用视觉输入的能力，需在训练数据中显式植入这一机制

3. **Periodic Visual Calibration (PVC) — 推理阶段**
    - **功能**：在推理过程中定期重新注入压缩后的视觉 token
    - **核心思路**：(a) Token 压缩——使用 4×4 平均池化将视觉 token 数量压缩 16 倍，保留空间语义；(b) Visual Cache Reset——在自我反思间隔前置桥接提示指令，重置 KV cache 并重注入压缩后的图像 token
    - **设计动机**：压缩是必要的——过多视觉 token 会导致模型遗忘之前的文本推理步骤；KV cache 重置确保新注入的视觉信息能有效参与后续注意力计算

## 实验关键数据

### 主实验——与 SOTA 方法对比（5 个数学推理基准）

| 方法 | 尺寸 | MathVista | MathVision | MathVerse | Dynamath | OlympiadBench | 平均 |
|------|:----:|:---------:|:----------:|:---------:|:--------:|:-------------:|:----:|
| Qwen2-VL | 7B | 60.9 | 16.3 | 24.6 | 11.0 | 3.2 | 23.2 |
| InternVL2.5 | 8B | 64.5 | 17.0 | 22.8 | 9.4 | 0.1 | 22.8 |
| LLaVA-COT | 11B | 52.5 | 19.9 | 22.6 | 7.8 | - | - |
| QVQ-72B-preview | 72B | 71.4 | 35.9 | 41.5 | 30.7 | 20.4 | 40.0 |
| **TVC** | **7B** | **68.1** | **22.7** | **38.9** | **15.1** | **9.8** | **30.9** |
| **TVC** | **72B** | **72.2** | **41.9** | **48.8** | **30.0** | **24.3** | **43.4** |

### 消融实验（Qwen2-VL-7B 基准）

| 配置 | MathVista | MathVision | MathVerse | 平均 |
|------|:---------:|:----------:|:---------:|:----:|
| Baseline | 60.9 | 16.3 | 24.6 | 33.9 |
| Vanilla SFT | 63.5 | 19.8 | 31.6 | 38.3 |
| TVC w/o PVC | 66.7 | 21.8 | 35.6 | 41.4 |
| TVC w/o DVR | 66.2 | 22.3 | 34.7 | 41.0 |
| **TVC Full** | **68.1** | **22.7** | **38.9** | **43.2** |

### 关键发现

- TVC-72B 平均超越 QVQ-72B-preview（教师模型）3.4 分（43.4 vs 40.0），学生超越教师
- MathVision 和 MathVerse 上提升最大（+6.0 和 +7.3），这些基准要求持续的视觉推理
- TVC-7B 在 MathVerse 上（38.9）甚至超过多个 72B 模型
- DVR 和 PVC 贡献相当且互补：两者单独移除各降 ~2 分，完整组合增益更大
- 4×4 平均池化压缩视觉 token 不仅提升推理效率，还略微改善性能（43.2 vs 43.1 无压缩）
- 数据量扩展（50K→200K）持续带来提升，未出现饱和

## 亮点与洞察

1. **视觉遗忘现象的严谨诊断**：渐进式图像移除实验和注意力热图可视化双重验证，直观且令人信服地展示了问题的严重性（去掉图像才降 2%！）
2. **"回看图像"类比人类行为**：TVC 的核心直觉与人类解几何题时反复查看图形的行为完全对应，简洁而有效
3. **训练+推理一体化设计**：DVR 教会模型何时/如何回看，PVC 在推理时实际执行回看，两者缺一不可
4. **数据工程的多层质量控制**：双温度采样 + answer-centric reject sampling + 动态截断 + 反思词修剪，构成了完整的长链推理数据质量保证流水线

## 局限与展望

1. 对于高度复杂的推理任务，简单增加视觉回看次数不足，需增强模型本身的推理能力
2. 方法假设可延迟视觉处理，不适用于实时应用（如机器人导航）
3. 视觉重确认位置标定（中点 $r_1=0.5L$）较为启发式，未探索自适应触发机制
4. 仅在数学推理基准验证，未测试在图表理解、文档分析等其他视觉推理场景的效果
5. 训练需要 64×H20 GPU 跑 4 天（72B 模型），计算成本较高

## 相关工作与启发

- 视觉遗忘现象可能广泛存在于所有需要长上下文推理的多模态场景（如长视频理解、多页文档分析）
- PVC 的"压缩+重注入"思路可推广到其他长序列注意力优化场景
- 与 FastV（基于注意力权重修剪冗余视觉 token）不同，TVC 针对的是推理过程中的动态视觉保持
- 蒸馏流水线的质量控制方法（双温度+best-of-N+动态截断）对构建其他推理数据集有参考价值

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：视觉遗忘现象的定量诊断新颖且重要，TVC 方案直觉清晰
- **实验充分度** ⭐⭐⭐⭐：5 个基准、多尺度模型（7B/72B）、消融和数据扩展实验全面
- **写作质量** ⭐⭐⭐⭐：问题诊断到方案设计逻辑链清晰，可视化质量高
- **价值** ⭐⭐⭐⭐：视觉遗忘是多模态推理的重要瓶颈，TVC 提供了有效的工程化解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](../../CVPR2026/multimodal_vlm/narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)
- [\[ACL 2025\] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)

</div>

<!-- RELATED:END -->
