---
title: >-
  [论文解读] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models
description: >-
  [CVPR 2025][LLM安全][视觉取证] 发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "视觉取证"
  - "Token压缩"
  - "最优传输"
  - "深度伪造检测"
  - "推理加速"
---

# ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2603.12208](https://arxiv.org/abs/2603.12208)  
**代码**: [https://github.com/laiyingxin2/ForensicZip](https://github.com/laiyingxin2/ForensicZip)  
**领域**: 多模态VLM  
**关键词**: 视觉取证, Token压缩, 最优传输, 深度伪造检测, 推理加速

## 一句话总结

发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

## 研究背景与动机

**领域现状**：多模态 LLM 越来越多用于可解释的伪造检测——不仅预测真伪，还生成文字解释（异常线索、混合边界、反射不一致等）。但高分辨率图像/视频生成大量视觉 token，prefill 阶段成为计算瓶颈。

**现有痛点**：视觉 token 剪枝方法（FastV、SparseVLM 等）都是基于语义显著性（cross-modal attention、视觉-语言相似度）。但篡改痕迹（高频噪声、混合接缝、时间抖动）往往在语义上"不重要"的区域——背景、物体边缘。

**核心矛盾**：语义显著性与取证证据反相关——语义剪枝像低通滤波器，保留了"好看的"内容但丢弃了"有问题的"痕迹。在高压缩比下，取证性能急剧下降。

**本文目标** 如何在极端压缩（90% token 丢弃）下保留微妙的非语义取证证据？

**切入角度**：生成管道再逼真，也会违反帧间物理连续性。在 token 空间中，这表现为局部纹理/结构的"Birth"（凭空出现）和"Death"（突然消失）。

**核心 idea**：用 Birth-Death 最优传输检测帧间物理不连续性作为取证信号，替代语义显著性作为 token 保留准则。

## 方法详解

### 整体框架

训练无关（training-free）的即插即用框架。在 VLM 推理管道中，vision encoder 之后、LLM 之前，用 ForensicZip 选择保留哪些 token。两个阶段：Transport Novelty Estimation (TNE) 检测时域异常 + Forensic Scoring (FS) 融合空间高频先验 → Top-K 选择。

### 关键设计

1. **Birth-Death 最优传输 (TNE)**：

    - 功能：量化相邻帧之间 token 的物理连续性
    - 核心思路：对帧 $t-1$ 和 $t$ 的 patch token，建立 $(N+1) \times (N+1)$ 代价矩阵，多出的一行一列是 dummy node（slack）。cost 是 cosine distance。用 Sinkhorn 算法求解 entropic OT plan
    - 关键创新：dummy node 让"没有前驱"的 token 可以通过 Birth 节点路由，"没有后继"的通过 Death 节点路由。标准平衡 OT 会强制所有 token 配对，稀释异常信号
    - 提取两个得分：transport cost $e_j^{(t)}$（分布式异常）和 Birth evidence $b_j^{(t)}$（突然出现的异常）

2. **Forensic Scoring (FS)**：

    - 功能：融合时域异常与空域高频先验
    - 核心思路：$s_j^{(t)} = (e_j^{(t)} + \lambda_{birth} b_j^{(t)}) \cdot (1 + \eta_{forensic} U_j^{(t)})$，其中 $U_j^{(t)}$ 是 3×3 Laplacian 响应
    - 乘法形式实现"soft AND gate"：token 必须同时有时域异常 AND 空域高频活动才得高分。加法会让相机平移（大位移但无高频异常）或静态边缘（有高频但无时域异常）获得高分
    - 设计动机：自然运动（相机平移）有高 transport cost 但频谱特征正常；静态伪造有高频异常但无时域异常。乘法过滤掉这两种干扰

3. **Physical Top-K 选择**：

    - 保留比例 ρ 的最高分 token，全局 token 始终保留
    - 序列长度从 T(N+1) 降到 T(K+1)，直接减少 LLM 的 self-attention 计算

### 计算开销

OT 求解开销 $O((T-1) \cdot I_{sk} \cdot (N+1)^2)$，Sinkhorn 20 轮，在 LLM forward 之前一次性完成，相比多层 transformer 节省的计算量可忽略不计。

## 实验关键数据

### 主实验

在 FakeVLM、FakeShield 两个 backbone 上测试，覆盖 deepfake 和 AIGC 多个数据集：

| 方法 | Token 保留率 | Avg 性能 | FLOPs (T) | 延迟 (ms) | 加速比 |
|------|------------|---------|-----------|----------|--------|
| Vanilla (上界) | 100% | ~98.6% | 1.0x | baseline | 1.0x |
| 语义剪枝 (FastV 等) | 10% | 性能崩溃 (红字) | — | — | — |
| **ForensicZip** | **10%** | **~维持 SOTA** | **~0.1x** | — | **2.97x** |

### 消融实验

核心消融验证了：
- 去掉 dummy node（标准 balanced OT）→ 异常信号被稀释，性能下降
- 去掉高频先验 → 相机运动产生大量误报
- 加法替代乘法 → 不能有效过滤干扰
- Birth evidence 和 transport cost 互补：前者捕捉突然出现的异常，后者捕捉渐变异常

### 关键发现

- **语义剪枝在取证任务上灾难性失效**：在 10% 保留率下，语义驱动方法性能崩溃（表中红字），而 ForensicZip 基本不损失
- **取证证据与语义显著性反相关**：定量验证了这一核心发现（cross-modal attention vs forgery mask IoU 负相关）
- **Birth-Death OT 比标准 OT 显著更好**：dummy node 是关键——它将分散的异常信号集中到可解释的 Birth/Death 事件中
- **乘法融合 >> 加法融合**：验证了 soft AND gate 的必要性

## 亮点与洞察

- **"取证-语义反相关"是一个深刻的洞察**：这不只适用于 token 剪枝——任何基于语义显著性的注意力机制在取证任务上都可能有偏。这对整个 forensic VLM 设计有启示。
- **Birth-Death OT 的 slack node 设计**非常巧妙：把"没有前驱"从一个匹配失败转变为一个可量化的信号。类比：标准 OT 像是"所有快递都必须签收"，Birth-Death OT 允许"这个包裹没人寄"——更真实地建模了伪造过程。
- **Training-free 且即插即用**：不需要重训任何模型，直接作为 inference-time 插件，实际部署价值高。

## 局限与展望

- **只在特定 forensic MLLM 上验证**：需要测试更多 backbone 和更多伪造类型
- **静态图像 fallback 较弱**：单张图片只能用空间 outlier 检测，没有时域 OT 信号
- **Laplacian 作为高频先验较粗糙**：更复杂的频域分析（DCT、小波）可能更精准
- **固定保留比例 ρ**：自适应 ρ（根据视频内容决定保留多少 token）可能更优

## 相关工作与启发

- **vs FastV / SparseVLM**：这些纯语义剪枝方法在 VQA/Captioning 上有效，但在 forensic 上灾难性失败。ForensicZip 证明任务特异性的 token 选择至关重要。
- **vs 最优传输在 CV 中的应用**：OT 常用于分布匹配、域适应。这里首次用 unbalanced OT 的 slack 机制检测取证异常——将 OT 从"对齐"工具转为"异常检测"工具。
- **启发**：其他需要关注"非显著区域"的任务（如医学影像中的微小病灶检测）也可能受益于类似的反语义剪枝策略。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Birth-Death OT + 取证-语义反相关洞察，非常原创
- 实验充分度: ⭐⭐⭐⭐ 多个 deepfake/AIGC 基准，详细消融，效率分析到位
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但公式密集
- 价值: ⭐⭐⭐⭐ 对 forensic MLLM 的实际部署有直接意义，training-free 即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[NeurIPS 2025\] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](../../NeurIPS2025/llm_safety/unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)

</div>

<!-- RELATED:END -->
