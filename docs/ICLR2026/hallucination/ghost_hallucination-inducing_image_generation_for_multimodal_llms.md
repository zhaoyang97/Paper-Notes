---
title: >-
  [论文解读] GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs
description: >-
  [ICLR 2026][幻觉检测][object hallucination] GHOST 不再用固定的静态 benchmark 评测多模态大模型（MLLM）的物体幻觉，而是**主动生成**一批"看起来自然、人眼看不出有目标物、却能让模型坚信物体存在"的图片，把幻觉成功率从已有方法的约 1% 拉到 28% 以上，并发现这些图片在不同模型间高度可迁移。
tags:
  - "ICLR 2026"
  - "幻觉检测"
  - "object hallucination"
  - "MLLM"
  - "stress-test"
  - "CLIP embedding optimization"
  - "扩散模型"
  - "transferability"
---

# GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=f4TACE7HhU](https://openreview.net/forum?id=f4TACE7HhU)  
**代码**: [https://github.com/sudoparsa/GHOST](https://github.com/sudoparsa/GHOST)  
**领域**: 多模态大模型 / 物体幻觉 / 对抗鲁棒性  
**关键词**: object hallucination, MLLM, stress-test, CLIP embedding optimization, diffusion, transferability  

## 一句话总结
GHOST 不再用固定的静态 benchmark 评测多模态大模型（MLLM）的物体幻觉，而是**主动生成**一批"看起来自然、人眼看不出有目标物、却能让模型坚信物体存在"的图片，把幻觉成功率从已有方法的约 1% 拉到 28% 以上，并发现这些图片在不同模型间高度可迁移。

## 研究背景与动机
**领域现状**：多模态大模型在图像描述、VQA、多模态推理上表现亮眼，但普遍存在物体幻觉（object hallucination）——把图里根本不存在的物体说成"存在"。在安全敏感场景里这是致命缺陷，业界一直想系统性地压测模型的视觉鲁棒性。

**现有痛点**：现有的幻觉评测几乎都依赖**静态 benchmark、固定视觉场景、人工策划的图集**（POPE、CHAIR 等）。这类方法只能在一个被框死的场景集合里测，既暴露不了**模型特有**的盲点，也说不清"到底哪类图会触发幻觉"——无法判断这些错误是孤立个例，还是结构性的失败模式。

**核心矛盾**：要发现模型特有的、未被预料到的幻觉漏洞，就必须让评测**能根据目标模型的反馈主动造图**；但把 MLLM、扩散模型、目标图、物体检测器全塞进一个优化回路（如最接近的工作 DASH），又会让 pipeline 慢且贵，被迫只能用蒸馏的单步扩散模型。

**本文目标**：做一个全自动、无需人工监督和先验知识的工具，给定一张不含目标物的图和一个目标物体，生成一张**视觉上接近原图、目标物仍然缺席、但能诱导模型幻觉**的自然图片。

**核心 idea**：**「在 CLIP 嵌入空间里优化，把优化和生成解耦」**——不在像素或扩散 latent 上硬怼，而是优化图像的 CLIP embedding，使它既诱导模型答"Yes"、又不真的编码物体本身，再用一个 unCLIP 扩散模型把这个 embedding 解码成自然图片；关键是用一个轻量 mapper 把 MLLM 的视觉空间和扩散模型的视觉空间对齐，从而避免对整条 pipeline 反向传播。

## 方法详解

### 整体框架
GHOST（Generating Hallucinations via Optimizing Stealth Tokens）把"造幻觉图"拆成三步：先训练一个 mapper Π，把 CLIP 嵌入空间映射到目标 MLLM 的视觉 token 空间，使得优化时**只需在 CLIP embedding 上做梯度下降**、再经 Π 喂给 MLLM 拿反馈，而无需穿过整个扩散+MLLM 管线；然后在 CLIP embedding 上联合优化三个目标（贴近原图、不含物体语义、诱导模型答 Yes）；最后用 unCLIP 扩散模型把优化后的 embedding 解码成自然图，并用开放词表检测器 OWLv2 过滤掉真的含物体的样本。

```mermaid
flowchart LR
    A[原图 Xv<br/>不含目标物 t] --> B[CLIP 编码 c0]
    B --> C[优化 CLIP embedding c<br/>L_total = L_adv + λ_clip·L_clip + λ_reg·L_reg]
    C -->|经 mapper Π| D[MLLM 反馈<br/>p Yes ≥ τ?]
    D -->|未达阈值| C
    D -->|达标| E[unCLIP 扩散<br/>从部分加噪的原图 latent 去噪]
    E --> F[OWLv2 检测<br/>确认不含物体]
    F -->|含物体则丢弃| E
    F -->|确认缺席| G[幻觉诱导图<br/>模型答 Yes 但物体真不存在]
```

### 关键设计

**1. 桥接两套视觉空间的 mapper Π：把优化从生成里解耦出来。** MLLM 用的视觉编码器（如 Qwen 的 ViT）和扩散模型用的 CLIP 编码器并不一样，要把 MLLM 的反馈引进生成过程，最朴素的办法是对整条管线（MLLM→生成图→扩散）反向传播，代价极高。GHOST 改成训练一个简单的 MLP 作为 mapper $\Pi: \mathbb{R}^{d_{CLIP}} \to \mathbb{R}^{N \times d_M}$，用 MSE 把 CLIP embedding 对齐到 MLLM 的视觉 token：$\mathcal{L}_{align} = \|\Pi(V_{CLIP}(X_v)) - V_M(X_v)\|_2^2$。一旦 Π 训好，优化阶段就只在 CLIP embedding 上动，通过 Π 直接预测 MLLM 会看到什么、再算幻觉损失，整条扩散管线不进优化回路——这正是 GHOST 比 DASH 快约 5×（约 10 秒/张，单卡 A100）的根源，也让它能兼容不同的扩散模型和 MLLM。

**2. 三项联合优化目标：在"诱导幻觉"和"物体真不存在"之间走钢丝。** 攻击的本质要求 embedding $c$ 同时满足三件互相拉扯的事，写成 $\mathcal{L}_{total} = \mathcal{L}_{adv} + \lambda_{clip}\mathcal{L}_{clip} + \lambda_{reg}\mathcal{L}_{reg}$。其中诱导项 $\mathcal{L}_{adv} = -\log p(y^\star \mid X_q, \Pi(c))$ 直接最大化模型对"Yes"这个 token 的概率（query 形如"Do you see a t in the image?"，并从一组语义等价模板里随机采样以防过拟合某种问法）；物体缺席项 $\mathcal{L}_{clip} = \mathbb{E}_{T_q \sim T_{clip}}[\cos(c, V_{CLIP}(T_q))]$ 惩罚 $c$ 与"a photo of a t"等文本模板的余弦相似度，逼它别真的把物体语义编码进去；贴近原图项 $\mathcal{L}_{reg} = \|c - c_0\|_2^2$ 防止 embedding 漂移过远、保住高层语义。三者合力让生成的图只引入**微妙的上下文误导线索**（如把香蕉的梗改得像刀刃边缘），而非直接画出物体。

**3. 阈值触发 + 部分加噪的引导扩散：把"模型相信"翻译成"自然的图"。** 优化每一步后检查 $p(y^\star \mid X_q, \Pi(c)) \geq \tau_{yes}$ 是否达标，最多优化 $M$ 步，达不到就丢弃换下一张。达标后做引导扩散时**不从纯噪声起步**，而是把原图编码进 VAE latent、前向加噪 $t$ 步得到 $z_t$，再在 $c$ 条件下反向去噪——噪声等级 $t$ 控制"保留原图结构"与"留出空间塞误导线索"的权衡。生成允许最多 4 次尝试以对冲扩散的随机性，最后用 OWLv2 把真含物体的样本保守地剔除：只有"模型幻觉了物体 + OWLv2 确认物体确实不在"才算一次成功，这保证了得到的图既骗过模型又对人眼诚实。

**4. 适配推理型 MLLM：优化 think 段而非最终答案。** 对 GLM-4.1V-Thinking 这类先输出 `<think>...</think>` 再给 `<answer>...</answer>` 的推理模型，GHOST 用 `<think>` 后第一个解码步上"Yes"的概率作为优化信号，保持目标和运行时与其它模型一致。虽然没直接优化最终答案，却足以把模型的**推理轨迹**整个带偏去为不存在的物体找理由——也因为优化的是 thinking token 而非答案，GLM 的图 FID 偏高、视觉漂移不如其它模型一致。

## 实验关键数据

### 主实验表格（COCO，10 个目标类，与 DASH 对比）

| 方法 | 模型 | 输入样本数 | 幻觉数 | 成功率 |
|------|------|-----------|--------|--------|
| GHOST | Qwen2.5-VL | 9423 | 2816 | **29.9%** |
| GHOST | LLaVA-v1.6 | 8786 | 2468 | **28.1%** |
| GHOST | GLM-4.1V | 8889 | 2880 | **32.4%** |
| DASH-LLM | Qwen2.5-VL | 118,000 | 57 | 0.1% |
| DASH-OPT | Qwen2.5-VL | 118,000 | 42 | 0.1% |

GHOST 在小得多的图池上，发现的幻觉样本数比 DASH 高出数个量级（Qwen 上 2816 vs DASH 合计 99）。

### 图像质量（FID，越低越好）

| 方法 | Qwen2.5-VL | LLaVA-v1.6 | GLM4.1V |
|------|-----------|-----------|---------|
| **分布真实性**（vs COCO val） | | | |
| SD v2.1 | 46.19 | 48.42 | 44.79 |
| SD unCLIP | 46.51 | 50.20 | 44.76 |
| GHOST | 47.03 | 50.78 | 51.70 |
| **语义保真**（vs 原图） | | | |
| SD v2.1 | 41.71 | 42.64 | 39.85 |
| SD unCLIP | 31.67 | 35.47 | 32.07 |
| GHOST | **25.00** | **26.39** | 34.94 |

真实性与基线相当，语义保真显著更好（保住了原图身份）。

### 迁移性（行=源模型，列=目标模型成功率 %）

| 源\目标 | Qwen2.5-VL | LLaVA-v1.6 | GLM4.1V | GPT-4o | Aya | LLaMA3.2 | Gemini |
|---------|-----------|-----------|---------|--------|-----|----------|--------|
| Qwen2.5-VL | – | 62.2 | 72.0 | **66.5** | 71.1 | 65.8 | 58.6 |
| LLaVA-v1.6 | 52.6 | – | 50.5 | 50.5 | 54.4 | 49.7 | 42.8 |
| GLM4.1V | 63.2 | 57.1 | – | 63.8 | 67.6 | 69.1 | 53.8 |

为 Qwen2.5-VL 优化的图，在闭源 GPT-4o 上诱发 66.5% 幻觉——指向跨模型的共享失败模式。

### 缓解实验（Qwen2.5-VL + LoRA，用 GHOST 图微调）

| | POPE ↑ | CHAIRs ↓ | CHAIRi ↓ | VQAv2 ↑ | Caption ↑ |
|------|--------|----------|----------|---------|-----------|
| Baseline | 88.7 | 3.8 | 3 | 89.5 | 72.8 |
| Finetuned | **93.2** | **2.9** | **2.6** | 89.4 | 71.5 |

### 关键发现
- **阈值 τ 越大幻觉越强**：高 τ 让优化更难达标（合格图变少），但合格图诱导幻觉的概率更高，证明 $p(y^\star)$ 是模型"相信物体存在"的有效代理。
- **λclip 越大物体越不易出现**：更强的 CLIP 相似度惩罚有效压制扩散模型把物体真画出来。
- **人类评测确认对人眼诚实**：40 名评审 3000+ 票，LLaVA 图 89%、Qwen 图 86.3% 认为目标物不存在，自然度与扩散基线相当。
- **微调缓解不伤通用能力**：POPE/CHAIR 显著改善，VQAv2、Caption 几乎不变。

## 亮点与洞察
- **从"静态评测"转向"主动造图压测"**：把幻觉评估从被动的固定图集，变成能针对目标模型反馈主动搜索盲点的攻击式诊断，理念上是范式转变。
- **解耦设计是性能与通用性的核心**：mapper Π 让优化彻底脱离扩散管线，既快又能在任意 MLLM × 扩散模型组合上复用，比 DASH 把所有组件塞进回路的做法优雅得多。
- **诊断+矫正一体**：同一批图既能暴露漏洞，又能作为微调数据修复幻觉且不损通用能力，工具闭环。
- **迁移性揭示系统性漏洞**：跨模型甚至跨闭源模型的高迁移率，说明这些不是单模型 bug，而是 MLLM 共享的伪相关/捷径偏置。

## 局限与展望
- **依赖 CLIP/unCLIP 与 OWLv2**：整条管线绑定 CLIP 嵌入空间和特定开放词表检测器，OWLv2 漏检会让"物体缺席"的保证打折。
- **推理模型上效果打折**：优化 thinking token 而非最终答案，使 GLM 的视觉漂移不一致、FID 偏高，对推理型 MLLM 的攻击仍不够精准。
- **缓解仅 toy setup**：微调只在 Qwen2.5-VL + LoRA 的小规模上验证，是否能规模化、是否对更广幻觉类型有效仍待考。
- **目标类与数据集有限**：只在 COCO 的 10 个类（外加 ObjectNet 附录验证）上测，目标物只覆盖"Yes/No 存在性"这一种幻觉形式。

## 相关工作与启发
- **vs DASH（最接近工作）**：DASH 直接在扩散 latent 上优化、把 MLLM+扩散+检测器全塞进回路、最后还要从真实数据集检索相似图；GHOST 用解耦+mapper 在 CLIP 空间优化，更快更省、且生成而非检索。
- **vs 无反馈的造图法**（Wu/Zhang 2024）：那些方法用精心设计的 prompt 让文生图模型造图，但不吸收 MLLM 反馈，抓不到模型特有盲点；GHOST 靠模型反馈做定向优化。
- **vs 像素级对抗攻击**（AnyAttack、AttackVLM）：后者追求像素级不可感知扰动、保持语义；GHOST 反其道——插入人眼可见但合理的**语义级误导线索**，目标是诱发幻觉而非误分类。
- **扩散表示更鲁棒的启发**：受 DEEM、Li et al. 2024 等"扩散模型学的是数据分布、比判别式模型更少走捷径"启发，用扩散模型来探测 MLLM 的脆弱性。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把幻觉评测从静态 benchmark 翻转成主动造图压测，CLIP 空间优化 + mapper 解耦的设计巧妙且有效。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 3 个开源模型 + 4 个迁移模型（含 GPT-4o/Gemini）、FID/人评/迁移/消融/缓解齐全；缓解实验偏 toy、目标幻觉形式单一。
- **写作质量**: ⭐⭐⭐⭐ 问题设定、三项损失、pipeline 叙述清晰，图示丰富。
- **价值**: ⭐⭐⭐⭐⭐ 既是诊断工具又是矫正数据源，迁移性揭示系统性漏洞，对构建可靠多模态系统有直接的工程与安全价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](../../ACL2025/hallucination/automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ICLR 2026\] Leveraging Pretrained Knowledge at Inference Time: LoRA-Gated Contrastive Decoding for Multilingual Factual Language Generation in Adapted LLMs](leveraging_pretrained_knowledge_at_inference_time_lora-gated_contrastive_decodin.md)
- [\[ICLR 2026\] Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs](cat-po_cross-modal_adaptive_token-rewards_for_preference_optimization_in_truthfu.md)
- [\[ICLR 2026\] HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs](halluguard_demystifying_data-driven_and_reasoning-driven_hallucinations_in_llms.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
