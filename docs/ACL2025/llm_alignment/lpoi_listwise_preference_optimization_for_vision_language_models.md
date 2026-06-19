---
title: >-
  [论文解读] LPOI: Listwise Preference Optimization for Vision Language Models
description: >-
  [ACL 2025 (Main)][LLM对齐][列表级偏好优化] 本文提出 LPOI，首个面向 VLM 的目标感知列表级偏好优化方法——通过识别并遮挡图像中的关键目标，在正样本和负样本之间插值生成渐进式遮挡序列，训练模型按目标可见度排序，从而在无需额外标注的情况下有效降低幻觉，在 MMHalBench、AMBER 和 Object HalBench 上超越现有偏好优化方法。
tags:
  - "ACL 2025 (Main)"
  - "LLM对齐"
  - "列表级偏好优化"
  - "视觉语言模型"
  - "幻觉缓解"
  - "目标遮挡"
  - "图像插值"
---

# LPOI: Listwise Preference Optimization for Vision Language Models

**会议**: ACL 2025 (Main)  
**arXiv**: [2505.21061](https://arxiv.org/abs/2505.21061)  
**代码**: [GitHub](https://github.com/fatemehpesaran310/lpoi)  
**领域**: 对齐RLHF / 多模态VLM  
**关键词**: 列表级偏好优化, 视觉语言模型, 幻觉缓解, 目标遮挡, 图像插值

## 一句话总结

本文提出 LPOI，首个面向 VLM 的目标感知列表级偏好优化方法——通过识别并遮挡图像中的关键目标，在正样本和负样本之间插值生成渐进式遮挡序列，训练模型按目标可见度排序，从而在无需额外标注的情况下有效降低幻觉，在 MMHalBench、AMBER 和 Object HalBench 上超越现有偏好优化方法。

## 研究背景与动机

**领域现状**：将 VLM 与人类偏好对齐是一个关键挑战。DPO 等方法在文本域取得成功后被移植到多模态场景，但简单替换文本偏好数据为多模态数据往往效果不佳甚至加剧幻觉。mDPO 等方法通过随机裁剪生成负样本图像来部分缓解此问题。

**现有痛点**：(1) VLM 在偏好学习中容易过拟合文本信息而忽略图像信息，导致目标幻觉；(2) 现有图像负样本生成方法（随机裁剪或扩散模型编辑）要么语义信息损失不可控，要么计算成本极高；(3) 列表级排序在文本域已被证明优于成对比较，但在图像域由于构造列表级样本的困难而无人探索。

**核心矛盾**：成对偏好数据只能捕获"好vs坏"的二元信号，无法让模型学到"好→较好→一般→差"的细粒度区分——而这种渐进式的理解对减少幻觉至关重要（模型需要学会目标"部分可见"和"完全可见"之间的差异）。

**本文目标**：设计一种自动构造列表级图像偏好样本的方法，使 VLM 能通过列表级排序学习来更精细地减少幻觉。

**切入角度**：作者观察到幻觉本质是模型描述了图像中不存在的目标——如果能让模型学会"目标越可见，越应该被提及"这种渐进关系，就能有效减少幻觉。

**核心 idea**：遮挡图像中的关键目标并通过插值遮挡比例自动生成按目标可见度排列的图像列表，用列表级偏好损失优化模型。

## 方法详解

### 整体框架

输入为图像-问题-回答三元组的偏好数据集（包含 chosen 和 rejected）。LPOI 在标准 DPO 损失基础上增加列表级损失：(1) 用目标检测模型找到图像中的关键目标；(2) 遮挡该目标并验证遮挡后确实产生幻觉；(3) 通过插值遮挡比例生成渐进式遮挡图像序列；(4) 用列表级损失训练模型按可见度排序。

### 关键设计

1. **目标感知硬负样本生成（Object-Aware Hard Negative Generation）**:

    - 功能：生成高质量的图像负样本，使原本正确的回答变成幻觉
    - 核心思路：使用 Grounding-DINO-Tiny（172M 参数）进行零样本目标检测。按优先级选择遮挡目标：chosen 答案首句中的目标 → 问题中的目标 → 答案其他目标 → 随机检测目标。遮挡选中目标的 bounding box 并用视觉提示（红色圆圈）高亮遮挡区域，引导模型关注缺失的目标。通过 Idefics2-8B 验证遮挡后确实产生幻觉——如果不幻觉则换一个目标重试
    - 设计动机：与随机裁剪不同，目标感知遮挡保留了图像的全局语义上下文，仅移除关键目标，产生更有针对性的"硬"负样本

2. **自动列表级样本构造（Automatic Listwise Sample Construction）**:

    - 功能：无需额外标注，自动生成渐进式遮挡的图像列表
    - 核心思路：给定正样本图像 $x_1$（原图）和硬负样本图像 $x_L$（完全遮挡），对中间的第 $k$ 个图像，从图像边缘开始渐进遮挡 $\frac{k-1}{L-1} \times 100\%$ 的 bounding box。最终得到 $L$ 张按目标可见度从高到低排列的图像序列，其中 $x_1$ 可见度最高（正样本），$x_L$ 可见度最低（硬负样本）
    - 设计动机：通过连续的遮挡比例插值，自动生成细粒度的列表级偏好数据，避免了昂贵的人工标注或扩散模型生成

3. **列表级偏好优化损失（Listwise Preference Loss）**:

    - 功能：训练模型使其生成正面文本的似然随目标可见度递增
    - 核心思路：定义列表级损失 $\mathcal{L}_{\text{Listwise}}(\theta) = -\log\left(\prod_{k=1}^{z} \frac{\exp(S_k)}{\sum_{j=k}^{z}\exp(S_j)}\right)$，其中 $S_k = \beta \log \frac{\pi_\theta(w|x_k, q)}{\pi_{\text{ref}}(w|x_k, q)}$ 是模型在给定图像 $x_k$ 下生成正面回答的归一化对数似然。最小化此损失使得 $S_1 > S_2 > \cdots > S_L$，即目标越可见，正面回答的概率越高。总损失为 $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Anchor}} + \mathcal{L}_{\text{Listwise}}$
    - 设计动机：列表级排序能捕获多个样本间的相对关系，比成对比较提供更丰富的梯度信号

### 损失函数 / 训练策略

总损失由三部分组成：标准 DPO 损失（文本偏好）、Anchor 损失（增强原图的正面回答概率）和列表级损失（图像可见度排序）。使用 LoRA 微调（rank=8, alpha=8），Idefics2-8B 训练 3 epoch，学习率 5e-6；LLaVA-v1.5 训练 1 epoch，学习率 1e-6。训练数据从 Silkie 和 LLaVA-Instruct-150K 中采样 10K 偏好数据。

## 实验关键数据

### 主实验（LLaVA-v1.5-7B）

| 方法 | ObjHal CHAIRs↓ | ObjHal CHAIRi↓ | MMHal Score↑ | MMHal HalRate↓ | AMBER CHAIRs↓ |
|------|----------------|----------------|-------------|----------------|---------------|
| Base | 49.7 | 26.1 | 2.02 | 0.65 | 7.7 |
| + DPO | 42.3 | 23.2 | 2.00 | 0.69 | 6.7 |
| + mDPO | 30.7 | 16.0 | 2.26 | 0.56 | 5.3 |
| + **LPOI** | **24.3** | **14.6** | **2.38** | **0.53** | **4.8** |

### 消融实验（Idefics2-8B, 5K数据）

| 配置 | ObjHal CHAIRs↓ | MMHal Score↑ | HalRate↓ | AMBER CHAIRs↓ |
|------|----------------|-------------|----------|---------------|
| Full LPOI | **5.7** | **2.74** | **0.40** | **2.8** |
| w/o DPO loss | 7.7 | 2.56 | 0.44 | 3.3 |
| w/o DPO + Anchor | 6.0 | 2.50 | 0.45 | 3.5 |
| List size 3 | 7.3 | 2.86 | 0.36 | 2.9 |
| List size 4 | 6.7 | 2.86 | 0.36 | 2.5 |
| List size 5 | **5.3** | **2.88** | **0.36** | 2.6 |
| w/o visual prompting | 5.3 | 2.74 | 0.40 | 2.7 |
| w/ visual prompting | **5.0** | **2.91** | **0.35** | **2.6** |

### 关键发现

- LPOI 在 Object HalBench 上相比 mDPO 将 CHAIRs 从 30.7 降至 24.3（-20.8%），在 LLaVA-v1.5-7B 上效果最显著
- 列表大小越大性能越好（3→5），说明更细粒度的排序信号有助于模型学习
- 视觉提示（红色圆圈标注遮挡区域）带来显著提升，通过注意力图验证模型确实更关注遮挡区域
- 在相同 GPU 预算（20小时）下，LPOI 仍优于 DPO 和 mDPO，训练效率合理
- 去掉 DPO 损失后性能下降，说明文本偏好信号（来自 rejected 回答）仍然不可或缺
- 人类评估中，在 80 个样本上三位标注者一致偏好 LPOI 生成的回答
- 在 HallusionBench 上也取得最佳综合表现（All Acc 49.78 vs mDPO 48.45）

## 亮点与洞察

- **"遮挡→插值→排序"的管道**极其优雅——用简单的几何操作（遮挡 bounding box 并渐变遮挡比例）就自动构造了高质量的列表级偏好数据，完全不需要额外标注或昂贵的扩散模型。这个 idea 可以迁移到任何需要列表级偏好数据的多模态任务
- **视觉提示（红色圆圈）**提升了模型对遮挡区域的注意力，saliency map 可视化清晰地证明了这一点。这个简单的技巧提醒我们在构造负样本时不仅要改变输入，还要引导模型注意改变的位置
- **验证模块**确保遮挡后确实产生幻觉，避免了"无效遮挡"的噪声——虽然去掉验证模块后仍优于基线，但有验证时效果更好

## 局限与展望

- 仅关注图像-文本域，未探索音频等其他模态的列表级偏好学习
- 提示语仅限英语，多语言支持有待探索
- Grounding-DINO 的检测精度直接影响负样本质量——约 80% 的关键目标能被正确检测
- 列表级损失带来额外训练开销（list size=5 时每 epoch 训练时间约为 DPO 的 3 倍）
- 对于难以通过 bounding box 遮挡的属性幻觉（如颜色、大小错误），该方法可能力有不及
- 可考虑扩展到视频理解中的时序幻觉缓解

## 相关工作与启发

- **vs DPO for VLMs**: 标准 DPO 仅学习文本偏好，忽略图像信息，容易过拟合文本模式；LPOI 通过图像级别的列表排序强制模型关注视觉信息
- **vs mDPO**: mDPO 用随机裁剪生成二元负样本，信息损失不可控；LPOI 通过目标感知遮挡保留上下文，并提供列表级（而非二元）的排序信号
- **vs V-DPO**: V-DPO 用扩散模型编辑图像生成负样本，计算成本高；LPOI 仅需目标检测+遮挡，效率高得多
- **vs LiPO (文本域)**: LiPO 在文本域应用列表级 DPO，LPOI 是首次将其拓展到图像域，并通过遮挡插值解决了图像排序数据构造的难题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个 VLM 列表级偏好优化，遮挡+插值构造列表数据的思路简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三个模型×三个基准，丰富的消融（列表大小、损失组成、视觉提示等），人类评估
- 写作质量: ⭐⭐⭐⭐ 方法流程图清晰，saliency map 可视化有说服力
- 价值: ⭐⭐⭐⭐⭐ ACL Main，为 VLM 偏好优化开辟了列表级排序的新方向，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)

</div>

<!-- RELATED:END -->
