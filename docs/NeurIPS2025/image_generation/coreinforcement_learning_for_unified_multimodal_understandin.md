# Co-Reinforcement Learning for Unified Multimodal Understanding and Generation

**会议**: NeurIPS 2025  
**arXiv**: [2505.17534](https://arxiv.org/abs/2505.17534)  
**代码**: [https://github.com/mm-vl/ULM-R1](https://github.com/mm-vl/ULM-R1)  
**领域**: 多模态VLM / 强化学习 / 图像生成  
**关键词**: unified multimodal model, GRPO, co-reinforcement learning, vision-language, text-to-image  

## 一句话总结
提出CoRL框架——通过"统一RL→精细RL"两阶段GRPO训练策略，在不依赖额外监督数据的情况下，让统一多模态模型(ULM)的理解和生成能力协同进化，在Janus-Pro-1.5B上取得生成+7%、理解+23%的平均提升。

## 背景与动机
当前多模态大模型分为"理解专用"和"生成专用"两大类，而统一模型(ULM)如Janus-Pro试图同时兼顾理解和生成。GRPO（Group Relative Policy Optimization）在LLM上已展示出优秀的后训练效果（如DeepSeek-R1），但在多模态领域的应用主要集中在理解侧的推理增强，对生成任务的RL优化几乎未被探索，更不用说在统一模型上同时优化双任务。关键痛点：直接对单个任务做RL反而可能伤害另一个任务的性能。

## 核心问题
如何用RL（不依赖大规模监督数据）同时增强统一多模态模型的理解和生成能力？单任务RL训练会相互干扰，需要一种能实现跨任务协同进化的RL范式。

## 方法详解

### 整体框架
CoRL采用"Foundation-then-Specialization"两阶段范式：
- **Stage 1 (Unified RL)**：将理解和生成任务放在同一个GRPO优化框架内联合训练，共享reward group中的advantage估计
- **Stage 2 (Refined RL)**：分别针对生成和理解任务，用各自的reward和数据集做精细化RL优化

基座模型为Janus-Pro（全自回归F-AR架构），包含VQ tokenizer + text tokenizer + LLM。

### 关键设计

1. **Pilot Study揭示的设计直觉**：系统对比了4种RL范式（单独训练、单独训练+权重merge、交替训练、统一训练），发现(a)单独RL对生成任务无效甚至有害，(b)统一RL在两个任务上都有优势。这说明双任务在共享policy下能产生协同效应。

2. **Bidirectional Cycle Consistency Reward (ℛcycle)**：针对图像生成设计的核心reward，包含两个方向：
   - 视觉一致性：用LPIPS计算生成图像与真实图像的感知相似度
   - 文本一致性：用BLIP对生成图像re-caption，再算SPICE分数衡量与原始prompt的语义一致性
   - 形成闭环反馈，同时约束视觉和语义

3. **Text-Image Matching Reward (ℛTIM)**：用ULM自身的表示空间计算文本-图像token间的双向最大余弦相似度，捕捉细粒度的跨模态对齐。比CLIP Score更适合密集长prompt场景（DPG-Bench上效果更好）。

4. **两阶段训练策略**：
   - Stage 1：联合reward = ℛcycle + ℛTIM + λ(ℛAcc + ℛFormat)，去掉KL约束以提升泛化
   - Stage 2：分别用ℛT2I-S2和ℛMCQ/OE-S2做精细化，重新引入KL约束保持稳定，最终权重通过高斯分布merge得到同时支持两种QA格式的模型

### 损失函数 / 训练策略
- GRPO基于group-wise relative advantage，不需要额外的value network
- Stage 1使用22K COCO样本（每个样本同时包含生成prompt和理解QA对）
- Stage 2使用16K生成样本 + MCoT-Instruct多选/开放式QA
- 训练硬件：8×H20 96G GPU，batch size 16/32

## 实验关键数据

| 数据集 | 指标 | ULM-R1 (1.5B) | Janus-Pro (1.5B) | 提升 |
|--------|------|----------------|------------------|------|
| GenEval | Overall | 0.77 | 0.73 | +0.04 |
| WISE | Overall | 0.33 | 0.26 | +0.07 |
| DPG-Bench | Overall | 83.92 | 82.63 | +1.29 |
| MMMU | Acc | 42.3 | 36.3 | +6.0 |
| WeMath | Acc | 21.1 | 5.9 | +15.2 |
| LogicVista | Acc | 34.5 | 23.9 | +10.6 |
| MMVet | Score | 43.9 | 39.8 | +4.1 |
| POPE | Acc | 88.9 | 86.2 | +2.7 |

### 消融实验要点
- 统一RL在生成和理解上都优于Cold-SFT、单独RL、交替RL等范式
- ℛcycle和ℛTIM有互补效果，组合最优（GenEval 77.3 + DPG 83.9 vs 单独74-77）
- LPIPS作为ℛcycle中的视觉一致性指标优于PSNR/MSE/SSIM
- λ=0.8为最佳平衡点（太大会损害生成）
- 在Janus-1.3B和Janus-Pro-7B上也有效，验证了可扩展性

## 亮点 / 我学到了什么
- 统一RL中跨任务的reward sharing能产生协同效应——这是一个重要的实证发现，说明理解和生成不是零和博弈
- 用模型自身的表示空间做text-image matching reward (ℛTIM) 比外部CLIP Score更有效，是很聪明的自举策略
- "Foundation-then-Specialization" 两阶段设计平衡了跨任务协同和任务特异性增强
- 仅用22K COCO数据就能通过RL获得显著提升，数据效率非常高

## 局限性 / 可改进方向
- 生成和理解之间仍有性能gap，理解侧的reward比较简单（仅accuracy+format）
- 低分辨率（384²）限制了生成质量，相比专用Diffusion模型差距明显
- 仅在Janus系列验证，未在AR-Diff hybrid架构（如Show-o, ILLUME+）上测试
- 生成侧的reward依赖BLIP re-captioning和LPIPS，这些外部模型也有其bias

## 与相关工作的对比
- vs **SimpleAR**：SimpleAR只用CLIP reward做生成侧RL，CoRL同时优化生成+理解且reward更精细
- vs **R1-like MLLMs** (Vision-R1等)：这些只做理解侧的reasoning RL，CoRL扩展到双任务协同
- vs **DPO方法** (Emu3, HermesFlow)：DPO需要preference data，CoRL只用verifiable reward，更高效

## 与我的研究方向的关联
- 与ideas/multimodal_vlm/20260316_three_level_decoupling_unified.md相关——CoRL验证了统一模型中理解和生成的协同效应，为该idea中的"三级表示解耦"提供了RL优化的可能路径
- 启发：能否将ℛTIM这种自举式reward设计扩展到更多任务（如视频理解+生成）？
- Foundation-then-Specialization范式可能对多任务VLM后训练有通用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统探索ULM上的co-RL，pilot study设计合理，但核心技术（GRPO+自定义reward）是现有工具的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验非常细致，覆盖了策略对比、reward设计、超参数和可扩展性
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pilot study→方法→实验的叙事逻辑好，但部分符号偏多
- 对我的价值: ⭐⭐⭐⭐⭐ 统一模型的RL后训练是当前热点，CoRL的framework和reward设计都有参考价值
