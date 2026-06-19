---
title: >-
  [论文解读] Less is More: Empowering GUI Agent with Context-Aware Simplification
description: >-
  [ICCV 2025][LLM Agent][GUI Agent] 提出 SimpAgent——一种上下文感知的简化框架，通过基于遮挡的元素剪枝（训练时随机遮挡无关元素区域）和一致性引导的历史压缩（在 LLM 中间层直接丢弃历史视觉 token + KL散度一致性约束），在降低27% FLOPs 的同时取得多个 GUI 导航基准的 SOTA。
tags:
  - "ICCV 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "上下文简化"
  - "元素剪枝"
  - "历史压缩"
  - "计算效率"
---

# Less is More: Empowering GUI Agent with Context-Aware Simplification

**会议**: ICCV 2025  
**arXiv**: [2507.03730](https://arxiv.org/abs/2507.03730)  
**代码**: [github.com/JiuTian-VL/SimpAgent](https://github.com/JiuTian-VL/SimpAgent)  
**领域**: LLM Agent  
**关键词**: GUI Agent, 上下文简化, 元素剪枝, 历史压缩, 计算效率

## 一句话总结

提出 SimpAgent——一种上下文感知的简化框架，通过基于遮挡的元素剪枝（训练时随机遮挡无关元素区域）和一致性引导的历史压缩（在 LLM 中间层直接丢弃历史视觉 token + KL散度一致性约束），在降低27% FLOPs 的同时取得多个 GUI 导航基准的 SOTA。

## 研究背景与动机

### 问题定义

GUI Agent 需要根据任务目标、当前截图和历史上下文，在图形界面中生成操作（点击、滑动等）完成复杂任务。纯视觉方案是未来方向，但面临严峻的上下文建模挑战。

### 已有方法的不足

现有纯视觉 GUI Agent 主要追求大规模预训练数据来提升 GUI 理解能力，存在两大被忽视的挑战：

**元素上下文的高密度和松散关联**：每张截图平均包含56-180个 UI 元素，但这些元素之间的关联松散。实验表明，遮挡一半截图中的无关元素后性能反而提升（66.0% → 68.8%）

**历史上下文的高冗余**：引入4帧历史观测会增加3.4倍计算开销，但仅提升3.0%性能。历史视觉信息存在严重冗余

### 核心矛盾

用更多数据预训练 vs 优化上下文建模的性价比问题。例如 OS-Atlas 用1.9M预训练数据仅提升0.8%，而 SimpAgent 不用预训练数据即可获得相当甚至更好的提升。

## 方法详解

### 整体框架

SimpAgent 包含两个核心组件：
1. 训练阶段：用遮挡操作随机剪枝截图中的无关元素区域
2. 训练+推理阶段：在 LLM 中间层丢弃历史视觉 token，并通过一致性损失引导压缩过程

### 关键设计

#### 1. **基于遮挡的元素剪枝（Masking-based Element Pruning）**
- **功能**：训练时在当前截图上随机遮挡一个矩形区域
- **核心思路**：
    - 矩形尺寸 $h, w \sim U(a, b)$，中心点 $p_c$ 从均匀分布中采样
    - 遮挡操作以概率 $p$ 执行，推理时不遮挡
    - 被遮挡区域像素设为固定值 $v$

$$o_t^m = \mathcal{M}(o_t) = \begin{cases} v, & (x,y) \in \mathcal{R} \\ o_t(x,y), & \text{otherwise} \end{cases}$$

- **设计动机**：
    - 核心操作点平均仅占截图面积的2%，无关元素占据绝大部分
    - UI 设计遵循模块化原则，元素间关联松散，遮挡无关元素不影响理解
    - 避免了复杂的元素关系建模——不需要判断哪些元素是无关的

#### 2. **基于 LLM 的历史丢弃（LLM-based History Dropping）**
- **功能**：在 LLM 的第 $k$ 层直接丢弃所有历史视觉 token
- **核心思路**：浅层 LLM 通过因果自注意力已将视觉信息压缩到相邻的动作 token 中（通过注意力可视化验证），因此丢弃历史视觉 token 后信息可保留在动作 token 中
- **设计动机**：
    - 无需额外参数
    - 相比 VoCo-LLaMA 的注意力掩码调整方法，兼容 FlashAttention 等高效注意力实现
    - 实现27%的 FLOPs 降低

#### 3. **一致性引导训练（Consistency Guidance）**
- **功能**：在训练时同时维护截断分支和完整分支，最小化两者输出分布的 KL 散度
- **核心思路**：

$$\mathcal{L} = -\mathbb{D}_{KL}[\pi_\theta(\tilde{a}_t|o_t^m, H_t, G) \| \pi_\theta(a_t|o_t^m, H_t^c, G)] - \sum_t \log \pi_\theta(\tilde{a}_t|o_t^m, H_t, G) - \sum_t \log \pi_\theta(a_t|o_t^m, H_t^c, G)$$

- **设计动机**：单纯的丢弃策略是隐式压缩，存在信息损失（AITW 上下降1.7%）。一致性引导提供显式指导，将压缩损失降到几乎为零

### 损失函数 / 训练策略

总训练目标包含三部分：
1. 完整分支的交叉熵损失
2. 截断分支的交叉熵损失
3. 两个分支之间的 KL 散度一致性损失

推理时仅使用截断分支，实现27% FLOPs 降低。

## 实验关键数据

### 主实验

| 方法 | 预训练数据量 | 参数量 | AITW | Mind2Web-CT | GUI-Odyssey |
|------|------------|--------|------|-------------|-------------|
| SeeClick | 850K | 9.6B | 59.3 | 25.5 | - |
| ShowUI | 256K | 2B | 70.0 | 37.2 | - |
| OdysseyAgent | - | 9.6B | - | - | 74.3 |
| Qwen2VL（基线）| - | 2B | 69.0 | 46.7 | 74.9 |
| **SimpAgent** | **-** | **2B** | **71.3** | **47.1** | **76.0** |
| **SimpAgent-M** | **-** | **2B** | **71.5** | **48.7** | **77.4** |

AndroidControl 结果：

| 方法 | 预训练数据量 | 参数量 | SR |
|------|------------|--------|-----|
| OS-Atlas | 1.9M | 4B | 67.5 |
| Qwen2VL（基线）| - | 2B | 68.4 |
| **SimpAgent** | **-** | **2B** | **69.1** |

### 消融实验

| 组件 | FLOPs (T) | AITW | GUI-Odyssey | 说明 |
|------|-----------|------|-------------|------|
| 基线（Qwen2VL）| 11.90 | 69.0 | 74.9 | 完整历史 |
| + 历史压缩 | 8.71 (↓27%) | 67.3 (↓1.7) | 71.8 (↓3.1) | 隐式压缩，信息损失 |
| + 一致性引导 | 8.71 | 68.9 (↑1.6) | 73.7 (↑1.9) | 显式引导恢复性能 |
| + 元素剪枝 | 8.71 | **71.3 (↑2.4)** | **76.0 (↑2.3)** | 三个组件叠加超过基线 |

### 与其他压缩方法的对比

| 方法 | Step SR | FLOPs (T) |
|------|---------|-----------|
| 无压缩 | 69.0 | 11.90 |
| TokenMerger | 68.9 | 9.28 |
| Victor | 67.6 | 9.28 |
| FastV-50 | 66.0 | 10.15 |
| FastV-0 | 63.8 | 8.71 |
| **Ours** | **68.9** | **8.71** |

### 关键发现

1. **元素剪枝提升显著**：即使遮挡截图50%的区域，性能仍然有提升，证明大量 UI 元素确实是"噪声"
2. **均匀分布优于高斯分布**：UI 元素的空间分布比预期更复杂，简单的均匀遮挡策略更鲁棒
3. **一致性引导是压缩成功的关键**：将压缩导致的1.7%性能下降恢复到接近无损
4. **无需预训练数据即可超越需要大量预训练数据的方法**：SimpAgent-2B 的0.7%提升 ≈ OS-Atlas-4B 用1.9M数据的0.8%提升

## 亮点与洞察

1. **对 GUI 截图特性的深刻分析**：系统量化了元素密度（56-180个/截图）和历史冗余（3.4× 计算开销 → 3.0%性能提升）
2. **巧妙的遮挡剪枝策略**：不需要识别哪些元素无关，随机遮挡就能以高概率删除无关元素（因为核心操作区域仅占2%面积）
3. **训练-推理不对称设计**：训练时遮挡增强鲁棒性，推理时不遮挡保留完整信息
4. **计算效率显著**：27% FLOPs 降低同时性能提升，证明"less is more"的主张

## 局限与展望

1. 遮挡策略是数据无关的，未利用 UI 的结构信息（如组件层级、布局规律）
2. 仅验证了 Qwen2-VL-2B 作为基座模型，对更大模型的泛化性未知
3. 历史压缩丢弃了所有历史视觉 token，可能遗失关键的视觉变化信息
4. 丢弃层 $k$ 的选择是经验性的（$k=3$），缺乏自适应机制
5. 仅评估了移动和网页场景，桌面应用的适用性未验证

## 相关工作与启发

- FastV 发现浅层 LLM 可将视觉特征聚合到锚点 token，SimpAgent 在此基础上进一步发展了直接丢弃策略
- ShowUI 和 Iris 尝试通过低层视觉线索消除背景，但在高密度元素场景中效果有限
- OdysseyAgent 提出历史重采样模块压缩历史，但引入额外参数且忽略多模态交互

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 遮挡剪枝的思路简单直觉但反直觉地有效，一致性引导是扎实的技术贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个数据集，10+基线对比，充分的消融和敏感性分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机分析深入，pilot experiments 令人信服
- **价值**: ⭐⭐⭐⭐ — 为 GUI Agent 提供了计算友好的训练方案，"不用预训练数据"的卖点很有实践价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](../../CVPR2025/llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[CVPR 2026\] ReFAct: Empowering Multimodal Web Agents with Visual and Context Focusing](../../CVPR2026/llm_agent/refact_empowering_multimodal_web_agents_with_visual_and_context_focusing.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)

</div>

<!-- RELATED:END -->
