# Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation

**会议**: CVPR 2025  
**arXiv**: [2603.12247](https://arxiv.org/abs/2603.12247)  
**代码**: https://github.com/VisionXLab/FIRM-Reward  
**领域**: 图像生成与编辑 / 奖励建模  
**关键词**: 奖励模型, 强化学习, 图像编辑, T2I生成, reward hacking, MLLM

## 一句话总结

提出 FIRM 框架——通过"差异优先"（编辑）和"计划-打分"（生成）的数据构建流水线训练专用奖励模型（FIRM-Edit-8B / FIRM-Gen-8B），配合"Base-and-Bonus"奖励策略（CME/QMA）解决 RL 中的奖励 hacking 问题，在图像编辑和 T2I 生成任务上均取得 SOTA。

## 研究背景与动机

1. **RL 在图像生成/编辑中的兴起**：DDPO、DPOK 等方法将扩散去噪过程建模为 MDP，利用 PPO 直接优化策略；Edit-R1 用 GRPO 在编辑任务中取得进展
2. **Critic 不可靠是核心瓶颈**：通用 MLLM（如 Qwen3-VL）作为零样本奖励模型存在严重的幻觉、物体忽略和空间推理缺陷——为 RL 提供嘈杂且误导性的奖励信号
3. **反直觉发现——MLLM 作为评估者 vs 描述者**：MLLM 在直接评判编辑质量时频繁遗漏细粒度细节，但在"解题"（如描述两图差异）时表现优异，启发了"差异优先"的评估设计
4. **Reward Hacking 的严重性**：简单的线性组合奖励（如 0.5·Execution + 0.5·Consistency）导致模型发现捷径——输出与输入几乎相同的图像获取高 Consistency 分数；T2I 中模型对短提示生成黑色剪影也能满足文本条件
5. **扩大奖励模型 ≠ 更好**：Qwen3-VL-32B 比 8B 更大，但在 ImgEdit 上反而导致 −0.07 的性能下降——简单放大通用 VLM 不保证更好的奖励建模
6. **缺乏统一的编辑+生成评估基准**：现有 benchmark 要么只覆盖编辑要么只覆盖生成，且缺少经过严格人工标注和分数平衡的评估集

## 方法详解

### 整体框架

FIRM 包含四个核心组件：数据构建流水线 → 奖励模型训练 → 评估基准 → RL 奖励策略

### 关键设计 1：FIRM-Edit 数据流水线（"差异优先"）

1. **双层差异分析**：给 SOTA MLLM 提供原始图和编辑图，要求识别明显修改 + 细微修改 → 生成统一差异报告
2. **差异条件评估**：将差异描述 + 图像对 + 编辑指令一起送入 MLLM 评估器 → 输出 Execution（1-5 分）和 Consistency（1-5 分）
3. **核心洞察**：先让模型"看到差异"再打分，比直接打分准确得多——将评估问题转化为模型更擅长的描述问题
4. **低分样本平衡**：通过改写指令人为制造低质量匹配，确保 1-5 分均匀分布
5. **数据来源**：OpenGPT-4o-Image、GPT-Image-Edit、ShareGPT-4o-Image、ImgEdit → 共 370K 样本

### 关键设计 2：FIRM-Gen 数据流水线（"计划-打分"）

1. **Stage 1 — 显式标准规划**：LLM（Qwen3-32B）从生成提示中提取评估清单——主体准确性、风格对齐、负约束等
2. **Stage 2 — 结构化分析打分**：MLLM（Qwen3-VL-235B）根据清单逐项检查生成图像 → 汇总为最终分数
3. **多模型生成确保多样性**：用 5 种不同架构/能力的模型（Ovis、Z-image、Flux.1-dev、SDXL、SD1.5）从同一提示生成图像，防止奖励模型过拟合单一生成器的伪影
4. **共 293K 样本**

### 关键设计 3："Base-and-Bonus" 奖励策略

**编辑：Consistency-Modulated Execution (CME)**：
$$R_{\text{CME}} = \text{Execution} \cdot (0.6 + 0.4 \cdot \text{Consistency})$$
- Execution 作为必要条件——如果 Execution 低，无论 Consistency 多高总奖励都被压制
- Consistency 作为加成项——在执行有效编辑的前提下细化结构保真度

**生成：Quality-Modulated Alignment (QMA)**：
$$R_{\text{QMA}} = \text{InsFollowing} \cdot (0.4 + 0.6 \cdot \text{Quality})$$
- 指令遵循是基础，图像质量作为调制项
- 阻止模型对短提示生成黑色剪影的 hacking 行为

### FIRM-Bench

- 807 个人工标注样本：FIRM-Bench-Edit（301 Execution + 256 Consistency） + FIRM-Bench-Gen（250 Instruction Following）
- 严格的数据隔离（与训练数据无重叠）、均匀的 GT 分数分布
- 评估指标：MAE（预测分数与人工标注的绝对误差均值）

## 实验关键数据

### FIRM-Bench 奖励模型评估（MAE ↓）

| 模型 | Edit Exec. | Edit Cons. | Edit 总 | Gen 总 |
|------|-----------|-----------|--------|--------|
| GPT-5 | 0.62 | 0.73 | 0.67 | 0.52 |
| Gemini-3-Pro | 0.54 | 0.57 | **0.55** | **0.40** |
| Qwen3-VL-235B | 0.72 | 0.91 | 0.81 | 0.56 |
| Qwen3-VL-8B | 0.66 | 1.12 | 0.87 | 0.63 |
| **FIRM-Edit-8B** | **0.53** | **0.73** | **0.62** | — |
| **FIRM-Gen-8B** | — | — | — | **0.51** |

### 图像编辑 RL 性能（Table 3）

| 模型 | GEdit-Bench Overall | ImgEdit Overall |
|------|-------------------|-----------------|
| Qwen-Image-Edit-2509 (baseline) | 7.54 | 4.35 |
| + RL w/ Qwen3-VL-8B | 7.69 (+0.15) | 4.36 (+0.01) |
| + RL w/ Qwen3-VL-32B | 7.65 (+0.11) | 4.28 (−0.07) |
| + **FIRM-Qwen-Edit** | **7.84 (+0.30)** | **4.42 (+0.07)** |
| GPT-Image | 7.53 | 4.20 |
| UniWorld-Qwen | 7.76 | 4.48 |

### T2I 生成 RL 性能（Table 4）

| 模型 | GenEval | DPG-Bench | TIIF | UniGen Short | UniGen Long |
|------|---------|-----------|------|-------------|-------------|
| SD3.5-Medium (baseline) | 0.52 | 84.08 | 70.17 | 60.71 | 64.67 |
| + RL w/ Qwen3-VL-8B | 0.76 | 86.87 | 75.99 | 67.17 | 74.50 |
| + **FIRM-SD3.5** | **0.77** | **87.16** | **77.12** | **69.56** | **76.22** |
| BAGEL | 0.82 | 85.07 | 71.50 | 59.91 | 71.26 |

### 奖励策略消融（Table 5，基于 FIRM-Edit-8B）

| 方法 | GEdit Overall | ImgEdit |
|------|-------------|---------|
| Baseline（无 RL） | 7.54 | 4.35 |
| Weighted (0.5+0.5) | 1.06 | 2.17 |
| Weighted (0.6+0.4) | 6.51 | 3.73 |
| Edit-R1 (Non-CoT Logits) | 4.06 | 2.75 |
| **CME (Ours)** | **7.84** | **4.42** |

### 关键发现

- **FIRM-Edit-8B 用仅 8B 参数超越 GPT-5**（MAE 0.62 vs 0.67）——专用训练远胜通用能力放大
- **仅 2,400 训练样本（150步×16采样）达到 UniWorld 的 27K 样本水平**——精准奖励信号大幅提升样本效率
- **简单线性奖励组合导致灾难性 hacking**：0.5+0.5 权重下 GEdit 从 7.54 暴跌至 1.06
- **FIRM-Gen-8B 在复杂提示上优势更显著**：UniGen Long 上 +11.55 vs Qwen3-VL-8B 的 +9.83——越复杂的场景越需要精准的奖励
- **编辑和生成的奖励曲线呈相反模式**：编辑中 FIRM 分数低于通用 VLM（因为 VLM 忽略细微变化给虚高分），生成中 FIRM 分数高于通用 VLM（因为 VLM 幻觉导致虚低分）

## 亮点与洞察

1. **"差异优先"评估范式**：利用 MLLM "解题强于评估"的特性，先让模型描述差异再打分——简洁而有效地绕过了直接评估的幻觉问题
2. **Reward Hacking 的深刻分析**：不仅发现了"懒编辑"和"黑色剪影"两种 hacking 模式，还通过乘法奖励（而非加法）从根本上解决——Execution 为零则总奖励为零，彻底堵死捷径
3. **"扩大不等于更好"的发现**：Qwen3-VL-32B 在 ImgEdit 上反而退化，揭示了通用 VLM 做领域特定评估的根本局限
4. **端到端覆盖 RL 全栈**：数据→模型→基准→奖励策略→下游训练，每个环节都有创新点

## 局限性

- **奖励模型基于 Qwen3-VL-8B 初始化**：受限于底座模型的能力上限，对极复杂的空间推理可能仍有盲点
- **FIRM-Bench 规模有限**：807 样本的人工标注集可能不够覆盖所有 edge case
- **CME/QMA 的权重仍需手动设定**：$w_1, w_2$ 的最优值可能因任务和模型不同而变化
- **未探索视频编辑/生成**：框架理论上可推广到视频，但未验证
- **T2I 生成基座较弱**：使用 SD3.5-Medium 作为基座，未在更强的模型（如 FLUX、DALL-E 3 级别）上验证

## 相关工作与启发

- **vs EditScore**：EditScore 也训练编辑奖励模型但仅覆盖编辑；FIRM 同时覆盖编辑和生成，且提出乘法奖励策略
- **vs Edit-R1**：Edit-R1 用通用 MLLM 的 logits 做奖励，信号嘈杂导致 GEdit 从 7.54 暴跌至 4.06；FIRM 的专用奖励模型提供精准信号
- **vs T2I-R1**：T2I-R1 用 GRPO + CoT 推理但不关注奖励模型质量；FIRM 证明奖励模型的精度才是 RL 的根本瓶颈
- **启发**："问题解构"大于"模型放大"——与其追求更大的通用 VLM 做评估，不如设计让 VLM 更擅长回答的评估流程（差异描述→打分、清单→打分）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "差异优先"流水线 + 乘法奖励策略，洞察深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖编辑+生成两大任务、多个 benchmark、多基线对比、完整消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，从问题到方案的逻辑链完整
- **实用性**: ⭐⭐⭐⭐⭐ — FIRM-Edit-8B/Gen-8B 可直接作为 RL 训练的 critic 使用，代码和模型开源
- **综合推荐**: ⭐⭐⭐⭐⭐
