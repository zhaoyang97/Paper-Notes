---
title: >-
  [论文解读] PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs
description: >-
  [ICML2025][图像生成][contextual bandit] 提出 PAK-UCB 上下文老虎机算法，通过为每个生成模型学习独立的核函数，在线预测给定 prompt 下的最优模型，实现 prompt 级别的生成模型/LLM 选择，并用随机傅里叶特征（RFF）降低计算开销。 核心问题：当前生成模型选择通常基于所有 p…
tags:
  - "ICML2025"
  - "图像生成"
  - "contextual bandit"
  - "生成模型选择"
  - "核方法"
  - "UCB"
  - "随机傅里叶特征"
  - "提示学习"
---

# PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs

**会议**: ICML2025  
**arXiv**: [2410.13287](https://arxiv.org/abs/2410.13287)  
**代码**: [github.com/yannxiaoyanhu/dgm-online-select](https://github.com/yannxiaoyanhu/dgm-online-select)  
**领域**: 在线学习 / LLM选择  
**关键词**: contextual bandit, 生成模型选择, 核方法, UCB, 随机傅里叶特征, prompt-aware

## 一句话总结

提出 PAK-UCB 上下文老虎机算法，通过为每个生成模型学习独立的核函数，在线预测给定 prompt 下的最优模型，实现 prompt 级别的生成模型/LLM 选择，并用随机傅里叶特征（RFF）降低计算开销。

## 研究背景与动机

**核心问题**：当前生成模型选择通常基于所有 prompt 上的平均分数选出单一最优模型，但不同模型在不同类型 prompt 上表现差异显著。例如：

- Gemini-2.5-Flash 在 Python 代码补全上 pass rate 92.3%，但 Java→C++ 翻译仅 48.8%；Qwen-Plus 则相反（86.5% vs 82.3%）
- Stable Diffusion v1.5 在"car"类 prompt 上 CLIPScore 更高（36.10 vs 35.68），但在"dog"类上不如 PixArt-α（36.37 vs 37.24）

**动机**：需要一种在线学习方法，根据输入 prompt 的特征动态选择最优生成模型，而非依赖全局平均分数。这本质上是一个**上下文老虎机（contextual bandit）**问题——prompt 是 context，模型是 arm，生成质量分数是 reward。

**与现有 CB 方法的区别**：传统 LinUCB/KernelUCB 假设所有 arm 共享同一权重向量，但在模型选择场景中，同一 prompt 输入到不同模型会产生不同分数，需要 **per-arm** 的独立预测函数。

## 方法详解

### 问题形式化

设有 $G$ 个生成模型 $\mathcal{G} = [G]$，在 $T$ 轮迭代中：

1. 每轮 $t$ 收到 prompt $y_t \sim \rho$
2. 算法选择模型 $g_t$，采样响应 $x_t \sim P_{g_t}(\cdot|y_t)$
3. 收到分数 $s_t = s(y_t, x_t) \in [-1, 1]$

目标：最小化累积遗憾（regret）：

$$\text{Regret}(T) = \sum_{t=1}^{T} \left( s_\star(y_t) - s_{g_t}(y_t) \right)$$

其中 $s_\star(y) = \max_{g \in \mathcal{G}} s_g(y)$ 是给定 prompt 下的最优期望分数。

### PAK-UCB 算法

**核心假设（Realizability）**：存在 RKHS 映射 $\phi$ 和**每个模型独立**的权重 $w_g^\star$，使得 $s_g(y) = \langle \phi(y), w_g^\star \rangle$。

**算法流程**：

1. 对每个模型 $g$ 维护**独立**的观测集 $\Psi_g$（仅包含选中模型 $g$ 的轮次）
2. 使用核岭回归（KRR）估计均值 $\hat{\mu}_g$ 和不确定性 $\hat{\sigma}_g$：
    - $\hat{\mu}_g = k_y^\top (K + \alpha I)^{-1} v$
    - $\hat{\sigma}_g = \alpha^{-1/2} \sqrt{k(y,y) - k_y^\top (K + \alpha I)^{-1} k_y}$
3. 计算 UCB 分数：$\hat{s}_g = \hat{\mu}_g + \eta \hat{\sigma}_g$
4. 选择 UCB 分数最高的模型：$g_t = \arg\max_g \hat{s}_g$

**关键设计**：per-arm 核函数——每个模型用独立的核矩阵和权重向量，避免共享权重带来的偏差。

### RFF-UCB：降低计算复杂度

PAK-UCB 每轮计算复杂度为 $O(t^3/G^2)$（核矩阵求逆随数据量增长），通过**随机傅里叶特征（RFF）**近似：

- 将 prompt 投影到 $2D$ 维随机特征空间：$\varphi(y) = \frac{1}{\sqrt{D}}[\cos(w_1^\top y), \sin(w_1^\top y), \ldots]$
- 用线性岭回归替代核岭回归
- 计算复杂度降至 $O(tD^2)$（线性增长）

**理论保证**：

| 算法 | 每轮时间复杂度 | 空间复杂度 | Regret 上界 |
|------|---------------|-----------|------------|
| PAK-UCB | $O(t^3/G^2)$ | $O(t^2/G)$ | $\tilde{O}(\sqrt{GT})$ |
| RFF-UCB | $O(tD^2)$ | $O(tD)$ | $\tilde{O}(\sqrt{GT})$ |

两者 regret 上界相同，但 RFF-UCB 在计算上更高效。

## 实验关键数据

### 实验设置

- **T2I 模型**：Stable Diffusion v1.5, PixArt-α, UniDiffuser, DeepFloyd IF
- **LLM**：Gemini-2.5-Flash, o3-mini, Deepseek-Chat, Qwen-Plus
- **评估指标**：CLIPScore（T2I）、pass rate（代码生成）
- **性能指标**：Outscore-the-Best (OtB) 和 Optimal Pick Ratio (OPR)
- **Baseline**：KernelUCB, LinUCB, One-arm Oracle, Random, Naive-KRR

### 核心结果

| 场景 | 方法 | 关键表现 |
|------|------|---------|
| T2I (dog/car) | PAK-UCB-poly3 | CLIPScore 超过 One-arm Oracle，OPR 高 |
| T2I (dog/car) | RFF-UCB-RBF | 接近 PAK-UCB，计算更快 |
| 代码生成 | RFF-UCB-RBF | 根据任务类型自动分配到更优 LLM |
| Sudoku | RFF-UCB-RBF | 超过所有 baseline 方法 |

### 关键发现

- PAK-UCB/RFF-UCB **超过 One-arm Oracle**（知道全局最优单一模型），验证了 prompt 级选择的优势
- 显著优于共享权重的 LinUCB 和 KernelUCB，证实 per-arm 设计的必要性
- RFF-UCB 在保持近似性能的同时大幅降低计算开销
- 算法能适应**新增模型**和**新 prompt 类型**（Setup 3）

## 亮点与洞察

1. **问题建模精准**：将"不同模型擅长不同 prompt"观察转化为 per-arm contextual bandit，解决了传统 CB 共享权重的局限
2. **理论扎实**：PAK-UCB 和 RFF-UCB 均有 $\tilde{O}(\sqrt{GT})$ 的 regret bound，RFF 不损失理论保证
3. **实用性强**：覆盖 T2I、image captioning、LLM 代码生成、Sudoku 等多种场景，验证了通用性
4. **动态适应**：能处理运行中新增模型和 prompt 分布漂移，适合实际部署
5. **计算高效**：RFF 将立方复杂度降为线性，使大规模在线部署可行

## 局限与展望

1. **评估指标单一**：当前仅优化单一分数（如 CLIPScore），未考虑生成多样性、新颖性等多目标
2. **核函数选择**：RBF 和 polynomial kernel 需要手动调参（$\sigma$、$\gamma$），缺乏自适应核选择机制
3. **冷启动问题**：新模型加入时需要初始探索期，实际中可能代价较高
4. **prompt 表示依赖**：依赖 CLIP/RoBERTa 等预训练模型的 embedding，embedding 质量直接影响选择效果
5. **成本未建模**：不同模型的查询成本（API 价格、延迟）可能差异很大，未考虑 cost-aware 选择
6. **实验规模有限**：T2I 实验主要在 MS-COCO 2个类别间选择，场景相对简单

## 相关工作与启发

- **FID-UCB** (Hu et al., 2025a)：非条件生成模型的 MAB 选择，本文扩展到条件生成（带 prompt）
- **Mixture-UCB** (Rezaei et al., 2025)：混合多个生成模型以提高多样性
- **KernelUCB** (Valko et al., 2013)：核化 CB 的基础工作，但假设共享权重
- **离线路由方法** (Luo et al., 2024; Qin et al., 2024)：训练神经网络做 prompt→model 路由，但需大量离线数据
- **启发**：per-arm 核方法思想可推广到其他"同一输入、不同后端"的在线选择场景（如 API 路由、模型集成）

## 评分

- 新颖性: ⭐⭐⭐⭐ — per-arm kernelized CB 用于 prompt-aware 模型选择是新颖的组合
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 T2I/LLM/image captioning 多场景，含适应性实验
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，理论与实验衔接好
- 价值: ⭐⭐⭐⭐ — 解决实际的模型路由问题，理论与实践兼顾

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](../../ICCV2025/image_generation/pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICML 2025\] Stealix: Model Stealing via Prompt Evolution](stealix_model_stealing_via_prompt_evolution.md)
- [\[ICML 2025\] Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction](sample_complexity_of_distributionally_robust_off-dynamics_reinforcement_learning.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](../../ICML2026/image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICLR 2026\] FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference](../../ICLR2026/image_generation/fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference.md)

</div>

<!-- RELATED:END -->
