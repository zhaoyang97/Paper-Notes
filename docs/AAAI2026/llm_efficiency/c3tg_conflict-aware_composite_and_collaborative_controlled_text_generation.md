# C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation

**会议**: AAAI 2026  
**arXiv**: [2511.09292](https://arxiv.org/abs/2511.09292)  
**代码**: 无  
**领域**: LLM效率 / 可控文本生成  
**关键词**: controlled text generation, multi-attribute control, KL divergence, energy function, conflict resolution

## 一句话总结
提出 C3TG 框架，通过两阶段方法实现多维度细粒度可控文本生成：生成阶段用加权 KL 散度融合属性分布调整 token 概率，优化阶段用能量函数（分类器分数 + 冲突惩罚项）结合 Feedback Agent 迭代重写，在 17 个属性子类上达到 90.4% 属性准确率且大幅降低毒性。

## 研究背景与动机

1. **领域现状**：可控文本生成（CTG）旨在控制生成文本的情感、风格、语调、主题等属性。现有方法分为两类：直接调节解码分布（PPLM、GeDi、COLD）和间接控制（prompting、fine-tuning）。
2. **现有痛点**：(1) 大多数方法只能控制单一或简单属性；(2) 多属性同时控制时缺乏冲突解决机制——增强一个属性可能抑制或放大另一个；(3) 缺少迭代反馈优化流程。
3. **核心矛盾**：多个属性之间可能存在冲突或依赖关系（如"幽默"和"正式"天然冲突），单次生成无法同时满足所有属性目标。
4. **本文要解决什么**：实现 17 个属性维度的细粒度同时控制，并处理属性间冲突。
5. **切入角度**：大模型生成 + 小模型评估的协作范式：LLM 负责生成，BERT 分类器负责评估属性对齐度，Feedback Agent 驱动迭代改写。
6. **核心 idea**：生成阶段用属性先验的几何加权平均采样 token；优化阶段用分类器分数+维度稳定性惩罚构建能量函数，通过三阶段 Chain-of-Prompt 迭代优化。

## 方法详解

### 整体框架
两阶段：(1) **Generation Phase**：从 Llama2 基础模型和 n 个属性模型中提取分布，通过加权 KL 散度最优解 $P^*(x|x_{1:t-1}) = \prod_i Q_i^{\lambda_i/\Lambda} / Z$ 采样 token；(2) **Optimization Phase**：BERT 分类器评估 17 个属性维度的对齐度，能量函数 $E(x) = \sum \alpha_i|C_{A_i}(x) - T_i| + \sum \beta_j|C_{A_j}(x) - C_{A_j}(x_{prev})|$ 驱动三阶段迭代改写。

### 关键设计

1. **Weighted KL-Divergence Fusion（生成阶段）**
   - 做什么：将多个属性模型的 token 分布融合为统一采样分布
   - 核心思路：最小化加权 KL 散度 $\mathcal{J}[P] = \sum_i \lambda_i D_{KL}(P \| Q_i)$，闭式解为各属性先验的几何加权平均。用户指定的 $\lambda_i$ 控制各属性的影响力
   - 设计动机：比直接在概率上做线性混合更有理论保证，几何平均自然地在多个分布之间做折中
   - 实现细节：每个属性模型是独立微调的 Llama2-7B，在属性标注语料上训练。推理时实时计算各模型 logits 的加权几何平均，归一化后采样

2. **Energy Function with Conflict Penalties（优化阶段）**
   - 做什么：量化文本与属性目标的偏差，并惩罚对非优化属性的干扰
   - 核心思路：$E(x) = \underbrace{\sum_i \alpha_i|C_{A_i}(x) - T_i|}_{\text{对齐项}} + \underbrace{\sum_j \beta_j|C_{A_j}(x) - C_{A_j}(x_{prev})|}_{\text{稳定性惩罚}}$。第一项衡量属性偏差，第二项防止优化某个属性时破坏其他已满足的属性
   - 设计动机：多属性优化中"按下葫芦浮起瓢"是核心难题，稳定性惩罚显式约束了非目标维度的变化

3. **Three-Stage Chain-of-Prompt Refinement（迭代改写）**
   - 做什么：通过 Feedback Agent 驱动的三阶段提示链逐步改善属性对齐
   - 核心思路：Stage 1 核心属性校准（优先修复偏差最大的维度）→ Stage 2 属性平衡调整（微调因 Stage 1 受影响的维度）→ Stage 3 全局精调（将所有属性推向目标值，直到 $E(x) \leq \tau$）
   - 设计动机：单次改写无法同时解决所有问题，分阶段从粗到细逐步逼近

### 属性覆盖
17 个子类涵盖 4 大类：emotion（joy, sadness, love, anger, fear, surprise）、style（formal, humor, poetic, sarcasm, academic）、tone（professional, casual, persuasive）、topic（courage, nature, technology）

## 实验关键数据

### 主实验（ROCStories + WritingPrompts）

| 方法 | ROC Acc↑ | ROC PPL↓ | ROC Dist-3↑ | WP Acc↑ | Toxic↓ |
|------|---------|---------|------------|--------|-------|
| COLD | 24.4 | 21.07 | 0.22 | 20.5 | 0.53 |
| BOLT | 36.5 | 17.33 | 0.38 | 32.1 | 0.76 |
| PPLM | 32.4 | 15.04 | 0.39 | 29.7 | 0.39 |
| Model Arithmetic | 87.5 | 11.08 | 0.81 | 84.2 | 0.16 |
| LLM Prompt | 89.5 | 5.37 | 0.89 | 80.0 | 0.29 |
| **C3TG** | **90.4** | **4.04** | **0.90** | **85.6** | **0.12** |

- C3TG 在属性准确率、流畅度（PPL）、多样性（Dist）、毒性上全面领先

### 消融实验

| 配置 | Acc↑ | 说明 |
|------|------|------|
| 仅 Generation Phase | ~85% | 无迭代优化 |
| + Optimization (无 conflict penalty) | ~87% | 有迭代但不保护非目标属性 |
| + Full C3TG (with conflict penalty) | **90.4%** | 完整方法 |

### 关键发现
- Generation phase 单独就能达到较好效果（~85%），但 optimization phase 再增加 5%
- Conflict penalty 对冲突属性场景贡献最大：如同时要求"joy+formal"时，无惩罚模型 formal 维度会被 joy 优化严重干扰
- 毒性降低到 0.12（vs LLM Prompt 的 0.29），说明属性控制的副产品是更安全的文本
- 通常 2-3 轮迭代即可收敛（$E(x) \leq \tau = 0.025$）
- 人工评估中，C3TG 在自然度（4.2/5）和属性一致性（4.5/5）上均领先基线
- 冲突属性对实验（如 joy+formal）中 C3TG 双属性同时达标率 82%，而 Model Arithmetic 仅 61%

## 亮点与洞察
- **"大模型生成+小模型评估"的协作范式**很高效：BERT 分类器轻量快速，提供实时属性反馈，无需修改 LLM 参数
- **维度稳定性惩罚**的设计解决了多属性优化的核心难题——显式保护非目标维度，类比于优化中的约束保持
- **17 个子类的细粒度控制**比以往只控制正/负情感或有毒/无毒要精细得多

## 局限性 / 可改进方向
- 需要为每个属性训练独立的 Llama2 属性模型和 BERT 分类器，前期准备成本高
- 迭代优化需要多次 LLM 推理，延迟和成本随迭代次数增加
- 17 个属性子类的选择和分类有一定主观性，扩展到新属性需要额外训练
- 基座模型仅使用 Llama2，未验证在更新模型（Llama3, GPT 系列）上的效果
- $\beta_j$（惩罚系数）基于实验确定的属性相关性，可能不够通用

## 相关工作与启发
- **vs PPLM**: 只在隐藏状态加梯度扰动，单属性控制；C3TG 多属性+迭代优化
- **vs Model Arithmetic**: 也做多属性融合（加减模型），Acc 接近但 PPL 高（11.08 vs 4.04），生成质量不如 C3TG
- **vs LLM Prompt**: 简单 prompt 指令控制，Acc 89.5%，但毒性更高（0.29 vs 0.12）且无冲突解决机制
- 启发：大模型生成+小模型评估的协作范式可推广到风格迁移、对话系统等其他受控生成场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 冲突感知的多属性控制框架设计完整
- 实验充分度: ⭐⭐⭐⭐ 与 10+ 基线对比，自动+人工评估，消融完整
- 写作质量: ⭐⭐⭐⭐ 公式推导严谨，流程图清晰
- 价值: ⭐⭐⭐⭐ 多属性可控生成的实用框架，冲突解决思路可迁移
