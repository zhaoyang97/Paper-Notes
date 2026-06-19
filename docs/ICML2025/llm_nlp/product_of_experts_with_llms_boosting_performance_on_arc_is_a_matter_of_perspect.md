---
title: >-
  [论文解读] Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective
description: >-
  [ICML 2025][LLM 其他][Product-of-Experts] 将 LLM 同时用作候选解生成器和评分器，通过基于 DFS 的搜索算法生成高概率候选解，再利用多视角增强下的 Product of Experts (PoE) 打分选出最优答案，在 ARC-AGI 公开评估集上以 71.6% 的准确率达到开源 SOTA，超越人类平均水平（60.2%），且单任务推理成本仅约 $0.02。
tags:
  - "ICML 2025"
  - "LLM 其他"
  - "Product-of-Experts"
  - "ARC-AGI"
  - "深度优先搜索"
  - "数据增强"
  - "测试时训练"
---

# Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective

**会议**: ICML 2025  
**arXiv**: [2505.07859](https://arxiv.org/abs/2505.07859)  
**代码**: [GitHub](https://github.com/da-fr/Product-of-Experts-ARC-Paper)  
**领域**: LLM/NLP  
**关键词**: Product-of-Experts, ARC-AGI, 深度优先搜索, 数据增强, 测试时训练

## 一句话总结

将 LLM 同时用作候选解生成器和评分器，通过基于 DFS 的搜索算法生成高概率候选解，再利用多视角增强下的 Product of Experts (PoE) 打分选出最优答案，在 ARC-AGI 公开评估集上以 71.6% 的准确率达到开源 SOTA，超越人类平均水平（60.2%），且单任务推理成本仅约 $0.02。

## 研究背景与动机

ARC-AGI（Abstraction and Reasoning Corpus）是 Chollet 于 2019 年提出的抽象推理基准，包含 900 个推理任务（400 训练 + 400 公开评估 + 100 私有评估），每个任务由若干输入-输出网格对（1×1 到 30×30，10 种颜色）和一个测试输入组成，要求模型从示例中推断变换规则并应用到新输入。该任务对人类简单（平均 60.2%），但对 AI 极具挑战性。

现有方法的主要瓶颈：

**规模不是万能的**：尽管 LLM 在诸多任务上表现优异，但仅靠扩大模型规模无法根本解决 ARC 上的抽象推理难题

**表征与 tokenization 阻碍性能**：大量研究表明 LLM 的很多失败模式源于数据表征和 tokenization 方式，而非推理能力本身的缺失

**自回归架构的局限**：自回归模型只能根据已生成的 token 预测下一个 token，导致在需要全局信息时可能产生高置信度的错误（如数独任务中需先内部求解整个谜题才能预测第一个格子）

**闭源方法成本高昂**：OpenAI o3 虽在 ARC-AGI 上达到 82.8%，但每个任务需要 $17 的计算成本，且缺乏可复现性

本文的核心洞察是：**模型往往已具备解决 ARC 的潜在能力，关键在于创造条件让这些能力可靠地表达出来**。通过语义保持的增强变换让模型从多个视角审视同一问题，可以有效提升推理的鲁棒性。

## 方法详解

### 整体框架

方法分三个阶段，数据增强贯穿始终：

1. **训练阶段**：在 RE-ARC 数据上进行初始微调 + 针对每个任务的测试时训练（TTT）
2. **生成阶段**：利用 DFS 在多个增强视角下系统搜索高概率候选解
3. **选择阶段**：用 Product of Experts 对候选解进行多视角评分，选出最终答案

形式化定义：给定任务 $p = ((x_i, y_i)_{i=1}^k, \hat{x})$，目标是找到 $s_p^* = \arg\max_{s \in S_p} P(s|p)$。定义一族语义保持的增强变换 $\Phi = \{\phi_1, \ldots, \phi_m\}$（满足 $P(s|p) = P(\phi_j(s)|\phi_j(p))$），包括旋转、反射、颜色置换和示例顺序重排。

### 关键设计

#### 1. 数据表征与 Tokenization

- 将词表从 120,000+ 个 token **大幅精简到仅 64 个 token**，包括字母表 A-Z/a-z（排除 I、O、i、o 以避免混淆）、数字 0-9（编码 10 种颜色）、换行符、输入/输出标记、起始/结束/填充 token
- 每个网格单元使用一个 token，不做数字合并压缩（避免 tokenization 带来的干扰）
- 在任务开头添加小段额外 token（字母表序列），作为"计算缓冲区"，模型在微调时学会利用这些 token 提升后续预测质量
- 此精简显著减小了嵌入层大小，同时消除了 BPE tokenization 中数字合并导致的歧义

#### 2. DFS 候选解生成

与多项式采样不同，使用**基于阈值的深度优先搜索 (DFS)** 系统探索解空间：

$$\mathcal{C}_{p,T} := \{s \in S_p \mid \exists \phi_j \in \Phi: \hat{P}(\phi_j(s)|\phi_j(p)) > T \}$$

- 在解的 token 序列上执行 DFS，当任何部分路径的累积概率低于阈值 $T$ 时立即剪枝
- 对所有 16 种增强（$D_8$ 对称群 × 2 组随机颜色置换和示例重排）分别运行 DFS
- 多个增强产生的相同解（模增强变换）合并为一个候选
- **缓存中间计算**加速推理：对第二个及后续增强，先将已知最优候选作为初始猜测做一次前向传递（比逐 token 生成快得多），再启动回溯搜索
- 相比多项式采样：$T=9\%$ 的 DFS 用 1/4 的推理时间（9:32h vs 39:47h）产生相近覆盖率（76.0% vs 77.3%），且假阳性数量仅为一半
- 相比 Beam Search：同等精度下 VRAM 需求仅一半（7.3GB vs 14GB），速度快 4 倍

#### 3. Product of Experts 候选排序

生成候选集后，用**多视角增强下的乘积评分**选出最优解：

$$\text{score}_{\text{agg}}(s) = \prod_{\phi_j \in \Phi} \hat{P}(\phi_j(s)|\phi_j(p))$$

关键思想：

- 这一步**不做生成采样**，只对每个候选在所有增强输入下直接计算 log-likelihood
- 使用 16 组**新随机化**的增强（与生成阶段独立）进行评分
- 乘积形式对异常值敏感：即使某个候选在大多数视角下概率较高，只要有一个视角给出低概率就会被过滤
- 最终选择 $s_p^* = \arg\max_{s \in \mathcal{C}_{p,T}} \text{score}_{\text{agg}}(s)$

等价于几何平均集成：$\bar{P}(s) = \frac{1}{Z} \prod_{j=1}^m [\hat{P}_j(s)]^{1/m}$

**理论保证（Theorem 4.1）**：PoE 集成的 KL 散度满足 $\text{KL}(P \| \bar{P}) = \frac{1}{m}\sum_{j=1}^m \text{KL}(P \| \hat{P}_j) + \log Z$，其中 $\log Z \leq 0$。这意味着 PoE 的误差不超过各增强预测器误差的平均值，且当各增强预测器之间存在分歧时（自回归架构自然导致），PoE 能提供更好的估计。

### 损失函数 / 训练策略

#### 初始微调

- **基础模型**：Mistral-NeMo-Minitron-8B-Base（性能最强），同时测试了 Llama-3.2-3B
- **训练数据**：仅使用 RE-ARC 数据集（400 个训练任务的程序化生成器产生的大量训练样本），避免"概念泄漏"
- **LoRA 配置**：rank 256，应用于所有层（含输入/输出嵌入），4-bit 量化，梯度检查点
- **梯度计算**：仅在输出网格（第2个起）和最终解上计算梯度，不在输入网格上训练
- **数据增强**：所有 $D_8$ 对称变换 + 颜色置换 + 示例重排
- **训练量**：NeMo 模型 1200 epochs（98h, 1×H100），Llama 模型 368 epochs（15h, 1×H100）

#### 测试时训练 (TTT)

- 对评估集每个任务，利用其示例对做二次微调
- LoRA rank 32，64 步，batch size 1
- 从初始微调后的模型出发
- 平均每任务 51 秒（NeMo, RTX 4090）/ 12 秒（Llama）
- TTT 单独就能将性能翻倍以上（NeMo: 18.3% → 44.5%）

## 实验关键数据

### 主实验

| 方法 | 公开评估集准确率 | 开源 |
|------|------|------|
| o1-preview | 21% | ✗ |
| Ryan Greenblatt | 42% | ✗ |
| Jeremy Berman | 58.5% | ✗ |
| GPT o3 | 82.8% | ✗ |
| 人类平均 | 60.2% | - |
| TTT | 53.5% | ✓ |
| BARC | 56.75% | ✓ |
| TTT+BARC | 62.8% | ✓ |
| **本文** | **71.6%** | ✓ |

### 消融实验

| 模型 | Baseline | +TTT | +16xAug | +PoE | +DFS |
|------|------|------|------|------|------|
| Llama-3.2-3B | 14.9% | 40.9% | 52.9% | 59.5% | 61.4% |
| NeMo-Minitron-8B | 18.3% | 44.5% | 62.5% | 67.6% | 71.6% |

采样与选择策略对比（NeMo-Minitron-8B，RTX 4090）：

| 采样方法 | 覆盖率 | 平均候选数 | 采样时间 | VRAM | PoE准确率 | 总时间 |
|------|------|------|------|------|------|------|
| Greedy | 70.8% | 6.7 | 9:39h | 7.0 GB | 67.6% | 18:52h |
| Stochastic (4x) | 77.3% | 17.6 | 39:47h | 7.0 GB | 70.8% | 58:55h |
| Beam Search (4x) | 79.0% | 34.7 | 37:36h | 14.0 GB | 71.6% | 71:39h |
| **DFS T=9% (Ours)** | **76.0%** | **9.3** | **9:32h** | **7.3 GB** | **71.6%** | **20:50h** |
| DFS T=0.5% | 83.5% | 84.7 | 80:56h | 7.3 GB | 71.8% | 134:43h |

### 关键发现

1. **PoE 显著优于其他聚合策略**：在 $T=9\%$ DFS 下，PoE 比概率平均高 5%（71.6% vs 66.6%），且在所有采样方法上均一致胜出
2. **DFS 是效率最优的采样方法**：与 Beam Search (4x) 达到相同精度（71.6%），但时间仅 1/3.4（20:50h vs 71:39h），VRAM 仅一半
3. **每个组件贡献可叠加**：TTT (+26.2%)、Aug (+18.0%)、PoE (+5.1%)、DFS (+4.0%) 逐步提升，表明方法各组件互补
4. **泛化能力良好**：ConceptARC 上 73.3%（无需调参），数独任务上 53%（远超 SOTA LLM 的 < 3%），且数独中正确解被采到时 100% 被选中
5. **成本仅为 o3 的千分之一**：$0.02/task vs $17/task

## 亮点与洞察

- **"视角即专家"**：核心创新在于将语义保持的增强变换视为不同"专家"，利用自回归模型在不同输入排列下自然产生的不一致性，构建了一个有效的集成系统——只用了一个模型
- **LLM 的双重角色**：同一个模型既做生成器又做评分器，生成阶段利用 DFS 搜索高概率路径（正向能力），评分阶段利用 log-likelihood 计算（检验能力），二者互补
- **DFS 的巧妙应用**：将 LLM 的下一 token 概率作为 DFS 的搜索启发式，阈值剪枝避免无效探索，同时其确定性保证了所有高于阈值的解都不会被遗漏
- **极简 tokenization 反直觉地有效**：从 120k+ token 词表精简到 64 个，看似信息丢失，实则通过消除 BPE 合并的歧义、减小嵌入层使微调更高效，带来性能提升
- **"计算缓冲区" token**：在输入前添加字母表序列，模型学会将其作为隐式 scratchpad，这一发现有趣且潜在可推广

## 局限与展望

1. **增强变换局限于结构化领域**：当前增强（旋转、反射、颜色置换）高度特定于网格推理任务，推广到自然语言推理等领域需设计新型语义保持变换
2. **仍依赖人工设计的增强**：增强集 $\Phi$ 需要领域知识来定义，缺乏自动发现有效增强的机制
3. **阈值 $T$ 需要调优**：DFS 的概率阈值是超参数，不同模型和训练配置下最优值不同
4. **与 o3 仍有差距**：71.6% vs 82.8%，但作者以极低成本实现了这一差距
5. **未来可探索方向**：文本领域的增强（语言风格变换等）、逻辑推理/程序合成等更广泛的结构化推理任务

## 相关工作与启发

- **TTT (Akyürek et al., 2024)**：测试时训练首次在 ARC 上大幅提升 LLM 性能，本文作为关键组件之一继承
- **BARC (Li et al., 2025)**：区分了 Induction（推断程序）和 Transduction（直接生成解）两种路径，本文使用 Transduction 路径
- **Product of Experts (Hinton, 1999, 2002)**：经典集成理论，本文将其与数据增强巧妙结合，用于结构化推理任务
- **RE-ARC (Hodel, 2024)**：提供了 400 个训练任务的程序化生成器，使大规模训练成为可能且不引入概念泄漏
- **启发**：PoE + 语义保持增强的框架具有通用潜力，只要能定义问题的对称变换群，就能将单个模型转化为有效集成

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | PoE + DFS + 增强的组合巧妙，"视角即专家"思想有趣 |
| 技术深度 | ⭐⭐⭐⭐ | 有完整理论分析，DFS 算法设计精巧 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 详尽消融、多基线对比、跨域验证（ConceptARC + Sudoku） |
| 实用性 | ⭐⭐⭐⭐⭐ | 开源、低成本（RTX 4090 即可）、可复现 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，理论与实验平衡好 |
| 总评 | ⭐⭐⭐⭐½ | 工程与理论结合典范，开源 ARC SOTA，极具实用价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](../../ECCV2024/llm_nlp/promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)
- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](../../ACL2025/llm_nlp/unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](../../ACL2025/llm_nlp/praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](../../ACL2025/llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)

</div>

<!-- RELATED:END -->
