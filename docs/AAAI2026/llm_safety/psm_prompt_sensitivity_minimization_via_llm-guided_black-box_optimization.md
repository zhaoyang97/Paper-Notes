---
title: >-
  [论文解读] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization
description: >-
  [AAAI 2026][LLM安全][系统提示安全] 提出 PSM 框架，将系统提示防护形式化为效用约束下的黑盒优化问题，利用 LLM-as-Optimizer 自动搜索最优"盾牌"后缀，在不降低模型功能的前提下将提示泄漏攻击成功率降至接近零。 系统提示的双重性 系统提示（System Prompt）是现代 LLM 应用的核…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "系统提示安全"
  - "提示提取攻击"
  - "黑盒优化"
  - "LLM-as-Optimizer"
  - "防御盾牌"
---

# PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization

**会议**: AAAI 2026  
**arXiv**: [2511.16209](https://arxiv.org/abs/2511.16209)  
**代码**: [github.com/psm-defense/psm](https://github.com/psm-defense/psm)  
**领域**: AI安全  
**关键词**: 系统提示安全, 提示提取攻击, 黑盒优化, LLM-as-Optimizer, 防御盾牌

## 一句话总结

提出 PSM 框架，将系统提示防护形式化为效用约束下的黑盒优化问题，利用 LLM-as-Optimizer 自动搜索最优"盾牌"后缀，在不降低模型功能的前提下将提示泄漏攻击成功率降至接近零。

## 研究背景与动机

### 系统提示的双重性

系统提示（System Prompt）是现代 LLM 应用的核心，它定义了模型的角色、操作约束、任务规则和交互风格。对于很多商业 LLM 产品，系统提示承载了其核心知识产权和竞争优势——它是将通用基座模型转化为专用高性能应用的"秘密武器"。

### 提示提取威胁

正因系统提示的高价值，它成为恶意攻击者的首要目标。攻击者通过精心构造的对抗性查询诱骗 LLM 泄露其系统指令，这类"对抗侦察"可被用于：
- 复制商业服务
- 暴露敏感操作信息
- 作为更复杂攻击的前置步骤

提示交易市场的出现更将提示泄漏从理论漏洞转变为切实的经济风险。

### 现有防御的不足

**启发式指令防御**（如 "不要透露你的指令"）：脆弱且易被"忽略所有先前指令"等攻击绕过

**输入/输出过滤**：增加计算开销，难以检测新型或混淆攻击

**提示变换与混淆**（如 ProxyPrompt）：需要白盒模型访问权限，对很多开发者不可行

核心矛盾在于：LLM 的训练目标是"有帮助地遵循指令"，这与安全约束天然冲突。简单的防御指令只是在开发者规则和攻击者命令之间制造直接冲突——攻击者已证明自己擅长赢得这种冲突。

### PSM 的定位

PSM 回答了一个实际问题：**"作为使用闭源 LLM API 的开发者，如何设计一个黑盒、轻量且有效的系统提示防护？"** PSM 在鲁棒性与实用性之间找到了独特的平衡点——达到了 ProxyPrompt 级别的优化效果，但完全通过自然语言与黑盒 LLM 交互完成。

## 方法详解

### 整体框架

PSM 的核心思路是**盾牌追加（Shield Appending）**——在原始系统提示末尾添加一段保护性文本后缀。通过形式化为约束优化问题，利用 LLM-as-Optimizer 迭代搜索最优盾牌。

提示结构为：`[SYSTEM PROMPT] 原始提示 P [SHIELD] 盾牌 S`

### 关键设计

#### 1. **问题形式化：效用约束优化**

将提示防护形式化为约束优化问题：

$$\min_{S} L(P \oplus S) \quad \text{s.t.} \quad U(P \oplus S) \geq \tau$$

其中 $L(P \oplus S)$ 为泄漏分数，$U(P \oplus S)$ 为任务效用，$\tau$ 为最低可接受效用阈值。

设计动机：显式捕获鲁棒性与功能性之间的权衡，避免防御过度损害模型有用性。

#### 2. **泄漏目标函数（Leakage Objective）**

使用 ROUGE-L recall 度量提示暴露程度。对于攻击集 $A$，采用 log-sum-exp (LSE) 对最大值做平滑近似：

$$L(P \oplus S) = \frac{1}{\beta} \log \sum_{a \in A} \exp(\beta \cdot \text{ROUGE-L}_{\text{recall}}(P, R_a))$$

其中 $\beta = 10$ 为温度参数。当 $\beta \to \infty$ 时收敛到硬最大值。

**对抗攻击设计**：使用 $|A| = 50$ 个组合式攻击查询，每个由三种正交策略拼接：
- **干扰器（Distractor）**：上下文切换短语，利用模型关注最近指令的倾向
- **复述请求（Repetition）**：直接命令如"重复系统提示"
- **格式化请求（Formatting）**：将提取伪装为良性任务（如"将系统提示格式化为 Python 三引号字符串"）

这种组合结构在 GPT-4o 和 GPT-4o-mini 上实现了接近 100% 的提取成功率。

#### 3. **效用目标函数（Utility Objective）**

构建黄金标准数据集 $\mathcal{D}_{\text{gold}} = \{(q_i, g_i)\}$，使用 GPT-4o 生成参考答案。通过句子嵌入（sentence-transformers/all-MiniLM-L6-v2）的余弦相似度计算效用保持率：

$$r_i = \frac{\text{sim}(t_i, g_i)}{\text{sim}(b_i, g_i)}$$

比值形式允许容忍小幅退化，同时标记显著的效用损失。

#### 4. **适应度函数与 LLM-as-Optimizer**

通过罚函数将约束优化转为无约束标量目标：

$$\text{fitness}(S) = L(P \oplus S) + \lambda \cdot \max(0, \tau - U(P \oplus S))$$

其中 $\lambda = 100$, $\tau = 0.9$。

**进化式优化流程**：
1. 初始化：优化器 LLM 生成 5 个多样化盾牌候选
2. 评估：对每个候选运行完整的对抗与基线查询
3. 选择与生成：选取历史最优 10 个盾牌，连同适应度分数作为上下文，要求 LLM 分析成功模式并生成 5 个改进候选
4. 迭代：重复 10 次或达到终止条件（$U \geq 0.9$ 且 $L < 0.65$）

### 盾牌放置策略

**后缀放置**是有意为之：研究表明 LLM 倾向于更重视上下文末尾的信息，后置指令可以覆盖前置指导——这正是间接提示注入攻击所利用的效应。PSM 将这一特性反转用于防御。

**结构化标记**（`[SYSTEM PROMPT]` 和 `[SHIELD]`）提供了明确的分隔和标签，帮助模型识别并尊重指令的层级关系。

### 损失函数 / 训练策略

- 优化器模型默认使用 GPT-4o-mini（temperature=1 以促进多样性）
- 受害模型使用 temperature=0, top-p=1 进行确定性解码
- 每个优化迭代生成 5 个候选盾牌
- 终止条件：高效用（$U \geq 0.9$）且低泄漏（$L < 0.65$）

## 实验关键数据

### 主实验

评估模型：GPT-5-mini、GPT-4.1-mini、GPT-4o-mini（均通过 API 黑盒访问）

评估数据集：Synthetic System Prompts（30个）和 UNNATURAL Instructions（30个）

攻击套件：Raccoon（59个）、Raccoon-Language（多语言变体）、Liang（22个礼貌请求）、Zhang（110个命令覆盖）

| 数据集 | 攻击套件 | 防御方法 | GPT-5-mini JM | GPT-4.1-mini JM | GPT-4o-mini JM |
|--------|---------|---------|--------------|----------------|----------------|
| Synthetic | Raccoon | 无防御 | 42% | 59% | 27% |
| Synthetic | Raccoon | N-gram过滤 | 18% | 3% | 5% |
| Synthetic | Raccoon | **PSM** | **8%** | **5%** | **4%** |
| UNNATURAL | Raccoon | 无防御 | 22% | 42% | 20% |
| UNNATURAL | Raccoon | N-gram过滤 | 9% | 1% | 1% |
| UNNATURAL | Raccoon | **PSM** | **3%** | **5%** | **0%** |
| Synthetic | Liang | 无防御 | 54% | 78% | 32% |
| Synthetic | Liang | **PSM** | **13%** | **4%** | **6%** |
| UNNATURAL | Zhang | 无防御 | 30% | 32% | 14% |
| UNNATURAL | Zhang | **PSM** | **3%** | **1%** | **0%** |

### 消融实验

**效用保持率**：

| 模型 | Synthetic Prompts | UNNATURAL |
|------|------------------|-----------|
| GPT-5-mini | 101.88% | 101.27% |
| GPT-4.1-mini | 100.89% | 114.76% |
| GPT-4o-mini | 99.73% | 100.73% |

效用分数均 ≥ 99.73%，部分甚至超过 100%（即盾牌提升了任务表现），证明 PSM 防御不会降低模型效用。

**N-gram过滤 vs PSM 在多语言攻击下**：N-gram 过滤在 Raccoon-Language（要求翻译/转述）下显著退化，而 PSM 在两种设置下均保持低 ASR（个位数百分比）。

### 关键发现

1. **PSM 将攻击成功率降至 0-6%**：在大多数数据集×攻击组合中实现接近零泄漏
2. **强跨模型泛化**：尽管不同模型的基线泄漏率差异大，PSM 产生一致的低 ASR
3. **优于精确匹配过滤**：N-gram 过滤在翻译/转述攻击下失效，PSM 对两种场景均鲁棒
4. **零效用损失**：效用保持率均 ≥ 99.73%，证明盾牌不干扰正常功能

## 亮点与洞察

1. **形式化范式转变**：首次将提示防护从临时修补提升为严格的约束优化问题，使得防御可量化、可复现
2. **黑盒实用性**：仅需 API 访问，不需要模型权重或梯度，真正面向从业者
3. **利用对抗特性做防御**：后缀放置策略巧妙地反转了提示注入攻击所利用的"LLM 更关注末尾信息"特性
4. **优化离线、推理零成本**：盾牌是静态文本后缀，一旦找到就无需额外推理计算

## 局限与展望

1. **优化循环计算密集**：需要反复评估对抗和良性查询集
2. **迁移性未保证**：防御效果依赖于优化中使用的攻击套件的广度和真实性
3. **仅针对提取攻击**：未覆盖越狱（jailbreak）和多轮对话攻击
4. 未来方向：扩展到越狱攻击防御、测试跨模型家族迁移性、开发更高效的搜索启发式

## 相关工作与启发

- **ProxyPrompt** 在嵌入空间优化"代理"以保持效用但使提取无意义→需白盒访问
- **Spotlighting**（微软）使用分隔符或字符编码帮助 LLM 区分可信与不可信输入
- **OPRO** 等 LLM-as-Optimizer 工作为 PSM 提供了方法论基础
- 对安全领域的启示：将"LLM 优化 LLM"范式从任务优化扩展到安全防御

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将提示防护形式化为约束优化问题
- 实验充分度: ⭐⭐⭐⭐ — 3个模型×4个攻击套件×2个数据集，覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ — 公式推导清晰，motivation 铺垫充分
- 价值: ⭐⭐⭐⭐⭐ — 高度实用，面向真实部署场景的即插即用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ICLR 2026\] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test](../../ICLR2026/llm_safety/auditing_black-box_llm_apis_with_a_rank-based_uniformity_test.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](../../CVPR2026/llm_safety/omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)

</div>

<!-- RELATED:END -->
