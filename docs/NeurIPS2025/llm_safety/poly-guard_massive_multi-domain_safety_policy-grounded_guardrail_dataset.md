---
title: >-
  [论文解读] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset
description: >-
  [NeurIPS 2025 (Dataset & Benchmark)][LLM安全][guardrail benchmark] 提出首个**大规模、多领域、策略驱动**的安全护栏基准 Poly-Guard，从 150+ 真实行业安全策略中提取 400+ 风险类别和 1000+ 安全规则，生成 100K+ 实例覆盖 8 大安全关键领域，并系统评测 19 个护栏模型，揭示了领域特化、模型演进遗忘、模型缩放停滞、对抗脆弱性等 8 项关键发现。
tags:
  - "NeurIPS 2025 (Dataset & Benchmark)"
  - "LLM安全"
  - "guardrail benchmark"
  - "policy-grounded"
  - "multi-domain safety"
  - "adversarial attack"
  - "over-refusal"
---

# Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset

**会议**: NeurIPS 2025 (Dataset & Benchmark)  
**arXiv**: [2506.19054](https://arxiv.org/abs/2506.19054)  
**代码**: [github.com/AI-secure/PolyGuard](https://github.com/AI-secure/PolyGuard)  
**数据**: [huggingface.co/datasets/AI-Secure/PolyGuard](https://huggingface.co/datasets/AI-Secure/PolyGuard)  
**作者**: Mintong Kang, Zhaorun Chen, Chejian Xu, Jiawei Zhang, Chengquan Guo, Minzhou Pan, Ivan Revilla, Yu Sun, Bo Li  
**机构**: UIUC, UChicago, CSU, Virtue AI  
**领域**: AI Safety / 内容安全 / 护栏评测  
**关键词**: guardrail benchmark, policy-grounded, multi-domain safety, adversarial attack, over-refusal

## 一句话总结

提出首个**大规模、多领域、策略驱动**的安全护栏基准 Poly-Guard，从 150+ 真实行业安全策略中提取 400+ 风险类别和 1000+ 安全规则，生成 100K+ 实例覆盖 8 大安全关键领域，并系统评测 19 个护栏模型，揭示了领域特化、模型演进遗忘、模型缩放停滞、对抗脆弱性等 8 项关键发现。

## 研究背景与动机

- **领域现状**: LLM 在金融、法律、医疗等高风险领域的广泛部署催生了大量安全护栏模型（LlamaGuard 系列、ShieldGemma、WildGuard、Granite Guardian 等）和评测基准（ToxicChat、HarmBench、SALAD-Bench 等），但现有基准在系统性和现实对齐度上存在严重不足。

- **现有痛点**:
    1. **临时性风险分类**: 现有基准基于各组织自行设计的 ad hoc 风险分类体系，缺乏与标准化安全策略（政府法规、平台行为准则、行业伦理标准）的原则性对齐
    2. **忽视领域特异性**: 同一风险类别（如隐私泄露）在社交媒体 vs 人力资源 vs 金融领域含义截然不同，现有基准以通用领域为主，无法揭示领域间的差异
    3. **良性数据匮乏**: 缺乏高质量的"难安全"（hard safe）样本——表面涉及敏感话题但实际合规的内容，导致无法检测模型的过拒绝（over-refusal）问题
    4. **攻击场景不充分**: 现有攻击增强数据集（如 JailbreakBench）主要测试 LLM 本身的越狱漏洞，而非专门针对护栏模型的对抗鲁棒性

- **核心矛盾**: 护栏模型需要在真实的、跨领域的安全策略框架下评测才能反映实际部署效果，但目前没有这样的统一基准——既要覆盖足够多的领域和策略，又要具备足够的粒度（规则级别而非类别级别）来精确定位模型失败点。

- **本文目标**: 构建第一个与真实行业安全策略对齐的大规模多领域护栏评测基准，并在此基础上对 19 个先进护栏模型进行系统性基准评测，揭示其优势和盲点。

- **切入角度**: 从真实安全策略文档出发进行自动化提取和结构化（策略爬取 → 风险类别/安全规则两级层次提取 → 规则条件化数据生成 → 去毒化配对 → 攻格格式增强 → 对抗攻击增强），形成端到端的数据构建管线。

- **核心 idea**: 以 150+ 真实安全策略为基石，通过两级风险层次提取 + 规则条件化安全/不安全配对生成 + 对抗增强，构建首个策略驱动的跨领域护栏基准。

## 方法详解

### 整体框架

Poly-Guard 的构建遵循**两阶段管线**（图 2）：
1. **策略 → 结构化风险提取**: 自动爬取 8 个领域的 150+ 官方安全策略文档 → 用 GPT-4o 提取两级层次：高层风险类别（400+）+ 细粒度安全规则（1000+）
2. **结构化风险 → 数据集生成**: 用非安全对齐/未审查 LLM 生成规则条件化的不安全样本 → 去毒化 prompting 生成配对安全样本 → 交互格式多样化（陈述句/指令/多轮对话）→ 对抗攻击增强

最终数据集覆盖 8 个领域、100K+ 实例、400+ 风险类别、1000+ 安全规则。

### 关键设计

1. **策略驱动的两级风险层次提取**

    功能：从原始安全策略文档中自动提取结构化的风险分类体系  
    核心思路：设计两阶段 prompting 框架——第一阶段让 LLM 扮演策略分析师，从平台原始安全文档中提取原子级、可操作的行为限制规则（如"不得发布、传播儿童性虐待材料"）；第二阶段对提取的规则进行去重、语义聚类和抽象，生成"风险类别 → 安全规则"的两级层次。在此之前，先用安全策略爬取 Agent 解决策略文档格式多样（PDF/HTML/Markdown）、分布分散、结构不一致等问题  
    设计动机：现有基准仅在类别级别操作（如"仇恨言论"），粒度太粗，无法精确定位模型在哪条具体规则上失败。两级层次实现了规则级别的精细评测，便于针对性改进

2. **去毒化配对生成（Detoxification Prompting）**

    功能：为每条不安全样本生成高质量的配对安全样本，用于检测护栏模型的过拒绝  
    核心思路：采用非对称 prompting——不安全生成 prompt 要求产生明确违规内容（涵盖显性到隐性违规的谱系），去毒化 prompt 则以最小编辑原则反转意图，保留敏感上下文和语义但使内容合规。明确禁止生成免责声明或过度净化的版本，确保"难安全"样本在语言和语义上仍足够挑战性  
    设计动机：现有基准缺乏挑战性的安全样本（XSTest/OKTest 虽尝试构建但依赖人工标注、规模仅数百条），导致无法检测过拒绝。配对设计保证安全/不安全样本在话题上平衡，避免分类器学到话题偏差

3. **多策略对抗攻击增强**

    功能：评估护栏模型在对抗场景下的鲁棒性  
    核心思路：先设计三种利用常见护栏漏洞的攻击策略——① 风险类别转移（Risk Category Shifting，伪造类别变更误导模型）；② 推理干扰（Reasoning Distraction，插入无关推理任务分散注意力）；③ 指令劫持（Instruction Hijacking，利用模型的指令跟随倾向直接操纵输出）。这些策略作为种子，再用 PAIR 和 AutoDAN 对抗 prompt 优化算法迭代优化对抗后缀  
    设计动机：现有攻击增强数据集（如 JailbreakBench）面向 LLM 越狱而非护栏模型测试。本文专门针对护栏模型的判别边界设计攻击，更贴近真实部署场景

### 训练策略

Poly-Guard 本身是**评测基准**而非训练方法。数据生成使用非安全对齐 LLM 进行规则条件化生成，辅以 GPT-4o 进行风险层次提取。评测协议采用 F1、Recall、FPR 三个指标（不使用连续分数指标如 AUPRC，因为商业 API 如 Azure Content Safety、Bedrock Guardrail 不暴露置信度分值）。

## 实验关键数据

### 主实验

19 个护栏模型在 8 个领域的 F1/Recall 评测（表 1，F1 值 ×100）：

| 模型 | Social Media (Msg/Comm/Stream) | General Reg (EU/GDPR) | HR (Svc/Cust) | Finance | Law | Education | Code | Cyber |
|------|------|------|------|------|------|------|------|------|
| LlamaGuard 1 | 33.1/38.4/32.7 | 13.0/16.1 | 25.6/17.3 | 23.7 | 11.8 | 15.2 | 28.3 | 61.9 |
| LlamaGuard 2 | 49.7/60.9/55.6 | 47.8/64.4 | 52.5/52.1 | 64.6 | 62.2 | 44.7 | 51.0 | 88.0 |
| LlamaGuard 3 (1B) | 46.7/47.2/46.5 | 50.4/50.9 | 48.2/47.2 | 46.9 | 48.1 | 46.0 | 50.0 | 51.8 |
| LlamaGuard 3 (8B) | 61.2/63.3/63.5 | 37.0/32.7 | 27.4/26.8 | 49.6 | 44.2 | 28.6 | 13.8 | 81.6 |
| **MDJudge 2** | **73.7/75.3/75.9** | **64.0/81.7** | **80.4/75.6** | 76.9 | 65.6 | **77.9** | 56.5 | **89.1** |
| **WildGuard** | 76.0/74.3/76.0 | 56.6/66.4 | 77.0/71.7 | **86.5** | **76.4** | 69.4 | 55.0 | 80.2 |
| **Granite Guardian (3B)** | 71.1/70.5/71.9 | 67.9/78.2 | 80.1/78.7 | **90.4** | **80.2** | **80.0** | **63.8** | 85.0 |
| Granite Guardian (5B) | 69.5/70.3/67.4 | 63.3/80.3 | 84.6/81.6 | 85.0 | 66.8 | 75.8 | 64.0 | 87.7 |
| ShieldGemma (2B) | 4.8/5.5/4.5 | 0.0/0.0 | 8.8/4.4 | 0.0 | 0.0 | 2.2 | 16.5 | 26.8 |
| Azure Content Safety | 20.2/16.6/20.7 | 2.5/0.5 | 4.4/0.8 | 0.0 | 0.6 | 3.3 | 0.3 | 3.3 |

### 消融实验

**对抗攻击成功率（ASR）**——5 个最强护栏模型在 8 域的攻击成功率（表 2）：

| 模型 | Social Media | General Reg | HR | Finance | Law | Education | Code | Cyber | 均值 |
|------|------|------|------|------|------|------|------|------|------|
| Aegis Defensive | 0.759/0.717/0.767 | 0.559/0.884 | 0.689/0.420 | 0.555 | 0.892 | 0.435 | 0.768 | **0.677** |
| Granite Guardian (5B) | 0.989/0.992/0.994 | 0.674/0.966 | 0.993/0.842 | 0.863 | 0.997 | 0.990 | 0.912 | **0.928** |
| MDJudge 2 | 0.754/0.792/0.729 | 0.641/0.919 | 0.964/0.588 | 0.529 | 0.871 | 0.970 | 0.776 | **0.776** |
| **WildGuard** | **0.183/0.103/0.235** | **0.315/0.356** | **0.347/0.036** | **0.038** | **0.268** | **0.213** | **0.080** | **0.198** |
| LLM Guard | 0.470/0.452/0.608 | 0.781/0.991 | 0.864/0.332 | 0.388 | 0.854 | 0.990 | 0.368 | **0.645** |

**模型缩放对比**（F1 均值）：

| 对比 | 小模型 F1 | 大模型 F1 | 结论 |
|------|----------|----------|------|
| LlamaGuard 3 (1B) vs (8B) | **0.485** | 0.423 | 小模型更优 |
| Granite Guardian (3B) vs (5B) | **0.774** | 0.749 | 小模型更优 |

**LlamaGuard 系列演进**（Instagram 域 23 类风险平均 F1）：

| 版本 | 平均 F1 | Cybersecurity | Misinformation | Hate Speech |
|------|---------|--------------|----------------|------------|
| LlamaGuard 1 | 0.294 | 0.472 | 0.045 | — |
| LlamaGuard 4 | **0.605** | **0.797** | **0.692** | 0.734 (↓ vs v3 的 0.777) |

### 关键发现

1. **领域特化** (Finding 1): 护栏模型呈现明显的领域特化——Granite Guardian 在正式文体领域（HR/Finance/Education）表现突出，LLM Guard 在社交媒体领域领先，但子领域内性能趋势一致
2. **演进遗忘** (Finding 2): LlamaGuard 系列从 v1 到 v4 覆盖的风险类别更广（平均 F1 从 0.294→0.605），但在常见类别上性能不保证提升（Hate Speech 在 v4 反而下降）
3. **缩放停滞** (Finding 3): 小模型不一定弱于大模型——LlamaGuard 3 (1B) 平均 F1 高于 (8B)，Granite Guardian (3B) 高于 (5B)
4. **上下文增益** (Finding 4): 对话格式比单条指令/陈述更易被正确审核（14 个有效模型中 12-13 个在对话上 F1 更高，平均提升 >5%）
5. **对抗脆弱** (Finding 5): 所有模型对优化后的对抗攻击高度脆弱——Granite Guardian (5B) ASR 均值 92.8%，即使最优的 WildGuard 也有 19.8%
6. **严重度偏斜鲁棒性** (Finding 6): 高严重度风险类别（如 EU AI Act 禁止性 AI 实践）的对抗鲁棒性显著优于低严重度类别
7. **类别偏斜审核** (Finding 7): 风险类别间 F1 标准差普遍 >10%（如 Instagram 域 Hate Speech 平均 F1=0.715 vs Identity Misrepresentation 仅 0.273）
8. **保守偏差** (Finding 8): 模型系统性地高精度低召回（Social Media 域平均精度 0.701 vs 召回 0.479），倾向于漏检而非误报

## 亮点与洞察

1. **策略驱动的数据构建范式**: 论文开创了从真实安全策略文档到结构化评测数据的端到端管线（策略爬取 Agent → 两阶段 prompting 提取 → 规则条件化生成）。这不仅产出数据集，更提供了一个可推广到新领域/新策略的**通用框架**。

2. **"模型演进中的风险遗忘"现象**: LlamaGuard 系列在扩展覆盖范围时，常见风险类别性能反而下降——这是一个类似于连续学习中灾难性遗忘的现象，对安全模型的迭代开发有重要警示。

3. **规模不等于能力**: LlamaGuard 3 (1B) 优于 (8B) 的发现直接挑战了"更大的模型一定更安全"的假设，表明数据质量和训练策略对护栏模型可能比模型规模更重要。

4. **去毒化配对的方法论价值**: 通过最小编辑反转意图生成"难安全"样本的方法，比 XSTest 的人工构建更可扩展，且能系统性地检测过拒绝——这是现实部署中影响用户体验的关键问题。

5. **WildGuard 的防御优势**: 在所有模型中，WildGuard 以 19.8% 的平均 ASR 远领先其他模型（次优 Aegis Defensive 为 67.7%），值得深入分析其训练策略中的鲁棒性来源。

## 局限与展望

1. **文化/地域偏差**: 安全策略主要来自西方机构和全球平台，缺少非西方地区的法规和文化规范（如中国、中东地区的内容安全标准差异显著）
2. **生成模型偏差**: 使用 LLM 生成数据可能引入模型特有的语言模式偏差，尽管使用了未审查模型，生成样本的自然度和多样性仍受限于生成模型能力
3. **仅限文本模态**: 未涉及多模态安全（图像/视频/音频中的安全问题），而社交媒体和内容创作领域的风险往往是多模态的
4. **静态策略**: 安全策略具有时效性（如 EU AI Act 的持续修订），数据集需要定期更新以反映策略变化
5. **评测指标局限**: 部分商业 API 不暴露置信度分值，限制了更精细的评测（如 AUPRC）；离散判决的 F1 评测可能遮蔽模型在决策边界附近的差异

## 相关工作与启发

| 基准/数据集 | 规模 | 领域覆盖 | 策略驱动 | 良性数据 | 攻击增强 |
|------------|------|---------|---------|---------|---------|
| HarmBench / AdvBench | 中 | 通用 | ✗ | ✗ | ✗ |
| ToxicChat | 小 | 通用聊天 | ✗ | 有限 | ✗ |
| XSTest / OKTest | ≤数百 | 通用 | ✗ | ✓ (人工) | ✗ |
| AIRBench | 中 | 法规 | 部分 | ✗ | ✗ |
| CyberSecEval | 中 | 网络安全 | 部分 | ✗ | ✗ |
| GuardBench | 中 | 聚合 | ✗ | 继承 | ✗ |
| SALAD-Bench | 大 | 通用 | ✗ | ✗ | ✓ |
| **Poly-Guard** | **100K+** | **8 领域** | **✓ (150+ 策略)** | **✓ (去毒化)** | **✓ (PAIR+AutoDAN)** |

**启发方向**:
- 策略驱动的数据构建管线可直接迁移到其他安全场景（如自动驾驶安全策略、金融合规审计）
- "风险遗忘"现象提示安全模型训练应引入类似连续学习的策略，在扩展覆盖面时维持常见类别性能
- WildGuard 的对抗鲁棒性优势值得后续研究深入分析，可能成为护栏模型对抗训练的参考范本

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个将真实安全策略与大规模数据生成结合的多领域护栏基准，策略爬取 Agent + 两级风险提取的管线设计方法论创新性强
- 实验充分度: ⭐⭐⭐⭐⭐ 19 个模型 × 8 个领域的全面评测，8 项系统性发现各自有数据支撑，对抗评测和模型演进分析提供了丰富洞察
- 技术深度: ⭐⭐⭐⭐ 数据构建管线设计完整（策略爬取→规则提取→条件化生成→去毒化→对抗增强），但本质是工程管线而非新算法
- 实用价值: ⭐⭐⭐⭐⭐ 直接服务于安全护栏的开发与选型——领域特化、缩放停滞、演进遗忘等发现对工业界护栏部署有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](../../ACL2025/llm_safety/agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)
- [\[NeurIPS 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)

</div>

<!-- RELATED:END -->
