---
title: >-
  [论文解读] If open source is to win, it must go public
description: >-
  [ICML 2026 Spotlight][预训练][开源 AI] 这是一篇 ICML 2026 立场论文（position paper），论点是：当前形态的"开源 AI"无法像 Linux/PyTorch 那样真正民主化 AI 访问与提供公共产品，必须嵌入到"公共 AI（Public AI）"——由政府/国家实验室/大学/非营利机构提供的算力、推理、后训练、数据基础设施——之中，开源才能赢。
tags:
  - "ICML 2026 Spotlight"
  - "预训练"
  - "开源 AI"
  - "公共 AI"
  - "基础设施"
  - "治理"
  - "数字公共产品"
---

# If open source is to win, it must go public

**会议**: ICML 2026 Spotlight  
**arXiv**: [2507.09296](https://arxiv.org/abs/2507.09296)  
**代码**: 无（立场论文）  
**领域**: 其他 / AI 治理与开源生态  
**关键词**: 开源 AI、公共 AI、基础设施、治理、数字公共产品

## 一句话总结
这是一篇 ICML 2026 立场论文（position paper），论点是：当前形态的"开源 AI"无法像 Linux/PyTorch 那样真正民主化 AI 访问与提供公共产品，必须嵌入到"公共 AI（Public AI）"——由政府/国家实验室/大学/非营利机构提供的算力、推理、后训练、数据基础设施——之中，开源才能赢。

## 研究背景与动机

**领域现状**：过去十年开源（PyTorch、HuggingFace Transformers、OpenCLIP、Megatron-LM、lm-eval-harness 等）已成为 ML 社区的文化与技术常态；社区项目（EleutherAI Pile/Pythia、LAION-5B、Stable Diffusion、OLMo、RedPajama、Marin）也在多个时间点上达到甚至超过闭源前沿实验室的能力。

**现有痛点**：作者指出"开源 = 民主化 AI"这个等式在大模型时代正在断裂——

- *预训练成本*：现代大模型需在数千张 GPU 上训练数周/数月，需要 web-scale 数据与分布式工程团队，只有少数大公司或国家级机构能承担。
- *后训练壁垒*：fine-tuning、对齐、工具集成、prompt 编排这些"让模型真正可用"的环节通常是闭源的；RLHF 数据被各平台 silo，不流回社区。
- *推理成本*：不像传统开源软件托管几乎零成本，大模型推理需要持续 GPU、编排系统、成本管理。
- *许可证脆弱*："开源 weight"≠开源——LLaMA 协议含限制性条款与可撤销条款，Meta 可随时停止发布或加更严限制；OpenAI 禁止用其输出训练竞品。
- *透明度残缺*：只放 weight 不等于放 source——训练数据、数据清洗决策、RLHF 流程、算力配置都仍不公开，外部研究者无法验证安全声明或复现行为。
- *安全与治理*：开源模型往往是"研究 artifact"而非"deployment-ready"，缺持续的 red-teaming 与对齐投入；社区贡献（评测、数据集、微调技巧）反而最终被闭源前沿实验室所"co-opt"（俘获）。
- *coding agent 案例*：开源开发者使用 Claude Code/Codex 等订阅 agent 时，prompts、迭代过程、反馈、源码片段、API key 都被私有 harness 捕获为隐性数据劳动。

**核心矛盾**：开源软件时代的前提（贡献—使用—再分发循环对所有人开放、commodity 硬件就能参与）在大模型时代失效了——大模型本质上是"非纯公共产品（impure public goods）"或"俱乐部产品（club goods）"，需要稀缺的私有补全品（算力、能源、工程团队）才能 activate。作者用一个比喻：书还是非排他的公共产品，但目录大到普通人必须雇"私人向导"才能找到书，访问就变成 club-like 的。

**本文目标**：明确提出"开源 AI 不足以民主化 AI，必须由 Public AI 补全"这一立场，并给出 Public AI 的四原则、五类已经发生的实践案例、以及对五种反方观点的回应。

**切入角度**：从公共产品经济学（Mazzucato、Reiss、Gries & Naudé）与 STS / 开源研究（Kelty、Weber、Eghbal）出发，把 AI 当作类似公路、图书馆、自来水、电力的"数字公共基础设施（Digital Public Infrastructure, DPI）"来重新框定，而不是"另一种软件库"。

**核心 idea**：用"Public AI = 公共资助 + 公共访问 + 公共问责 + 私有承诺"四原则的制度补全，去填上单纯开源在算力、后训练、推理、治理四个层面的结构性缺口。

## 方法详解

立场论文没有方法实验，但有一套清晰的论证结构。它沿着 ICML position paper 典型的八节铺开——先用 Introduction 摆出开源 AI 的张力（理想 vs 商业 vs 大模型新约束），再用 Background 回顾 ML 开源软件与开源 AI 项目的成功史，第 3 节归纳三类挑战（资源、许可证、治理），第 4 节抛出核心主张并定义四原则，第 5 节用 BLOOM/Jean Zay、LAION/JUWELS、EuroLLM/OpenEuroLLM、Public AI Inference Utility、NDIF、AVERI、SEA-HELM 等落地项目作为"Public AI 不是空想"的存在性证明，第 6 节逐条回应反方，第 7 节按受众讲落地含义，第 8 节收束在那句标题。整篇的输入是"开源 AI 当前状况 + 大模型经济结构性变化"，输出是"四原则 Public AI 制度框架 + 落地路径示例 + 对反方意见的辩护"。

### 整体框架

把全文当成一条论证链来看：它先诊断单纯开源在大模型时代为什么失效，再用 Public AI 四原则给出制度补全，最后承担起对五种最强反驳的辩论责任。贯穿三段的方法学武器是公共产品经济学——"impure public good"、"club good"、灯塔财政这些 ML 社区不太熟的概念被引进来，配上时新的实证锚点（LLaMA 4 可能是最后一代、Qwen Code 免费版关停、coding agent 捕获用户工作流），让"潜在风险"变成"已发生事件"。

### 关键设计

**1. 三维度诊断：开源 AI 在大模型时代为什么不够**

作者把单纯开源在 AI 时代的失效结构化地拆成资源、许可证、治理三个层面，让"该不该补 Public AI"从直觉口号变成可论证的命题。落到经济学上，他用"impure public goods / club goods"解释为什么 weight 开源不等于公共产品——weight 需要算力、数据、后训练、推理这些私有补全品才能 activate；再用 LLaMA 许可证可撤销、OpenAI 禁止用其输出训练竞品等具体案例坐实"open weight ≠ open source"；最后用 coding agent 这个最新案例展示"用户贡献 → 私有 harness 捕获 → 转化为隐性数据劳动"的新型 co-optation。这一整套诊断是为了回应一种常见反驳——"开源已经赢了，市场在 work，何必再加官僚机构？"——只有把结构性弱点讲到让 ML 研究者感同身受（被撤许可证、被 silo RLHF、被夹在订阅 agent 里出让数据），才接得住后面的 Public AI。

**2. Public AI 四原则定义**

第二步是把"Public AI"从模糊口号收敛成可操作的制度规范，好让第 5 节的实例和第 6 节的反驳都 ground 在四条原则上。Public Support 要求公共资金与基础设施不止覆盖预训练，还要覆盖推理、部署、后训练、数据；Public Access 要求南方国家研究者、公民技术人、Big Tech 之外的本地社区都能 build/adapt/use 有竞争力的模型；Public Accountability 要求模型与基础设施由对公众负责的机构（政府、国家实验室、公共事业、大学、非营利）来 provision/host/maintain；Private Commitments 则鼓励或要求私人主体就开放、安全、社区控制做出承诺。这四条对应"资金—访问—问责—私有约束"四个治理维度，刻意避开被简化为"政府造模型"或"再发一笔补贴"——它既能容纳 BLOOM 那种"公共算力 + 非营利"模式，也能容纳 Public AI Inference Utility 那种"协调多国捐赠算力做免费推理"模式。底层类比是 DPI（数字公共基础设施）：身份、支付、数据交换的公共栈已有先例，AI 应纳入同一范式。

**3. 五种反方观点逐条回应**

position paper 区别于 survey 的核心，是必须把可能的最强反驳显式列出来逐条回应，否则就只是宣言。View 1"市场在 work，让 OpenAI/Meta 领跑"——回应是 access ≠ governance ≠ sovereignty，LLaMA 4 可能是家族最后一代、Qwen Code 免费版 2026 年 4 月被关，证明私有访问可被单方面撤销。View 2"开源最终会赢，耐心点"——回应是当前最强开源模型（LLaMA 3.1-8B 月下载 6M）大多仍由资本充裕的私企预训练，纯非营利的 Pythia（900k）、OLMo 3-7B（170k）下载量远低，唯一例外 LAION 系（CLAP 14M/月、openCLIP 单模型 1–2M/月）恰恰靠公共超算 + 公共存储支撑，反而证明 Public AI 的必要性。View 3"OSS + 商业 hosting 已经够用"——回应是 HF/Replicate/OpenRouter 都是可撤销的商业托管、LLaMA 许可证就是脆弱性证据，而 BLOOM on Jean Zay、openCLIP on JUWELS 已经在做 underwrite。View 4"监管比公共投资更好"——回应是监管能 curb 危害但保证不了 access、可用性、平等参与，Public AI 是 proactive 地建能力与机构（如 Canada SCALE AI 同时资助监管与能力建设）。View 5"公共 AI 会低效且易被俘获"——回应是 GPS、互联网、Hubble、ERC、CERN、W3C 都是成功的公共技术基础设施，Public AI 不等于政府独占模型，可以是 Airbus for AI 这种多边混合结构，且重点不是新加预算，而是把已经在花的 AI 采购公款结构化得更服务公共利益。作者把这五条反驳排成"市场派 → 进化乐观派 → 现状满足派 → 监管派 → 公共失败派"的光谱，几乎覆盖了所有可能的政治经济立场。

## 实验关键数据

立场论文无实验，但论文中引用了若干用于支撑论点的关键数据。

### 模型下载量对比（Hugging Face, 2026 年 1 月）

| 模型 | 月下载量 | 类型 | 含义 |
|------|---------|------|------|
| LLaMA 3.1-8B | 6M | 私企开源 | 商业实验室开源模型占绝对主导 |
| EleutherAI Pythia | 900k | 纯非营利 | 比 LLaMA 小一个数量级 |
| OLMo 3-7B | 170k | 学术非营利 | 比 LLaMA 小约 35 倍 |
| LAION CLAP | 14M | 公共算力 + 非营利 | 唯一与私企持平的反例 |
| openCLIP（单变体） | 1M–2M | 公共算力 + 非营利 | 全时累计 >60M |

### Public AI 算力规模（欧洲 OpenEuroLLM 联盟）

| 维度 | 数值 | 说明 |
|------|------|------|
| 参与机构 | 20 个欧洲机构 | consortium 规模 |
| 算力配额 | >10M GPU·小时 | EuroHPC 战略资源 |
| 接入超算 | 4 个 | Leonardo / LUMI / JUPITER / MareNostrum5 |
| 当前模型质量 | 仍逊于 Qwen / DeepSeek / gpt-oss / Nemotron / Kimi | 作者坦率承认欧洲公共投入虽巨但产出尚不达 frontier |

### 关键发现

- *资本与公共投入的非对称*：私企单一模型（LLaMA 3.1-8B）的月下载量约等于整个欧洲公共 AI 投入累积出的全部成果，凸显纯公共路径的规模劣势。
- *LAION 反例的解释力*：LAION 系列证明"只要给到公共超算支撑，非营利也能在多模态领域产出全球第一梯队的开源模型"——这是作者论证 Public AI 必要性最有力的存在性证据。
- *许可证脆弱性已成事实*：LLaMA 4 据报道是 LLaMA 家族最后一代、Qwen Code 免费版在 2026 年 4 月被关停，这些都是论文写作前刚发生的事件，直接打击 "View 3：OSS + hosting 已经够用" 的立场。

## 亮点与洞察

- **"open weight ≠ open source" 这一概念的精确化**：作者明确指出公众所谓"开源 AI"在技术上准确叫"open weight AI"，并区分了 source（蓝图）与 weight（成品）——这一区分有助于澄清当下许多关于 LLaMA/Mistral/DeepSeek "是不是真开源" 的混乱讨论。
- **coding agent 是"co-optation 2.0"的典型范式**：作者用 Claude Code/Codex 这一最新（2025 末）案例，说明开源开发者贡献的不再只是代码，而是 prompts、迭代、反馈、API key、文件系统内容——这些被私有 harness 捕获后形成"隐性数据劳动"，没有清晰的删除/治理机制。这一观察对开源治理领域很有迁移价值（同样适用于 Wikipedia、StackOverflow 的内容被 LLM 训练吞掉的讨论）。
- **把 AI 当 DPI 来设计的思路**：Public AI 不发明新概念，而是把数字身份、数字支付、数字数据交换这些已有的 DPI 实践范式迁移到 AI 基础设施，这让"政府要不要做模型"这种容易陷入意识形态的问题，转化为"如何按 DPI 的工程范式来做"的可执行问题。
- **五反方光谱式回应**：五种 view 不是随便挑的，而是覆盖了"市场原教旨—进化乐观—现状满足—监管中心—公共失败"完整谱系，这种排布本身就是 position paper 的范本写法，值得学习借鉴。

## 局限性 / 可改进方向

- *实证支撑较薄*：论文是 position 性质，除了 HF 下载量与 EuroHPC 算力数字外没有定量分析；许多论断（"closed-source co-optation"、"opaque safety claim"）依赖叙事性证据，缺乏系统性数据集级别的实证。
- *"Public" 边界含糊*：四原则没有明确公共机构 vs 非营利、国际机构 vs 单一国家、政府主导 vs 多边联盟之间的优先级。Airbus for AI、CERN for AI、OpenEuroLLM 这些方案的差异与张力没有充分展开。
- *南方国家视角薄弱*：尽管 Public Access 原则提到了 global south researchers，但 5 节实例几乎全部是欧洲 + 美国超算，对非洲、拉美、东南亚（除 SEA-HELM 一例外）公共 AI 实践的覆盖不足。
- *治理与监管的关系*：作者把 Public AI 与 regulation（View 4 回应）讲得有点抽象，缺少一个具体场景（如安全审计 / 数据访问 / 模型可撤销性）来演示二者如何叠加运作。
- *改进思路*：可在后续工作中提出可量化的 "Public AI Index"（覆盖资金、访问、问责、私有承诺四维度），对各国/各项目打分，给政策制定者一个 actionable 框架；也可以补充对 Public AI 失败模式（如俘获、低效、地缘割裂）的具体防御机制设计。

## 相关工作与启发

- **vs Bommasani et al., 2024 "Considerations for governing open foundation models"**：那篇侧重"治理 open foundation model"的政策框架，本篇更进一步主张"光治理不够，必须有公共基础设施提供 substrate"——从"治"延伸到"建"。
- **vs Widder et al., 2024（关于 open source 被大公司俘获）**：本文继承其"open source 实质上在为大公司打工"的诊断，但给出了 constructive 的制度回应——Public AI 而不是仅停留在批判。
- **vs Mazzucato 的 "Entrepreneurial State" 框架**：本文把 Mazzucato 把基础研究当公共产品的论点，具体应用到 AI 大模型的算力 / 数据 / 推理 / 后训练全栈，是其在 AI 领域的延伸。
- **对 ML 研究者的启发**：（1）选择平台时考虑长期可访问性而非短期免费，（2）参与公共 AI 项目（NDIF、OpenEuroLLM、Pythia 类）以保留对模型内部的研究通道，（3）警惕 coding agent 等工具对自己工作流的隐性数据捕获。
- **对开源社区的启发**：贡献评测套件、微调技巧、数据集时，要考虑这些贡献是否最终主要 accrue 给闭源前沿实验室，并相应选择许可证 / 治理结构来防止 co-optation。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"open weight ≠ open source"、coding agent co-optation、Public AI 四原则这几条线索缝合到 ICML 语境中是新角度，但单一观点（"光开源不够，需要公共基础设施"）此前在 AI policy / DPI 圈已被多次提出。
- 实验充分度: ⭐⭐ position paper 不要求实验，但定量数据相对单薄；引用了大量正面/反面案例作为论证锚点。
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、八节布局完整、反方观点光谱化排布、用词精确（"impure public good"、"open weight" 等概念都给了定义），是 position paper 的范本。
- 价值: ⭐⭐⭐⭐ 对 ML 研究者、开源贡献者、政策制定者三方都有可操作的提示（参与公共 AI 项目、警惕 co-optation、把已花的 AI 公款结构化），潜在影响超出 ICML 学术圈本身。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](../../ECCV2024/llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[ECCV 2024\] I Can't Believe It's Not Scene Flow!](../../ECCV2024/llm_pretraining/i_canapost_believe_itaposs_not_scene_flow.md)
- [\[ICML 2026\] Incremental BPE Tokenization](incremental_bpe_tokenization.md)

</div>

<!-- RELATED:END -->
