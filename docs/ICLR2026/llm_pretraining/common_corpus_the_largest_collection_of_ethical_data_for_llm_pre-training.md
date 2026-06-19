---
title: >-
  [论文解读] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training
description: >-
  [ICLR2026][预训练][预训练语料] 作者发布了 **Common Corpus**——目前规模最大（约 2 万亿 token）、且**全部由公共领域或开放许可内容**构成的 LLM 预训练数据集，覆盖多语言、多领域、跨数百年时间跨度，并配套开源了一整套 OCR 纠错 / 文本分割 / PII 脱敏 / 毒性过滤工具；用它训出的 350M 与 1.2B 小模型在多语言基准上与同量级闭源数据训练的模型表现相当，证明"合规也能训出好模型"。
tags:
  - "ICLR2026"
  - "预训练"
  - "预训练语料"
  - "开放许可"
  - "版权合规"
  - "多语言"
  - "数据治理"
---

# Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training

**会议**: ICLR2026  
**OpenReview**: [0wSlFpMsGb](https://openreview.net/forum?id=0wSlFpMsGb)  
**代码**: [PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus)（数据集 + 全套清洗工具均已开源）  
**领域**: LLM 预训练 / 开放语料库  
**关键词**: 预训练语料、开放许可、版权合规、多语言、数据治理

## 一句话总结
作者发布了 **Common Corpus**——目前规模最大（约 2 万亿 token）、且**全部由公共领域或开放许可内容**构成的 LLM 预训练数据集，覆盖多语言、多领域、跨数百年时间跨度，并配套开源了一整套 OCR 纠错 / 文本分割 / PII 脱敏 / 毒性过滤工具；用它训出的 350M 与 1.2B 小模型在多语言基准上与同量级闭源数据训练的模型表现相当，证明"合规也能训出好模型"。

## 研究背景与动机
**领域现状**：从 GPT-3 开始，主流 LLM 的预训练范式就是"海量网络抓取（Common Crawl）+ 数字化书籍（Books3）"，数据量从 3000 亿 token 一路涨到 14–36 万亿 token。哪怕是所谓"小模型"（如 Qwen3 0.6B 用了 36T token），也远超 Chinchilla 最优估计的几千倍。数据规模成了刚需。

**现有痛点**：这些海量数据里**夹杂大量受版权保护或专有内容**，合法性存疑。论文列举了一连串真实"翻车"事件：NYT 起诉 OpenAI；2024 年 C4 中 45% 因 ToS 抓取限制被收紧、5% 完全禁止抓取，且受影响的恰恰是最优质来源；Books3 因 DMCA 下架；LAION 被发现含 CSAM 后下架；荷兰模型 GEITje、数学基准 MATH 也相继因法律纠纷被撤。结果是大量本应推动开放科学的研究工件突然消失，**让过往工作不可复现**，也让独立小团队损失惨重。

**核心矛盾**："数据可公开获取"≠"数据可合法用于训练"。绝大多数网络数据**缺乏足够元数据**来判断其许可状态，NLP 从业者过去只能靠"合理使用（fair use）"的转化性辩护来兜底，而这层保护正在被法律挑战快速瓦解。要做真正可审计、可在欧盟最严监管下发布的开放模型，就必须有**从源头就干净的预训练数据**。

**已有努力的不足**：2024 年起出现了若干开放许可语料（Open License Corpus、KL3M、Common Pile 等），但它们**几乎全是英文单语**，把模型能力局限在英语受众；唯一雄心勃勃的多语言开放集 C4C 还诞生在 LLM 时代之前、规模有限。

**本文目标**：造一个**同时满足"多语言 + 多领域 + 超越网络抓取 + 完全开放许可"四项的最大规模预训练语料**，并把"如何在合规前提下大规模采集、清洗、清权"的方法论沉淀下来，激励后续社区接力。

**核心 idea**：不去网络上抓"公开但许可不明"的内容，而是**系统性聚合那些本就处于公共领域或带开放许可、却在网络上"隐形"的优质来源**（政府文件、文化遗产、开放科学、开放代码等），再用专门工具把这些"脏但干净（指版权干净、但 OCR 脏）"的数据清洗到可训练标准。

## 方法详解

### 整体框架
Common Corpus 不是一个"方法 pipeline"，而是一个**按来源组织的语料库 + 配套清洗工具集**。其构建逻辑分两层：

第一层是**来源聚合**——把语料切成六大 collection（Open Government / Open Culture / Open Science / Open Code / Open Web / Open Semantic），每一类对应一批本就开放许可、但分散且利用率低的数据源；总量约 $1.998 \times 10^{12}$ token，且大部分处于公共领域。每个数据对象都带 **许可、语言、领域、来源 URL** 等元数据，使用者可按需筛子集。

第二层是**清洗与清权**——因为这些来源大量是历史扫描件、跨语种、含 PII，作者专门开发并开源了一套工具（文本分割、OCR 质量检测、OCR 纠错、PII 脱敏、去重、毒性过滤），把"版权干净但数字化质量差"的原始材料提升到可预训练标准。

最后用清洗后的子集训练 PleIAs 350M / 1.2B 两个小模型，在多语言基准上验证语料的可用性。下面分别讲清"语料怎么构成"和"为什么这套清洗是关键"。

### 关键设计

**1. 六大 collection 的来源聚合：把"开放却隐形"的优质数据系统性挖出来**

作者识别出一个"开放数据悖论"——最大宗的开放内容恰恰在网上很少可见、在主流预训练源里更少。于是他们绕开网络抓取，逐源聚合六类内容：**Open Government**（>407B token，含 Finance Commons 的 136 万份 AMF/WTO 公共领域 PDF + Legal Commons 的欧美法律行政文件，天然多语言、多模态）；**Open Culture**（13+ 语言的专著与期刊，多来自 Internet Archive ~200 万册、Delpher 等"Collections As Data"批量转储，富有历史与文体多样性，可训创意写作 / 历史断代模型）；**Open Science**（依托 OpenAlex，按 CC-BY / CC0 / CC-BY-SA 三种许可筛选论文，约 85% 为英文）；**Open Code**（283B token，来自 The Stack v1/v2，按文件扩展名 + 许可 + 质量多重过滤）；**Open Web**（Wikipedia/Wikisource 的 HTML 渲染版、YouTube Commons 的 206 万条 CC-BY 字幕、StackExchange 的 CC-BY-SA 问答）；**Open Semantic**（与 Wikimedia Deutschland 合作，把整个 Wikidata 的 RDF 三元组转成自然语言序列，覆盖 300 种语言，如 `Q41309 — P:27 — Q171150` → "Franz Liszt country of citizenship Kingdom of Hungary"，并把同一条目的多语言翻译相邻排列以促进语言对齐）。正是这种"逐源 + PDF 优先"的策略，让 Common Corpus 与 FineWeb 这类抓取语料**重叠率极低（页面 <2%、域名 <1%）**——它给生态**新增**的是真正不同的内容，而非又一份 Common Crawl 的衍生品。

**2. 严格的"开放/合规"定义与逐源清权：让"无需 fair use 即可合法发布"成为可能**

论文把"开放"用到最强含义：不仅数据可得，还提供完整 provenance，且符合 OSI 对开源 AI 的定义——"无需许可、任意用途"。为此每条数据都要确权。文化遗产部分**全部为公共领域**（版权已过期）；当无法依赖机构担保时，作者执行**自有的权利核验流程**（依据作者卒年、作品创作时间，且只采集美国/欧盟机构的内容）。Open Science 按三种 CC 许可在 OpenAlex 上过滤，Open Code 只保留 permissive 许可。这一设计直接对应前面"fair use 兜底正在瓦解"的痛点：因为源头就干净，模型可以**在欧盟最严监管下合法发布源数据**，避免 Books3/LAION 式的事后下架风险。与 KL3M（仅英文行政文书）、Common Pile（仅英文）相比，Common Corpus 是唯一**四项标准全满足**的语料（多领域 ✓ / 超越网络抓取 ✓ / 多语言 ✓ / 开放数据 ✓）。

**3. 一整套面向"脏 OCR + 多语种 + PII"的清洗工具链：把合规但低质的原料救活**

公共领域数据的代价是数字化质量差、含历史毒性内容、混杂个人信息。作者没有简单丢弃，而是开发并开源了专门工具：**Segmentext**（抗数字化噪声的文本分割语言模型）；OCR 质量检测双管——**OCRoscope**（基于 cld2、用非常规 7-gram 比例快速估计 80+ 语言的 OCR 质量，便宜但精度低）与 **OCRerrcr**（400M DeBERTa-v2 风格模型，标注 OCR 错误，精度高但更贵），两者形成"先粗筛后精修"的成本权衡；**OCRonos**（基于 Llama 3 8B，纠正断词/并词/结构破坏，对极破损内容可退化为合成重写）；**PII 脱敏**用 Microsoft Presidio 识别电话/邮箱/IBAN/IP/URL，针对电话号码格式多样导致基线召回仅 55–60% 的问题，作者补充自定义正则把准确率提到 85%，且**不删不打标，而是替换成虚构但真实感的值**，以免破坏文本格式、削弱模型对真实 PII 的处理能力；**去重**因大机构本就避免重复数字化而重复率极低，辅以 PDF 元数据过滤；**毒性过滤**用自训的多语言分类器 **Celadon**（~140M DeBERTa-v3-small，在 2M 合成标注样本上从零训练），专门清理 80 年以上历史文本中不符合现代伦理标准的内容。这套工具是整篇论文区别于"只列清单"的关键——它把"版权干净"和"质量可用"两件事同时打通。

### 损失函数 / 训练策略
语料验证侧训练了两个 Llama 架构小模型：先在 Common Corpus 代表性子集上训了一个词表 65536 的自定义 Llama 风格 tokenizer。**PleIAs 350M** 在约 1T token 的过滤子集上训练（2944 H100 小时）；**PleIAs 1.2B** 在全量 Common Corpus + 三个 epoch 的过滤子集上训练（23040 H100 小时）。评测统一用 LM Evaluation Harness。

## 实验关键数据

### 语料构成与对比

| 维度 | Common Corpus |
|------|---------------|
| 总规模 | $\approx 1.998\times10^{12}$ token（约 2T） |
| collection 数 | 6 类（Gov / Culture / Science / Code / Web / Semantic） |
| 多语言 | ≥9 种语言各有 10B+ token；Open Semantic 覆盖 300 语 |
| 许可 | 全部公共领域或开放许可，多数在公共领域 |
| 与 FineWeb 重叠 | 页面 <2%、域名 <1%（高度互补） |

四项标准对比（节选 Table 1）：

| 数据集 | 多领域 | 超越网络抓取 | 多语言 | 开放数据 |
|--------|--------|--------------|--------|----------|
| C4 | ✓ | ✗ | ✓ | ✗ |
| Dolma | ✓ | ✓ | ✗ | ✗ |
| KL3M | ✗ | ✓ | ✗ | ✓ |
| Common Pile | ✓ | ✓ | ✗ | ✓ |
| **Common Corpus** | ✓ | ✓ | ✓ | ✓ |

### 模型验证（Table 2，多语言基准，分数越高越好）

| 模型 | 规模 | MultiBLiMP | XStoryCloze | XCOPA |
|------|------|-----------|-------------|-------|
| **Ours** | 350M | **0.774** | 0.509 | 0.533 |
| Gemma 3 | 270M | 0.762 | 0.533 | 0.544 |
| XGLM | 564M | 0.711 | 0.537 | 0.550 |
| BLOOM | 560M | 0.683 | 0.532 | 0.541 |
| **Ours** | 1.2B | 0.797 | 0.526 | 0.541 |
| Gemma 3 | 1B | 0.799 | 0.594 | 0.593 |
| XGLM | 1.7B | 0.710 | 0.569 | 0.574 |
| OLMo | 1B | 0.699 | 0.517 | 0.518 |

### 关键发现
- **MultiBLiMP 上表现突出**：该基准覆盖语言最多，最能体现多语言语料价值。350M 模型在此项**反超多个 1B 级模型**（仅次于 Gemma 3 1B），1.2B 版几乎与 Gemma 3 1B 持平（0.797 vs 0.799）。
- **稳定优于同类开放数据模型**：两档模型都稳定超过同样在公开数据集上预训练的 OLMo 1B，说明合规语料并不牺牲多语言能力。
- **XStoryCloze / XCOPA 略逊于 Gemma 3**：这两项更偏英语常识推理，作者模型小且数据无英语偏置，差距可理解；但作为"完全合规"语料的代价已相当可控。
- **语料质量画像合理**：在 30 万文档上的字符重复率、类符比、空白字符等指标分布符合预期——代码因严格语法重复率偏高、空白偏低，政府数据因术语固定语言多样性偏低，均为可解释的"特征"而非缺陷。

## 亮点与洞察
- **"合规可行性证明"本身就是核心贡献**：论文不是单纯发数据，而是用一个 2T token 的实例 + 一整套开源工具，证明"在欧盟最严监管下也能做开放 LLM 研究"，把社区从"只能靠 fair use 兜底"的法律灰色地带里解放出来。
- **"开放数据悖论"是个精准洞察**：最优质的开放内容（政府/文化遗产/科学）恰恰在网上最不可见、在抓取语料里最稀缺。与 FineWeb <2% 的重叠率量化了这一点——它是**增量**而非**替代**，可与现有开放数据叠加使用。
- **PII 用"替换成虚构真实值"而非删除/打标**很巧妙：既满足 GDPR，又避免破坏文本格式、保留模型处理真实 PII 的能力，这个思路可直接迁移到任何需要脱敏又要保格式的数据治理场景。
- **OCR 双工具的成本分层**（OCRoscope 粗筛 + OCRerrcr 精修 + OCRonos 纠错）是处理大规模历史扫描件的可复用范式：先用便宜模型在全量上估质量，再把贵模型用在刀刃上。
- **Wikidata → 自然语言 + 多语言相邻排列**：把知识图谱三元组转成可训练文本、且把同一实体的多语言翻译相邻放置以促进语言对齐，是个被长期忽视的优质来源利用法。

## 局限与展望
- **作者承认的局限**：① "开放数据悖论"意味着这 2T 远未穷尽可得的开放资源，呼吁后续接力扩充；② 2T token 单独使用只够训小模型，大模型仍需更多数据；③ 语料不含指令微调数据，不能直接做任务微调（但其多样性可用于构建合规的微调集）；④ 任何清洗工具都无法 100% 准确，OCR 误差等问题仍残留（不过保留元数据，使用者可自行过滤可疑 collection）。
- **自己发现的局限**：模型验证只到 1.2B，**规模化（scaling）结论尚未验证**——"合规语料能否撑起 7B/70B 级模型"是悬而未决的关键问题。XStoryCloze/XCOPA 上落后 Gemma 3 也提示英语高质量数据的稀缺可能在某些能力上形成天花板。
- **改进思路**：未来工作明确点名"按 permissive 许可过滤的 web archive 整合"——但 C4C/CCCC/C5 都卡在许可识别难题上（一个页面里不同图片/文章各自许可不同），作者期望用"域名级筛选 + LLM 细粒度标注"组合破解，这是把语料从 2T 推向更大规模的关键路径。

## 相关工作与启发
- **vs Common Pile / KL3M / Open License Corpus**：它们同样坚持开放许可，但**几乎全是英文单语**，把模型能力锁在英语受众；Common Corpus 是同规模区间里唯一具备高多语言多样性的，是唯一同时满足四项标准的语料。
- **vs C4 / FineWeb 2 / ROOTS / DCAD-2000**：这些虽多语言多领域，但**主体是网络抓取、许可不一**；Common Corpus 走"逐源 + PDF 优先 + 全开放许可"路线，与它们重叠率 <2%，是生态的互补增量而非又一份 crawl。
- **vs C4C（2016 多语言 CC 语料）**：C4C 早于 LLM 时代、规模仅千万级网页且卡在许可识别难题；Common Corpus 在规模、领域广度和工具链成熟度上全面超越，并把 C4C 没解决的清权/清洗工程化。
- **启发**：这套"先确权、再用工具救活低质开放数据"的方法论，可迁移到任何受监管领域（医疗、法律、政务）的合规数据集构建——版权干净是前提，工具链负责把质量补齐。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是算法创新，但"最大规模全合规多语言语料 + 全套开源清洗工具"是稀缺且高价值的工程与方法贡献。
- 实验充分度: ⭐⭐⭐⭐ 语料画像 + 双模型多基准验证齐全；但模型只到 1.2B，scaling 结论留白。
- 写作质量: ⭐⭐⭐⭐⭐ provenance、清权标准、工具链交代得极清晰，"开放数据悖论"等洞察提炼到位。
- 价值: ⭐⭐⭐⭐⭐ 直接缓解开放 LLM 研究的法律不可复现困境，是社区级基础设施。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->
