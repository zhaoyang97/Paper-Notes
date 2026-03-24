# CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation

**会议**: ICLR 2026  
**arXiv**: [2602.20170](https://arxiv.org/abs/2602.20170)  
**代码**: [https://github.com/selectstar-ai/CAGE-paper](https://github.com/selectstar-ai/CAGE-paper)  
**领域**: AI安全 / Red-Teaming  
**关键词**: red-teaming, cultural adaptation, semantic mold, multilingual safety, benchmark generation  

## 一句话总结
提出 CAGE 框架，通过 Semantic Mold（语义模具）将红队攻击 prompt 的对抗结构与文化内容解耦，能系统性地将英语红队基准适配到不同文化语境中，生成的文化扎根 prompt 比直接翻译的 ASR 显著更高。

## 研究背景与动机
1. **领域现状**：LLM 安全评估主要依赖英语红队基准（AdvBench、HarmBench 等），跨语言评估通常通过直接翻译实现。但不同文化中的刻板印象、社会规范、法律框架差异巨大。
2. **现有痛点**：直接翻译丢失文化特异性——焚烧国旗在美国是言论自由，在韩国是违法犯罪；某些种族歧视在英语语境中有意义但在韩语语境中不存在。模板化生成（KoBBQ 等）语义多样性有限；从头构建原生数据集（KorNAT）成本极高。
3. **核心矛盾**：文化保真度与可扩展性的 trade-off——要么高保真低规模（人工写），要么高规模低保真（机器翻译），缺少两全其美的方案。
4. **本文要解决什么？** 如何在保留英语红队 prompt 的对抗结构的同时，注入目标文化的具体内容？
5. **切入角度**：将 prompt 的"对抗意图"（做什么恶事）和"文化内容"（用什么具体实体/场景）视为可分离的两个维度。
6. **核心idea一句话**：用 Semantic Mold 将 prompt 拆解为 slot-tagged 结构（保留攻击框架），再填入目标文化的合法/社会内容（实现文化落地）。

## 方法详解

### 整体框架
三步流水线：(1) Seed 收集——从 6 个英语红队数据集收集 prompt，用 6 个 frontier LLM 一致投票筛选并映射到三级风险分类体系 (2) Refine-with-Slot——将 prompt 重写为 slot-tagged 语义模具 (3) Translate-with-Context——用目标文化内容填充 slot 生成本地化 prompt。

### 关键设计

1. **三级风险分类体系**：
   - 做什么：定义 5 个风险领域 → 12 个类别 → 53 个类型的层次分类。
   - 内容：Toxic Contents、Unfair Representation、Misinformation Harms、Info & Safety Harms、Malicious Use。
   - 设计动机：在现有分类（Weidinger 等）基础上精细化，使 slot schema 可以按类别定制。

2. **Semantic Mold（语义模具）**：
   - 做什么：定义每个风险类别表达有害场景所需的最小语义元素。
   - 核心思路：为每个类别定义 Required Slots（必填）和 Optional Slots（选填），如 Illegal Activities 需要 [Act], [Target], [Method/Approach]；Bias and Hate 需要 [Target/Group], [Attribute]。Seed prompt 被 LLM 重写为 slot-tagged 形式（如"There are rumors that [Fake Event] in [Time]"），保留修辞结构但抽象化具体内容。
   - 设计动机：Mold 定义的是"应该包含什么语义"而非"句子应该怎么写"，确保生成的 prompt 在语言多样性和攻击保真度之间取得平衡。

3. **文化内容库构建**：
   - 做什么：为目标文化（如韩国）收集填充 slot 的本地化内容。
   - 双轨策略：(a) Taxonomy-Driven——从法律、判例、执行条例中获取客观类别内容 (b) Trend-Driven——从新闻门户和在线社区自动抓取热门话题和关键词。
   - 质量控制：content 经过二分类通过/不通过过滤，而非逐条人工撰写。

4. **多模型一致投票 + 人工验证**：
   - 做什么：确保 seed prompt 分类准确。
   - 6 个 frontier 模型（GPT-4.1, Claude 3.5/4, Gemini 2.5 Pro, Llama 3.3, Qwen 2.5）独立分类，仅保留一致结果，再人工验证。

## 实验关键数据

### 主实验：KorSET vs 翻译基线 ASR
| 分类 | 攻击方法 | Llama-3.1-8B | Qwen2.5-7B | gemma2-9B | EXAONE-3.5-7.8B | gemma3-12B |
|------|---------|-------------|------------|-----------|----------------|------------|
| Toxic Language | Direct | 32.8 | 11.9 | 27.2 | 27.0 | 13.5 |
| | GPTFuzzer | 35.3 | 39.3 | 28.8 | 41.8 | 39.5 |
| Misinformation | Direct | 48.8 | 21.2 | 20.9 | 13.9 | 12.3 |
| | GPTFuzzer | 47.4 | 56.3 | 56.3 | 50.4 | 42.6 |
| Malicious Use | Direct | 34.7 | 10.3 | 5.8 | 9.5 | 9.2 |
| | AutoDAN | 50.4 | 27.2 | 37.3 | 36.1 | 27.5 |

### CAGE vs 直接翻译对比
CAGE 生成的文化扎根 prompt 的 ASR 显著高于直接翻译的英语基准（详见论文 Table 4-5），验证了文化适配的必要性。EXAONE（韩语优化模型）在 KorSET 上仍被攻破，说明语言能力不等于安全能力。

### 关键发现
- CAGE 在韩语（KorSET）上生成 7,161 个 prompt，覆盖 12 个类别 53 个类型。
- 直接翻译的 prompt ASR 通常低于 CAGE prompt 15-30 pp，证实了"文化天真"基准的盲点。
- GPTFuzzer 在 CAGE prompt 上表现最强，GCG 在韩语上效果欠佳（可能因为梯度优化在非英语 token 上失效）。
- 框架成功迁移到高棉语（极低资源语言），证明跨文化可扩展性。

## 亮点与洞察
- **Semantic Mold 的核心洞察**：将红队 prompt 拆解为"攻击结构"和"文化填充"两个正交维度。这不仅让框架可扩展到任意文化，也让研究者能精确控制变量——在相同攻击结构下对比不同文化填充的效果。
- **"文化天真"的定量证据**：首次系统性地证明直接翻译的安全基准会低估模型在非英语语境下的脆弱性，这对全球 LLM 部署的安全评估有重要政策意义。
- **低资源语言扩展**：CAGE 成功应用于高棉语（Khmer），证明框架不依赖目标语言的丰富资源。

## 局限性 / 可改进方向
- 文化内容库的质量仍依赖于目标文化的可用信息源——极低资源文化可能缺少法律文本和新闻数据。
- Semantic Mold 由人类专家定义 slot schema，这一步不可避免地引入主观性。
- 仅在韩语和高棉语上验证，更多语言/文化的实验仍需扩展。
- 生成的 prompt 可能被滥用——论文主要关注评估工具的构建，对使用限制的讨论较少。

## 相关工作与启发
- **vs XSafety/PolyGuardPrompts（直接翻译）**：直接翻译丢失文化语境，ASR 偏低。CAGE 通过 Semantic Mold 保留攻击结构同时替换文化内容。
- **vs KoBBQ/MBBQ（模板化适配）**：模板化方法受限于预定义实体列表，表达多样性不足。CAGE 的 Mold 定义语义而非语法，生成更自然多样的 prompt。
- **vs Align Once (MLC)**：MLC 从训练侧解决多语言安全，CAGE 从评估侧解决多语言安全。两者互补——用 CAGE 评估 MLC 对齐后的模型在文化扎根场景下是否仍然安全。

## 评分
- 新颖性: ⭐⭐⭐⭐ Semantic Mold 概念简洁有力，首次系统化跨文化红队基准生成
- 实验充分度: ⭐⭐⭐⭐ 5 个模型 × 5 种攻击方法 × 12 个风险类别，规模可观，但缺少更多语言验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，分类体系详尽
- 价值: ⭐⭐⭐⭐ 填补了跨文化安全评估的重要空白，对全球 LLM 部署有直接政策意义
