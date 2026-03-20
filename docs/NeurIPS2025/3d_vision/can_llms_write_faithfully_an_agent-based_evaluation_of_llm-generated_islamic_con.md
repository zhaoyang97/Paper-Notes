# Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content

**会议**: NEURIPS2025 (MusIML Workshop)  
**arXiv**: [2510.24438](https://arxiv.org/abs/2510.24438)  
**代码**: 待确认  
**领域**: LLM Evaluation / Religious NLP  
**关键词**: LLM评估, 伊斯兰内容生成, 双Agent框架, 引用验证, 高风险领域生成  

## 一句话总结

提出双Agent（定量+定性）评估框架，从神学准确性、引用完整性和文体恰当性三个维度系统评估 GPT-4o、Ansari AI 和 Fanar 在伊斯兰内容生成任务上的忠实度，发现即使最优模型也在引用可靠性上存在显著不足。

## 研究背景与动机

1. **高风险领域的特殊需求**：伊斯兰内容生成对神学准确性、引用归属和语气恰当性要求极高，细微错误（如错引古兰经经文、误引圣训）可能传播错误信息并造成精神伤害。
2. **传统指标的局限**：BLEU、ROUGE 等表面重叠指标无法衡量教义忠实度、引用完整性或神学正确性。
3. **现有评估缺口**：医学、法律等高风险领域已有专门评估流水线，但宗教领域几乎空白，现有伊斯兰聊天机器人（Ansari AI、Fanar）仅在通用阿拉伯语 benchmark 上评测，未涉及神学层面。
4. **基础设施不足**：大量经典伊斯兰文本仍以非结构化 PDF 或扫描图像存在，阻碍了计算化利用。
5. **跨领域借鉴**：法律领域（Mata v. Avianca 案）暴露虚构引用问题，医学领域 50-90% 回复未被引用充分支撑，新闻领域 CNET 77 篇 AI 文章中 41 篇需纠错——宗教领域面临类似甚至更严峻的风险。
6. **核心研究问题**：当前 LLM 能否生成在神学上准确、引用正确、语气恰当的伊斯兰内容？如何系统化评估？

## 方法详解

### 整体框架

提出 **双Agent评估框架**（Dual-Agent Framework），包含定量评估Agent和定性比较Agent，两者共享引用验证工具链，从不同视角对 LLM 输出进行全面评估。

### 三大设计模块

**1. 提示收集与响应获取**

- 从 5 个权威伊斯兰博客平台（The Thinking Muslim、IslamOnline、Yaqeen Institute 等）收集 50 个提示，由知名伊斯兰学者撰写的博文标题构成
- 覆盖 5 个领域：教法学(Fiqh)、古兰经注释(Tafsir)、圣训学(Ulum al-Hadith)、神学(Aqidah)、精神行为(Adab)
- 分别发送给 GPT-4o、Ansari AI、Fanar，获得 150 篇回复

**2. 定量评估Agent**

- 基于 OpenAI o3 推理模型，配备三个验证工具：Qur'an Ayah（古兰经经文检索）、Internet Search、Internet Extract
- 将每篇文章分为引言/正文/结论三段，在 **6 个维度**上打分（1-5分）：
  - **文体与结构**（4维）：结构连贯性(Structure)、主题聚焦(Theme)、清晰度(Clarity)、原创性(Originality)
  - **伊斯兰内容**（2维）：伊斯兰准确性(Islamic Accuracy)、引用与来源使用(Citation)
- 检测到引用时自动检索验证，返回四级标注：confirmed / partially confirmed / unverified / refuted
- 对未完全确认的引用进行扣分

**3. 定性比较Agent**

- 同时处理三个模型的回复（用 XML 标签 `<R1>/<R2>/<R3>` 分隔），进行 side-by-side 对比
- 在 5 个维度评估：Clarity & Structure、Islamic Accuracy、Tone & Appropriateness、Depth & Originality、Comparative Reflection
- 每个维度识别最强/最弱回复，用具体文本摘录作为依据
- 使用与定量Agent相同的验证工具链，确保一致性

### 损失/评分设计

- 定量维度采用 1-5 分制，引用验证结果直接影响 Islamic Accuracy 和 Citation 分数
- 定性维度采用 Best/Worst 投票制，每个维度每个提示对三个模型做二元判定
- 两种评估的对齐提供收敛效度(convergent validity)证据

## 实验

### 主要定量结果

| 模型 | 总均分 | 标准差 | Structure | Theme | Clarity | Originality | Islamic Accuracy | Citation |
|------|--------|--------|-----------|-------|---------|-------------|------------------|----------|
| GPT-4o | **3.90** | 0.589 | 4.16 | **4.43** | 4.10 | 3.10 | **3.93** | **3.38** |
| Ansari AI | 3.79 | — | — | — | — | — | 3.68 | 3.32 |
| Fanar | 3.04 | 0.923 | — | — | — | 2.73 | 2.76 | 1.82 |

### 定性对比结果（Best/Worst 投票，满分各 200）

| 模型 | Best 总数 | Worst 总数 | 最强维度 |
|------|-----------|------------|----------|
| Ansari AI | **116** | 3 | Clarity & Structure (41), Islamic Accuracy (42) |
| GPT-4o | 84 | 4 | Tone & Appropriateness (48) |
| Fanar | 0 | 193 | 全维度最弱 |

### 关键发现

1. **GPT-4o 定量最优**：总均分 3.90/5，在结构、主题和伊斯兰准确性上领先，且方差最低（std=0.589），表现稳定
2. **Ansari AI 定性最优**：Best 投票 116/200，在清晰度和宗教忠实度方面表现突出，体现了领域适配的优势
3. **Fanar 整体落后但有创新**：9B 参数和 4096 token 上下文窗口限制了其推理能力，但其形态学分词器、区域特定数据集和伊斯兰 RAG 管道是有价值的创新
4. **引用问题普遍存在**：即使最好的模型（GPT-4o Citation=3.38/5），引用准确性仍是最大短板——这在信仰敏感写作中是核心要求
5. **模型规模影响显著**：GPT-4o（128K上下文）vs Fanar（4096上下文）的性能差距与模型规模和上下文长度高度相关

## 亮点

- **首个系统化的伊斯兰内容忠实度评估**：填补了宗教领域 LLM 评估的空白，框架设计可迁移至医学、法律等其他高风险领域
- **双Agent互补设计精巧**：定量Agent提供可比数字，定性Agent捕捉语气、修辞等微妙差异，两者使用相同工具链确保一致性
- **引用验证工具链实用**：自动检索古兰经经文和圣训并进行四级标注（confirmed/partially/unverified/refuted），具有实际应用价值
- **实验设计严谨**：50 个提示覆盖 5 个伊斯兰知识领域，采用盲审协议减少偏差，并引入人工审查作为 sanity check

## 局限性

1. **评估器偏差**：定量和定性Agent均基于 OpenAI 模型，存在同族偏差风险；未来需引入 Claude、Gemini 等异构评估器做交叉验证
2. **规模有限**：仅 50 个提示，未覆盖不同教法学派(madhahib)、边缘案例及当代法学议题
3. **语言单一**：仅评估英文回复，未在阿拉伯语（Fanar 的主要语言）上进行评测，可能对 Fanar 不公平
4. **缺乏多专家验证**：仅一位人工审查员，未形成 3-5 位学者的专家组共识
5. **领域分类不严谨**：部分提示的领域归属可能存在交叉

## 相关工作

- **高风险领域评估**：法律(LEGAL-BERT, LegalBench)、医学(SourceCheckup)、新闻领域已有幻觉和引用验证研究，本文将其范式扩展至宗教领域
- **伊斯兰 NLP**：AraBERT、Qur'anQA 等推动了阿拉伯语理解，Ansari AI 和 Fanar 是代表性伊斯兰聊天机器人，但评估局限于通用 benchmark
- **Agent-based 评估**：结合 RAG、CoT、多Agent协作（LangChain/CrewAI/CamelAI）的工具增强方法在通用任务上提升了可验证性，本文首次将其应用于神学验证
- **数据基础设施**：Usul.ai、SHARIAsource、Shamela、OpenITI 等平台提供了机器可读的伊斯兰法律数据，但尚未系统集成到 LLM 评估流水线中

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个系统化的伊斯兰内容忠实度评估框架，双Agent设计有创意
- 实验充分度: ⭐⭐⭐ — 50 个提示规模偏小，单一语言和单一人工审查限制了结论强度
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机充分，与跨领域工作的联系做得好
- 价值: ⭐⭐⭐⭐ — 框架可迁移至其他高风险领域，问题定义和评估维度设计有参考价值
