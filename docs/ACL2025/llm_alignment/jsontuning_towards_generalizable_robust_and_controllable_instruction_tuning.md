# JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning

**会议**: ACL 2025  
**arXiv**: [2310.02953](https://arxiv.org/abs/2310.02953)  
**代码**: https://github.com/gao-xiao-bai/JsonTuning (有)  
**领域**: 对齐RLHF / 指令微调  
**关键词**: instruction tuning, JSON format, structured output, generalization, controllability

## 一句话总结
提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

## 研究背景与动机

1. **领域现状**：标准指令微调（TextTuning）将所有任务的输入输出序列化为自然语言文本，模型通过 text-to-text 生成范式学习。这种方法与 LLM 预训练的语言建模目标一致，是当前主流方案。
2. **现有痛点**：
   - **泛化差**：TextTuning 将任务元素（问题、选项、标签等）和指令混合在自然语言中，模型容易记忆特定的文本模板而非理解任务逻辑结构，导致对新任务泛化不足
   - **鲁棒性差**：自然语言的歧义性导致模型对 prompt 措辞、标签格式、选项顺序等变化敏感，换个说法性能就下降
   - **可控性差**：难以在自然语言中精确描述或强制执行特定的输出结构（如嵌套对象、数组、类型约束），模型可能生成格式不合预期的输出
3. **核心矛盾**：自然语言的灵活性既是优势也是劣势——灵活性导致训练信号模糊，模型难以区分"任务逻辑"和"文本表达习惯"，在遇到不同表述时性能波动大。
4. **本文要解决什么**：通过引入显式结构化表示，让模型聚焦于任务逻辑本身而非文本模板，同时提供精确的输出格式控制。
5. **切入角度**：JSON 天然具有键值对结构，可以明确标注每个信息片段的语义角色（字段名=语义标签），且 JSON Schema 可以精确定义输出约束（类型、嵌套结构、格式要求）。
6. **核心idea一句话**：把指令微调的 text-to-text 范式替换为 JSON-to-JSON 的 structure-to-structure 范式，让模型学"做什么"而非"怎么说"。

## 方法详解

### 整体框架
将标准指令微调从 text-to-text 改为 structure-to-structure：
- **输入** $S_I$：JSON 结构，包含 `input`（任务元素键值对 + `instruction` 指令文本）和 `output control`（JSON Schema 定义输出约束）
- **输出** $S_O$：JSON 结构，包含任务输出元素的键值对
- **训练**：用 LoRA 微调，50K Flan 2022 样本 + 10K InstructUIE 结构化任务样本，3 epochs
- **推理**：greedy decoding，输入输出都是 JSON 格式

### 关键设计

1. **统一的 JSON 输入表示**:
   - 做什么：将所有任务的输入统一为 JSON 格式，每个字段名标注语义角色
   - 核心思路：对于 MCQA 任务，TextTuning 输入为 `"Answer the following question: Who is CEO? (A) Sundar..."`, JsonTuning 输入为 `{"input": {"question": "Who is CEO?", "options": "(A) Sundar...", "instruction": "..."}, "output control": {"answer": {"type": "string"}}}`。字段名 `question`, `options` 显式标注了每个文本片段的语义角色
   - 设计动机：消除自然语言中任务元素和指令的混淆——模型不需要从连续文本中"猜"哪部分是问题、哪部分是选项，而是直接从结构中获取。这降低了对特定 prompt 模板的依赖，提升泛化性

2. **Output Control（JSON Schema 控制信息）**:
   - 做什么：在输入中用 JSON Schema 显式描述期望的输出结构——字段名、类型（string/array/object）、嵌套结构
   - 核心思路：以语言检测任务为例，control 信息为 `{"language": {"type": "string"}, "probability scores": {"type": "object", "properties": {"French": {"type": "number"}, ...}}}`，精确定义输出应包含哪些字段、每个字段什么类型
   - 设计动机：(1) 提升可控性：JSON Schema 比自然语言描述更精确，模型可以学习 Schema→结构化输出的映射；(2) 提升泛化到新结构：模型学会了 Schema 中基本组件（string/array/object）的组合规则，遇到训练中未见过的复杂结构也能正确生成；(3) 增加训练一致性：不同任务可能需要不同输出结构，Control 信息统一了"告诉模型输出什么格式"的方式

3. **结构化任务数据增强**:
   - 做什么：在 Flan 2022 的基础上加入 InstructUIE 的信息抽取（NER + RE）任务，用于训练时学习复杂输出结构
   - 核心思路：Flan 2022 的输出几乎都是纯文本（string），缺乏数组和嵌套对象。加入 NER/RE 任务后，模型学习生成包含 array 和 object 的复杂 JSON 输出。评估时用未见过的 Event Extraction（EE）任务测试结构泛化能力
   - 设计动机：如果只训练简单结构，模型无法泛化到复杂结构。通过在训练中引入适量复杂结构任务，模型能学会基本结构组件的组合规则

### 训练策略
- LoRA 微调，peak lr = 1e-3，AdamW + linear decay
- 最大长度 2048 token
- 每个任务使用多个 prompt（NER/RE 各 10 个手工构造的 prompt），增加训练多样性
- JsonTuning 和 TextTuning 在完全相同的数据上训练，仅数据格式不同，保证公平对比

## 实验关键数据

### 主实验
7 个模型 × 6 类任务的零样本泛化结果：

| 模型 | TextTuning Avg | JsonTuning Avg | 提升 |
|------|---------------|---------------|------|
| Falcon-7B | 12.37 | **17.64** | +5.27 |
| Mistral-7B | 30.95 | **35.74** | +4.79 |
| LLaMA-7B | 22.80 | **27.06** | +4.26 |
| LLaMA-13B | 27.79 | **31.10** | +3.31 |
| LLaMA2-7B | 26.29 | **29.19** | +2.90 |
| LLaMA2-13B | 30.27 | **33.47** | +3.20 |
| LLaMA3-8B | 37.01 | **41.96** | +4.95 |
| **所有模型平均** | **26.78** | **30.88** | **+4.10** |

结构化任务提升尤为显著：NER 平均 37.51→45.28，EE（未参与训练的复杂结构任务）从几乎为 0 提升到 4.83/10.17。

### 消融实验（鲁棒性与可控性）

| 测试维度 | TextTuning | JsonTuning | 说明 |
|---------|-----------|-----------|------|
| Prompt 鲁棒性 | 高方差 | **低方差+高均值** | 10 种 prompt 下 Json 模型性能波动小 |
| 标签鲁棒性 (MMLU) | 未见标签大幅下降 | **未见标签仍保持** | 换 {W,X,Y,Z} 或 {$,€,£,¥} 仍稳定 |
| EE 结构泛化 | Text 几乎 0 分 | **Json 可生成有效结构** | 仅训练 NER+RE 就能泛化到更复杂的 EE |
| 输出格式控制 | 无法精确控制 | **可通过 Schema 控制** | 语言检测+概率分数的嵌套输出案例 |

### 关键发现
- **弱模型获益更大**：Falcon-7B 提升 +5.27，说明 JsonTuning 对能力较弱的模型帮助更大——结构化格式降低了理解任务指令的难度
- **结构化任务提升最显著**：NER +7.77，EE 从近 0 到可用，说明 JSON 格式天然适合需要结构化输出的任务
- **鲁棒性提升不是代价而是额外收益**：JsonTuning 在提升平均性能的同时降低了跨 prompt 的方差
- **Control 信息是可控性的关键**：去掉 output control 后，模型在复杂结构任务上性能显著下降
- **JsonTuning 的 token 开销可接受**：JSON 格式确实比纯文本多出一些 token（括号、引号、字段名），但性能提升远超这点开销

## 亮点与洞察
- **极简但有效的改动**：不改模型架构、不改训练算法，只改数据格式，就能在 7 个模型上一致提升——这是最佳的"低投入高回报"research
- **与 LLM 生态趋势高度一致**：OpenAI、Anthropic 都在推 structured output / JSON mode，本文从学术角度验证了这个方向的合理性
- **Output Control 的 JSON Schema 设计**：将输出约束显式编码到输入中，让模型不仅知道"做什么任务"还知道"输出什么格式"，是可复用的设计模式
- **弱模型获益更大的发现**：暗示结构化格式降低了指令理解的认知负荷，对资源受限场景（小模型部署）有实际价值

## 局限性 / 可改进方向
- **JSON 的 token 开销**：括号、引号、字段名等额外 token 增加输入长度，对上下文窗口有限的模型可能是问题
- **JSON Schema 需要手工设计**：每个任务需要设计合理的 JSON Schema，增加了数据准备成本
- **仅测试 LoRA 微调**：未验证全量微调或更大规模模型（70B+）上的效果
- **训练数据有限**：仅用 60K 样本，更大规模数据下 JsonTuning 的优势是否保持未知
- **开放域生成未深入评估**：主要验证结构化任务和选择题，对自由文本生成（如对话、创意写作）的效果未充分测试

## 相关工作与启发
- **vs TextTuning (Flan, T0)**：传统 text-to-text 指令微调，本文证明只改数据格式就能一致提升，说明训练数据的表示形式被低估了
- **vs Structured Output (OpenAI JSON mode)**：工业界推动 JSON 输出模式是通过推理时约束（constrained decoding），本文从训练端解决——让模型在微调阶段就学习结构化思维
- **vs InstructUIE**：InstructUIE 为 IE 任务设计了特定的指令格式，本文将结构化思路推广到所有任务类型

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路简洁、直觉合理，虽然 JSON 格式不算技术创新，但系统性验证其优势是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型 × 6 类任务 × 泛化/鲁棒/可控三维度评估，对比非常全面
- 写作质量: ⭐⭐⭐⭐ 对比设置公平清晰，案例直观
- 价值: ⭐⭐⭐⭐ 即插即用的数据格式改进，与行业趋势一致，实用性强
