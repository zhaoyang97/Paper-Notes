# DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling

**会议**: AAAI 2026  
**arXiv**: [2508.08961](https://arxiv.org/abs/2508.08961)  
**代码**: [https://github.com/lavendery/UUG](https://github.com/lavendery/UUG)  
**领域**: Audio & Speech / 语音大模型  
**关键词**: 语音大模型, 双token建模, 语音理解与生成, 语音分词器, 统一框架

## 一句话总结

提出 DualSpeechLM 框架，通过理解驱动语音分词器（USTokenizer）提取高层语义 token 作为 LLM 输入、声学 token 作为输出，在一个端到端框架中同时优化语音理解和生成能力。

## 研究背景与动机

1. **领域现状**：近年来基于文本 LLM 扩展的语音大模型（Speech LLM）蓬勃发展，包括理解类（QwenAudio、SALMONN）和生成类（SEED-TTS、UniAudio）。统一理解与生成的工作（SpeechGPT、Moshi、Mini-Omni2）也在探索中。
2. **现有痛点**：
   - **数据依赖**：由于语音和文本之间的巨大模态鸿沟，将文本 LLM 适配为统一语音 LLM 需要大量配对数据（SpeechGPT 需 70K 小时，SpiritLM 需 570K 小时）
   - **任务矛盾**：生成任务需要丰富的声学细节（韵律、情感、说话人特征），理解任务需要高层语义特征。使用同一种 token 难以兼顾两者——用声学 token 理解差，用语义 token 生成差
3. **核心矛盾**：单一 token 类型无法满足理解（偏语义）和生成（偏声学）的不同信息需求，提升一方往往损害另一方。
4. **本文要解决什么？** 在小规模数据下，实现语音理解和生成的相互增益而非相互冲突。
5. **切入角度**：从语音分词（tokenization）和语言建模（language modeling）两个维度分别提出创新——设计面向理解的分词器和双 token 建模框架。
6. **核心idea一句话**：输入用高层语义 token（USToken）降低模态对齐难度并增强理解，输出用声学 token 保留声学细节确保高质量生成，二者在统一端到端框架中联合训练。

## 方法详解

### 整体框架

DualSpeechLM 包含两个核心模块：

1. **USTokenizer**：从语音中提取与文本 LLM 语义空间对齐的理解驱动 token
2. **DualSpeechLM 主框架**：以 USToken 为输入、声学 token 为输出的双 token LLM

### 关键设计

1. **理解驱动语音分词器（USTokenizer）**：
   - 架构：预训练 Whisper 编码器 → 下采样 Encoder → 向量量化（VQ，单 codebook）→ 上采样 Decoder
   - **关键创新**：增加 Adapter 模块将 VQ 量化向量投影到冻结文本 LLM 的输入空间，通过理解任务的反向传播来优化 token 的语义内容
   - 训练损失：$\mathcal{L}_{\text{USTokenizer}} = \alpha \cdot \mathcal{L}_{\text{commit}} + \beta \cdot \mathcal{L}_{\text{Under}} + \gamma \cdot \mathcal{L}_{\text{reconstruction}}$
   - 其中理解损失 $\mathcal{L}_{\text{Under}}$ 是文本 LLM 在语音输入上的自回归生成似然。这样 token 的优化直接受文本 LLM 语义空间的指导
   - 与之前基于 SSL 量化（HuBERT）或 ASR 中间层量化（CosyVoice）的语义分词器不同，USTokenizer 显式与文本 LLM 的语义能力对齐，从而显著降低模态对齐难度

2. **双 token 建模架构**：
   - **输入侧**：USToken 提供高层语义信息，直接进入文本 LLM
   - **输出侧**：不直接输出 USToken（因缺少声学细节），而是通过 **AcousticGPT** 模块将 LLM 的隐状态转换为声学 token
   - AcousticGPT 集成在文本 LLM 内部联合训练，形成端到端流水线
   - 理解路径：语音 → USToken → LLM → 文本输出
   - 生成路径：(提示 + USToken) → LLM 预测目标 USToken → AcousticGPT 产生声学 token → 波形

3. **语义监督损失（Semantic Supervision Loss）**：
   - 在生成路径中增加对中间 USToken 预测的监督，确保 LLM 不会"遗忘"语义信息
   - 作为正则化手段，稳定双 token 联合训练

4. **条件链策略（Chain-of-Condition, CoC）**：
   - 在生成任务中，不直接从输入 USToken 一步生成声学 token，而是先让 LLM 逐步生成目标 USToken，再基于此生成声学 token
   - 类似 Chain-of-Thought 的思路但用于语音生成，提供更稳定的中间条件

### 损失函数 / 训练策略

- USTokenizer：commitment loss + 理解损失 + 重建损失
- DualSpeechLM：理解分支使用交叉熵，生成分支使用声学 token 预测损失 + 语义监督损失
- 仅使用 4.5K 小时训练数据（对比 SpiritLM 的 570K 小时）
- 基于 Phi3.5-3B，采用 LoRA 微调而非全参数微调

## 实验关键数据

### 主实验

**理解能力**（WER↓ 越低越好）：

| 模型 | LLM | 训练数据 | ASR-Clean | ASR-Other | SQA (b4↑/gs↑) |
|------|-----|---------|-----------|-----------|--------------|
| SpeechGPT | LLaMA-7B | 70K hrs | 42.73 | 78.54 | 3.58/40 |
| SpiritLM | LLaMA-7B | 570K hrs | 6.0 | 11.0 | — |
| Baseline-Acoustic | Phi3.5-3B | 4.5K hrs | 36.52 | 80.06 | 17.68/76 |
| Baseline-Semantic | Phi3.5-3B | 4.5K hrs | 5.70 | 14.32 | 42.01/85 |
| **DualSpeechLM (USToken)** | Phi3.5-3B | 4.5K hrs | **4.22** | **9.71** | **44.38/88** |

**生成能力**（TTS，SIM↑/WER↓/DNSMOS↑）：

| 模型 | Clean | Other |
|------|-------|-------|
| Baseline-Acoustic | 0.88/22.11/3.76 | 0.87/26.38/3.69 |
| Baseline-Semantic | 0.80/21.72/3.29 | 0.81/22.32/3.26 |
| **DualSpeechLM (USToken)** | **0.90/9.25/3.86** | **0.88/9.88/3.82** |

### 消融实验

**数据比例实验**（核心发现）：
- Baseline 模型：增加生成数据会恶化理解性能，增加理解数据会恶化生成性能（任务冲突）
- DualSpeechLM：增加任一方面的数据都能同时改善两方面的性能（相互增益）

**Token 类型对比**：
- DualSpeechLM + HuBERT token：理解和生成都有改善但有限
- DualSpeechLM + USToken：理解和生成均大幅提升，验证 USToken 的核心贡献

### 关键发现

- 仅用 4.5K 小时数据就超越了使用 570K 小时数据的 SpiritLM，证明 USToken 显著降低了模态对齐的数据需求
- 双 token 设计成功打破了理解-生成的零和博弈，实现了正向互促
- USToken 比 HuBERT token 在理解和生成上都显著更优

## 亮点与洞察

- 将"输入 token"和"输出 token"分离是一个简洁而深刻的设计洞察：理解和生成对信息粒度的需求本质上不同，用同一种 token 是不必要的约束
- USTokenizer 通过文本 LLM 的理解能力反向指导语音 token 的学习，是一种巧妙的跨模态知识蒸馏
- 仅使用 1%（4.5K vs 570K）的数据就超越了之前的方法，数据效率提升惊人

## 局限性 / 可改进方向

- 基于 Phi3.5-3B（较小的 LLM），未在更大模型上验证
- USTokenizer 仍依赖 Whisper 编码器的输出质量
- 声学 token 使用的是 WavTokenizer（单 codebook），多 codebook 方案可能进一步提升生成质量
- 仅评估了英文数据，多语言泛化能力未知
- CoC 策略增加了推理延迟（需要先生成 USToken 再生成声学 token）

## 相关工作与启发

- SpeechGPT / SpiritLM：使用 HuBERT token 的统一模型，但需要额外的 Mel→波形阶段
- Moshi：实时对话模型，使用多 codebook 声学 token
- Qwen2.5-Omni：使用连续 Whisper 特征而非离散 token
- 启发：双 token 思路可推广到视觉-语言模型（理解用高层视觉 token、生成用像素级 token）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 双 token 分离设计和理解驱动分词器都是清晰有力的创新
- 实验充分度: ⭐⭐⭐⭐ 理解+生成双向评估，数据比例消融说服力强
- 写作质量: ⭐⭐⭐⭐ 图示直观，思路递进清晰
- 价值: ⭐⭐⭐⭐⭐ 为统一语音大模型提供了一个优雅而高效的范式
