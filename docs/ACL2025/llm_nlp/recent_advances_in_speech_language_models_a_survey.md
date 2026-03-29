# Recent Advances in Speech Language Models: A Survey

**会议**: ACL 2025
**arXiv**: [2410.03751](https://arxiv.org/abs/2410.03751)
**代码**: [GitHub](https://github.com/dreamtheater123/Awesome-SpeechLM-Survey)
**领域**: 语音/LLM
**关键词**: speech language model, end-to-end speech, speech tokenizer, vocoder, survey

## 一句话总结
首篇 Speech Language Models (SpeechLMs) 综合综述，系统梳理从"ASR+LLM+TTS"级联架构到端到端语音语言模型的演进，提出按三大组件（speech tokenizer / language model / vocoder）和训练方案分类的分类体系，覆盖下游能力、评估指标、挑战与未来方向。

## 研究背景与动机

1. **领域现状**：LLM 在文本交互中表现卓越，但自然人机交互依赖语音。传统方案"ASR+LLM+TTS"三段级联虽直观，但存在三大问题：(a) 信息损失（副语言信息如音调/情感在文本中丢失）；(b) 高延迟（三段串行）；(c) 级联错误累积（ASR 错误传播到 LLM 再到 TTS）。
2. **现有痛点**：缺乏对 SpeechLM 领域的系统综述。已有综述要么聚焦传统语音技术（SLU/SSL），要么关注多模态 LLM 中的语音子集，没有以"端到端语音语言模型"为核心的全景概览。
3. **核心矛盾**：SpeechLM 快速发展（GPT-4o voice、Moshi 等），但研究社区对其架构选型、训练策略、能力边界缺乏系统认知。
4. **本文要解决什么**：提供首个 SpeechLM 领域综述，覆盖架构组件、训练方案、能力分类、评估体系。

## 方法详解

### SpeechLM 形式化定义
SpeechLM 是一个自回归基础模型，直接处理和生成语音序列 $\mathbf{M}^{\text{out}} = \text{SpeechLM}(\mathbf{M}^{\text{in}}; \theta)$，其中 $\mathbf{M}$ 可以是语音、文本或交织的多模态序列。

### 三大核心组件

1. **Speech Tokenizer（语音分词器）**
   - 做什么：将连续音频波形 → 离散 token（供 LM 自回归建模）
   - 三种类型：
     - **Semantic tokenizer**：如 HuBERT/wav2vec 2.0 + k-means 量化，提取语义特征，丢失副语言信息
     - **Acoustic tokenizer**：如 EnCodec/SoundStream，用 RVQ（残差向量量化）保留声学细节（音色/音高），但语义可能被稀释
     - **Hybrid tokenizer**：结合两者（如 SpeechTokenizer 分离 semantic 和 acoustic 层），兼顾语义+副语言
   - 关键权衡：语义 token = 高层抽象利于理解 vs 声学 token = 低层细节利于生成

2. **Language Model（语言模型主干）**
   - 做什么：在 speech token 上做 next-token prediction，核心"大脑"
   - 整合方式：
     - **直接建模**：在 speech token 上预训练 decoder-only Transformer（如 GSLM, AudioPaLM）
     - **适配已有 TextLM**：冻结 LLM + speech adapter（如 Qwen-Audio, SALMONN）
     - **联合训练**：text + speech token 混合训练（如 Spirit-LM 交织 text/speech token）
   - 多流生成：单流自回归 vs 多流并行解码（如 VALL-E 用 2 阶段：AR 生成粗 token → NAR 补全细 token）

3. **Vocoder（语音合成器）**
   - 做什么：将 LM 输出的 token/表示 → 音频波形
   - 主要方法：
     - **HiFi-GAN 系列**：直接从 mel-spectrogram/token → waveform，快速
     - **扩散模型**：如 DiffWave，质量好但慢
     - **Token decoder**：如 EnCodec decoder 直接从 RVQ token → waveform

### 训练方案分类

| 阶段 | 方法 | 代表工作 |
|------|------|---------|
| 预训练 | 语音续写（next-token prediction on speech） | GSLM, AudioLM |
| 预训练 | 语音+文本联合预训练 | Spirit-LM, SpeechGPT |
| 对齐 | ASR/TTS 多任务训练 | Whisper, Qwen-Audio |
| 对齐 | Speech-text token 交织训练 | Spectron, LauraGPT |
| 微调 | 指令微调+RLHF 对齐 | GPT-4o, 部分闭源模型 |

## 实验关键数据

### 代表性 SpeechLM 对比

| 模型 | Speech Tokenizer | LM | Vocoder | 能力 |
|------|-----------------|----|---------|----|
| GSLM | HuBERT+kmeans | Transformer | code-HiFiGAN | 语音续写 |
| AudioLM | w2v-BERT+SoundStream | Transformer | SoundStream | 语音生成 |
| VALL-E | EnCodec | AR+NAR Transformer | EnCodec dec. | 零样本 TTS |
| SpeechGPT | HuBERT+kmeans | LLaMA | code-HiFiGAN | 对话 |
| Spirit-LM | HuBERT+pitch/style | LLaMA | HiFi-GAN | 交织 Text+Speech |
| Qwen-Audio | Whisper encoder | Qwen-7B | - | 理解（无生成） |

### SpeechLM 能力分类

| 能力类别 | 具体任务 | 说明 |
|---------|---------|------|
| 语音理解 | ASR, SLU, 情感识别 | 基础能力 |
| 语音生成 | TTS, 声音克隆, 语音编辑 | 核心输出 |
| 对话交互 | 语音对话, 实时打断 | GPT-4o 级能力 |
| 副语言 | 情感表达, 说话风格控制 | 区分 SpeechLM vs ASR+LLM+TTS |
| 多语言 | 跨语言语音翻译 | 扩展能力 |

### 关键发现
- **Semantic vs Acoustic tokenizer 是核心设计选择**：理解任务偏好 semantic token，生成任务需要 acoustic token，两者融合是趋势
- **适配已有 TextLM 比从头训练更实用**：冻结 LLM + adapter 的方案在资源效率和性能间取得最优平衡
- **实时交互仍是未解难题**：当前 SpeechLM 的延迟（尤其是 AR 解码）难以满足实时对话需求
- **评估体系不统一**：不同工作用不同指标（WER/MOS/PESQ/speaker similarity 等），缺乏统一 benchmark

## 亮点与洞察
- **首个 SpeechLM 领域综述**：在 GPT-4o voice 引爆关注后及时提供系统化梳理
- **三组件分类体系清晰**：tokenizer → LM → vocoder 的分解框架使复杂架构一目了然
- **"ASR+LLM+TTS 三大缺陷"的精准概括**：信息损失、高延迟、级联错误——为 SpeechLM 存在的合理性提供了清晰论证
- **Hybrid tokenizer 方向**：分离语义和声学层（如 SpeechTokenizer）是解决"理解 vs 生成"矛盾的优雅方案

## 局限性 / 可改进方向
- **领域发展极快**：GPT-4o、Moshi 等闭源工作细节未知，综述可能遗漏
- **缺乏定量对比**：不同 SpeechLM 在统一 benchmark 上的系统对比缺失（各自用不同数据集/指标）
- **安全/伦理讨论不足**：语音克隆的滥用风险、深度伪造检测等未深入讨论
- **未涉及多模态**：仅关注 speech+text，未考虑 speech+vision 等更广泛的多模态 LM

## 相关工作与启发
- **vs Whisper (Radford et al., 2023)**：Whisper 是 encoder-only 的理解模型，SpeechLM 综述涵盖理解+生成的完整范式
- **vs AudioLM (Borsos et al., 2023)**：AudioLM 是 SpeechLM 的早期代表，本综述涵盖其后的快速演进
- **vs 多模态 LM 综述 (Zhang et al., 2024)**：多模态综述涉及 vision/audio/text 多模态，本文专注 speech modality 的深入分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 首篇 SpeechLM 综述，分类体系有价值
- 实验充分度: ⭐⭐ 纯综述无实验
- 写作质量: ⭐⭐⭐⭐ 分类体系清晰，图表丰富（尤其 Figure 4 的分类树）
- 价值: ⭐⭐⭐⭐⭐ 对快速发展的 SpeechLM 领域提供了急需的系统化参考
