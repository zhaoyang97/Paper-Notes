# Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video

**会议**: NeurIPS 2025  
**arXiv**: [2510.14560](https://arxiv.org/abs/2510.14560)  
**代码**: [https://zhangyl4.github.io/publications/eyes-wide-open/](https://zhangyl4.github.io/publications/eyes-wide-open/)  
**领域**: 视频理解 / 流式处理 / 第一视角  
**关键词**: streaming video, proactive response, ego-centric, just-in-time, dynamic compression, ESTP-Bench  
**作者**: Yulin Zhang (ShanghaiTech), Cheng Shi (HKU), Yang Wang (ShanghaiTech), Sibei Yang† (SYSU)

## 一句话总结
定义"第一视角流式视频主动理解"新任务——给定ego-streaming视频，AI助手在恰当时机主动回答多样化、随事件演变的问题，同时保持感知与推理的同步。提出ESTP-Bench评估框架、ESTP-F1指标，以及含数据引擎、多阶段训练和主动动态压缩的完整技术pipeline（VideoLLM-EyeWO），在ESTP-Bench上比最强baseline MiniCPM-V高11.8%。

## 背景与动机
现有Video-LLM处于**被动响应模式**——用户问了才答，且通常处理预录制的完整视频（离线）。但真实场景下的AI助手（智能穿戴设备、自动驾驶copilot等）需要处理**实时流式视频**并**主动**地在关键时刻提供信息。这一新任务需要三个关键属性：

1. **Proactive Coherence（主动一致性）**：处理多样化问题，即使答案依赖未来视频帧也能主动响应，并在相关问题间维持上下文一致性。例如对话中绿色段落在语义上依赖紫色段落的内容，需要跨时间整合过去与当前信息
2. **Just-in-Time Responsiveness（即时响应）**：基于视觉就绪度判断何时作答——太早证据不足会出错，太晚则错过帮助时机；不确定时应保持沉默，避免不必要的重复
3. **Synchronized Efficiency（同步效率）**：回答与视觉感知同步进行，不能因回答而错过新的视觉输入；随帧数增长需保持时间和内存效率

现有评估框架（StreamingBench、OVO-Bench等）无法统一评估这三者。多数在线benchmark问题类型单一、缺乏上下文连续性，几乎不评估即时响应性或同步效率。

## 核心问题
如何让Video-LLM从被动的"问了才答+离线处理"升级为主动的"自主判断何时响应+实时流式处理"，并同时满足主动一致性、即时响应和同步效率三个约束？

## 方法详解

### 1. ESTP-Bench 与 ESTP-F1 指标
基于Ego4D验证集构建benchmark，包含890个视频、100+种场景、2264个经人工验证的QA实例，涵盖14种任务类型（物体识别、属性感知、动作识别、意图预测等）。问题分为三种主动类型：
- **显式（Explicit）**：8种任务，直接利用视觉信息回答（OR/AP/TRU/OL/OSC/EOL/EOSC/AR）
- **隐式（Implicit）**：4种任务，需推理超越直接观察（OFR/IFR/NAR/TU）
- **上下文（Contextual）**：2种任务，需跨时间维持对话一致性（ORC/TRC）

每个问题标注了平均3.96个有效回答时间区间，46%的问题有上下文关联。ESTP-F1 综合衡量三个维度：
- **回答质量**：用LLM评估预测内容与GT的正确度 $\mathcal{S}_{\text{answer}}$
- **响应时机**：通过 $\mathcal{S}_{\text{time}}$ 衡量时效性，FN惩罚漏答
- **时间精度**：通过FP惩罚误报

$$\text{ESTP-F1} = \frac{2 \times \sum_{k=1}^{M} S(g_k)}{2\sum_{k=1}^{M} S(g_k) + \text{FP} + \text{FN}}$$

### 2. 数据引擎（ESTP-Gen）
利用Ego4D训练集自动生成60K单轮和20K多轮训练数据，三阶段pipeline：
- **One-to-One**：用LVLM生成字幕并提取初始QA对（单一时间区间）
- **One-to-Many**：通过RAG将每个答案扩展到多个有效时间区间
- **Many-to-Many**：将相关QA对组合为连贯的多轮问答

### 3. 多阶段训练策略
基于LLaMA3 + SigLIP架构，使用LoRA训练，渐进式赋予模型三层能力：

**Stage-1：被动区间响应**  
在有效回答区间内施加加权监督（而非简单二分类），使用线性衰减函数 $f$ 作为权重，按时间点距区间终点的距离调节监督强度，解决相邻帧高度相似导致的训练冲突

**Stage-2：主动即时响应与精准回答**  
引入第三种动作 $a_{\text{ask\_high}}$——在不确定时刻主动请求高分辨率帧。模型先学会何时请求高分辨率（$\mathcal{L}_{\text{ask\_high}}$），再基于高分辨率信息判断是否是正确响应时刻并给出精准回答（$\mathcal{L}_{\text{determine}}$），总损失为两者之和

**Stage-3：多轮QA一致性**  
仅在多轮问答数据上训练，在保持即时响应能力的同时提升上下文理解和跨轮次一致性

### 4. 主动动态压缩机制
为保证内存效率，提出两层压缩策略：
- **两级压缩**：模型主动决定压缩时机和压缩等级——当预期可能需要响应时，主动请求高分辨率输入（低压缩）；否则对历史内容施加高压缩率；响应完成后将其之前的内容进一步压缩
- **统一压缩方法**：在输入段之后插入特殊压缩token ⟨ct⟩（用EOS embedding初始化），利用causal attention机制将前序信息压缩到KV cache的紧凑表示中。平均token使用量仅为原始序列的约1/10

## 实验关键数据

### ESTP-Bench主表
| 模型 | Overall ESTP-F1 |
|------|----------------|
| LIVE (th=0.9) | 15.5 |
| MMDuet | 17.8 |
| MiniCPM-V (Polling) | 22.9 |
| Qwen2-VL (Polling) | 21.3 |
| **VideoLLM-EyeWO** | **34.7** |

- 比基线LIVE提升 **+19.2%**，比最强Polling模型MiniCPM-V提升 **+11.8%**
- 在显式任务上23.6 vs 基线9.5，隐式任务52.5 vs 25.6，上下文任务43.6 vs 20.3

### 消融实验
- ESTP-IT数据为LIVE基线带来+7.1/+6.8的提升（单轮/上下文）
- Stage-1渐进式训练解决二分类训练冲突，无需人工阈值调优
- 主动动态压缩将KV cache消耗降至基线的约 **0.11%**（9636→942 token）
- Stage-2（多轮一致性）在上下文任务上进一步+4.9

### 其他benchmark
- OVO-Bench zero-shot：32.76 vs VideoLLM-online的20.79（+57.6%）
- COIN benchmark Top-1准确率：66.0 vs 63.4（+2.6%），验证架构泛化性

## 亮点
- **"主动式流式视频理解"是范式级新任务定义**——被动→主动、离线→流式的转换
- **三个关键属性的形式化**（Proactive Coherence / Just-in-Time / Sync Efficiency）及其"不可能三角"分析
- **ESTP-Bench是首个全面评估流式主动理解的benchmark**——14种任务、3种主动类型、精确时间区间标注
- **主动请求高分辨率帧**是核心创新——模型自主决定何时需要更细粒度的视觉信息
- **动态压缩将KV cache压至基线的0.11%**，使长视频流处理成为现实
- 从任务定义、数据构建、训练策略到推理优化的**完整技术栈**

## 局限性 / 可改进方向
- Recall-Precision存在明显负相关，主动响应时机判断仍会产生误报
- NAR/TU任务因有效区间占比大而分数虚高，指标可能需要归一化
- 第一视角数据收集和精确时间标注成本很高
- 当前仅在Ego4D上验证，泛化到其他domain（如自动驾驶、安防）待探索
- 同步效率方面，APS（Actions Per Second）与性能仍存在tradeoff

## 与相关工作的对比
- **vs. LIVE/VideoLLM-Online**：共享LLaMA3+SigLIP架构和Ego4D数据源，但LIVE使用简单二分类监督且无主动高分辨率请求机制；EyeWO通过多阶段训练和动态压缩大幅超越
- **vs. MMDuet**：MMDuet追求高recall但precision极低（过度响应），EyeWO在两者间取得更好平衡
- **vs. 离线MLLM（Qwen2-VL/MiniCPM-V等）**：离线模型用Polling策略可达到一定性能，但无法实现真正的实时同步；Response-in-Last策略下性能显著下降，暴露时间定位能力弱

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 主动式流式视频理解是范式级新任务定义
- 实验充分度: ⭐⭐⭐⭐⭐ 新benchmark+新指标+多baseline对比+充分消融+跨benchmark验证
- 写作质量: ⭐⭐⭐⭐ 三属性形式化清晰，不可能三角分析有洞察力
- 价值: ⭐⭐⭐⭐⭐ 定义了Video-LLM的重要新方向，完整技术栈可落地
