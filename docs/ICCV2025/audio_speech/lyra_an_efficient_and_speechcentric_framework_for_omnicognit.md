# Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

**会议**: ICCV 2025  
**arXiv**: [2412.09501](https://arxiv.org/abs/2412.09501)  
**代码**: [https://github.com/dvlab-research/Lyra](https://github.com/dvlab-research/Lyra)  
**领域**: 多模态VLM / 全模态理解 / 语音模态  
**关键词**: omni-cognition, speech-centric, multi-modality LoRA, latent cross-modality regularizer, long speech, token extraction  

## 一句话总结
提出Lyra，一个以语音为中心的全模态MLLM框架，通过三大核心组件（DTW-based跨模态正则化器、多模态LoRA、Latent多模态提取器）和首个12K长语音SFT数据集，在仅用2.7M数据和少量训练的情况下，同时在视觉-语言、视觉-语音、语音-语言benchmark上达到SOTA，并能处理长达2小时的语音输入。

## 研究背景与动机
1. **领域现状**：现有omni-model（如VITA、EMOVA、Intern-Omni）虽开始探索多模态融合，但语音模态被严重低估——它们的语音评估局限于speech-text ASR指标（如LibriSpeech WER），忽略了语音与视觉等其他模态的交互。
2. **现有痛点**：(1) 语音token与文本token的语义重叠但长度差异大（Whisper输出token远长于对应文本），导致直接训练效果差；(2) 长语音支持极弱（VITA只支持~1分钟）；(3) 训练大型omni-model需要海量数据（VITA 5M、Intern-Omni 27M），效率低。
3. **核心矛盾**：在LLM中加入语音模态时，联合训练会损害已有的视觉/语言能力；直接用语音token替代文本instruction会导致显著性能下降（MM-Vet用语音指令比文本指令低8%）。
4. **本文要解决什么**：高效构建一个真正支持image/video/speech/sound全模态理解和交互的MLLM，重点解决语音与其他模态的深度融合。
5. **切入角度**：(1) 用DTW对齐语音token和转写文本token减小模态gap；(2) 用模态专属LoRA保护已有能力；(3) 用query-aware token提取器减少长上下文开销。
6. **核心idea一句话**：以语音为中心，通过DTW跨模态正则化+多模态LoRA+渐进式多模态token提取，用最少数据高效构建全模态MLLM。

## 方法详解

### 整体框架
基于Qwen2-VL，加入Whisper-v3语音编码器和ImageBind音频编码器。输入多模态token → 各模态projector → LLM（带多模态LoRA和Latent提取器）→ 文本+语音流式输出。四阶段训练：(1)语音对齐预训练 → (2)三模态联合训练 → (3)长语音SFT → (4)语音生成。

### 关键设计

1. **Latent Cross-Modality Regularizer (LCMR)**:
   - 做什么：在训练时对齐语音token和对应的转写文本token的潜在表示。
   - 核心思路：语音token $X^{[speech]} \in \mathbb{R}^{d \times L}$和STT文本token $X^{[STT]} \in \mathbb{R}^{d \times S}$长度不同（$L \gg S$），用Dynamic Time Warping（DTW）算法找到最优对齐路径，最小化对齐后的余弦距离：$\mathcal{L}_{LCMR} = \frac{1}{L+S} D_{L,S}$。其中DTW的距离矩阵$D_{l,s} = \text{dist}(l,s) + \min\{D_{l,s-1}, D_{l-1,s}, D_{l-1,s-1}\}$。
   - 效果：加入LCMR后，语音指令的MM-Vet从53.1提升到58.1（+5.0），同时文本指令也从61.1提升到62.6，两个模态的性能gap从8%缩小到4.5%。
   - 设计动机：语音和文本在语义上是同一内容的不同表示形式，但长度不同。DTW能处理变长对齐，确保语音token在进入LLM前就携带与文本等价的语义信息。

2. **Multi-Modality LoRA (MLoRA)**:
   - 做什么：为不同模态组合训练不同的LoRA适配器，避免新模态训练损害已有能力。
   - 核心思路：$H = (B^{[M]}A^{[M]} + W)X^{[M]}$，其中$M$是模态组合（text, image, speech等），每种组合对应独立的低秩适配器。
   - 效果：相比full MLoRA的SFT，MLoRA在保持原始视觉能力（TextVQA 82.6 vs 81.3）的同时更好地发展语音能力（MM-VetS 60.0 vs 54.0），且只需50%数据量。
   - 设计动机：Qwen2-VL等预训练模型已经非常强大，全参数微调在有限数据下会导致灾难性遗忘。LoRA冻结原始权重+低秩更新，从根本上减小了模态间干扰。

3. **Latent Multi-Modality Extractor (LMME)**:
   - 做什么：在LLM的每个block末层，根据文本query与多模态token的attention相关性，逐步剔除冗余token。
   - 核心思路：将LLM分为$n$个block，每个block末层计算$\text{topk}(\text{softmax}(\frac{Q^{[text]} K^{[\neg text]T}}{\sqrt{d}}))$，只保留$\rho L$个最相关的多模态token。跨block token数量指数衰减。
   - 效果：LMME(4,0.7)将prefill时间减半（0.65s→0.37s @$2^{14}$ tokens），训练时间减少29-54%，GPU内存减少50%以上。性能几乎不降：多个benchmark上平均+0.1%~1.5%。
   - 设计动机：长视频/长语音场景中token数可达数十万，但大部分与当前问题无关。渐进式提取（而非一次性剪枝）让模型在不同深度保留不同粒度的信息。

4. **长语音能力集成**:
   - 首个12K长语音SFT数据集，覆盖数分钟到2小时的YouTube音频。
   - 类似LLaVA-NeXT的图像分割策略处理长音频：切片→Whisper编码→压缩到300token/片→展平。
   - Needle-in-Haystack测试：基线模型450秒后崩溃，加长语音SFT后支持到4500秒（98%准确），加LMME后支持到9900秒（2.75小时）。

## 实验关键数据

### 全模态Benchmark对比

| 模型 | 参数量 | 训练数据 | MME | TextVQA | MMMU | TextVQAS | DocVQAS | MM-VetS | WER↓ |
|------|--------|---------|-----|---------|------|---------|---------|---------|------|
| VITA | 66B | 5M | 2097 | - | 41.6 | - | - | - | 8.1 |
| EMOVA | 14B | 4M | 2205 | - | 55.8 | - | - | - | 4.0 |
| Intern-Omni | 8B | 27M | 2210 | - | 60.0 | 69.1 | 79.9 | 56.0 | - |
| **Lyra-Base** | **9B** | **2.7M** | **2335** | **82.6** | **63.5** | **80.0** | **85.5** | **61.0** | **2.0** |
| **Lyra-Pro** | **74B** | **2.7M** | **2485** | **83.5** | **71.4** | **81.0** | **89.4** | **68.5** | **1.8** |

### 消融：LCMR正则化器

| 配置 | TextVQAS | MM-VetS | TextVQAT | MM-VetT | WER↓ |
|------|---------|---------|---------|---------|------|
| w/o LCMR | 76.7 | 53.1 | 79.5 | 61.1 | 1.9 |
| **w/ LCMR** | **77.8** | **58.1** | **80.1** | **62.6** | **2.0** |

### 效率：LMME提取器

| 配置 | Prefill @$2^{14}$token | TPS @$2^{14}$ | Memory @$2^{14}$ | 训练1.5M数据 |
|------|----------------------|-------------|-----------------|-----------|
| Baseline | 0.65s | 27.3 tok/s | 30G | 66h |
| LMME(4,0.7) | **0.37s (-43%)** | **32.5 (+19%)** | **19G (-37%)** | **47h (-29%)** |

### 关键发现
- **语音改变视觉理解评估方式**：speech-text WER指标无法反映omni-model的真实能力——WER相近的模型在vision-speech任务上差距可达9%。应该用speech-centric多模态评估。
- **最少数据最强性能**：Lyra用2.7M数据超过了Intern-Omni (27M)和VITA (5M)，数据效率提高了10倍。核心归功于MLoRA保护预训练能力+LCMR提升语音对齐质量。
- **长语音是MLLM的蓝海**：现有模型最多支持~1分钟语音，Lyra首次支持2小时。通过长语音可以解决1/3的VideoMME问题（纯音频78.6%准确率，超过GPT-4o+字幕在long类别上的表现）。
- **LMME的token提取是可视化的**：保留的token区域与用户问题高度相关（不同问题保留不同视觉/音频区域），最终只保留10-25%的token。

## 亮点与洞察
- **DTW对齐语音-文本是优雅的跨模态对齐**：利用了语音和文本的天然对应关系，DTW处理变长成本低且算法成熟。这种思路可以迁移到任何有自然对应关系的跨模态对齐场景。
- **MLoRA的模态组合路由**：不同模态组合用不同LoRA，既简单又有效。相比全参数SFT，用一半数据就能达到更好效果——这是一个非常实用的多模态高效训练范式。
- **首次系统验证"长语音+视觉"的价值**：证明了长语音信息可以显著补充视觉理解（VideoMME +6.5%），且纯音频就能解决大量视频理解问题。这开辟了"音频作为视觉辅助"的新方向。

## 局限性 / 可改进方向
- 语音生成质量依赖CTC+vocoder，无法生成情感丰富的语音。
- Sound模态依赖ImageBind的单token编码，泛化性有限。
- 长语音推理仍需要大量GPU内存（即使有LMME）。
- 未与GPT-4o等闭源模型在语音交互上做直接对比。

## 相关工作与启发
- **vs VITA**: VITA用66B参数+5M数据只支持1分钟语音。Lyra-Base用9B+2.7M数据支持2小时，且多个benchmark上更强。关键差异是LCMR和MLoRA。
- **vs EMOVA**: EMOVA关注情感语音但不支持长语音和vision-speech交互。Lyra更全面。
- **vs Intern-Omni**: Intern-Omni需要27M数据，Lyra只需2.7M。Lyra在vision-speech上高出约9%（DocVQAS 85.5 vs 79.9）。

## 评分
- 新颖性: ⭐⭐⭐⭐ DTW跨模态正则化和多模态LoRA组合新颖，长语音SFT填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ Vision-language+vision-speech+speech-language全面评估，3个模型规模，长语音Needle测试，大量消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，speech-centric评估视角有洞察
- 价值: ⭐⭐⭐⭐⭐ 2.7M数据超过27M的竞品，长语音首次突破2小时，对omni-model方向有重要意义
