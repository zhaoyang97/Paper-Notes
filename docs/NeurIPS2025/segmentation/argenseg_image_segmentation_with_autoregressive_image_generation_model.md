# ARGenSeg: Image Segmentation with Autoregressive Image Generation Model

**会议**: NeurIPS 2025  
**arXiv**: [2510.20803](https://arxiv.org/abs/2510.20803)  
**代码**: 无  
**领域**: 图像分割 / 多模态大模型  
**关键词**: 自回归图像生成, VQ-VAE分割, MLLM统一框架, Next-Scale Prediction, 理解与生成统一  

## 一句话总结
提出ARGenSeg——首个利用自回归图像生成范式实现图像分割的统一MLLM框架，让模型直接输出visual tokens并通过VQ-VAE解码为分割mask，无需额外分割头，搭配next-scale prediction并行生成策略实现4×加速，在RefCOCO/+/g上以更少训练数据超越SOTA。

## 研究背景与动机

1. **领域现状**：将图像分割集成到MLLM中是当前热点。两类主流方案：（a）边界点序列表示（PolyFormer等）——将mask离散化为多边形点序列，但无法处理复杂形状；（b）专用分割解码器（LISA、PSALM等）——用special token或hidden states驱动SAM/Mask2Former解码，但模型架构复杂且LLM本身不学习像素级理解。
2. **现有痛点**：（a）点序列表示导致分割不完整和边界不自然；（b）专用解码器使LLM依赖外部模块而非自身学习fine-grained视觉理解；（c）推理速度慢（HiMTok等方法）。
3. **核心矛盾**：分割需要密集像素级输出，但LLM原生只做token级预测——如何让LLM"生成"分割mask而不依赖外部decoder？
4. **本文要解决什么**：让MLLM通过自回归图像生成直接产生分割mask，不需要任何额外分割头。
5. **切入角度**：将分割视为一种特殊的图像生成——生成的"图像"就是目标物体的mask。
6. **核心idea**：MLLM输出VQ-VAE的visual tokens → VQ-VAE解码器重建为mask图像 → 无外部分割解码器，分割能力完全来自MLLM的像素级理解。

## 方法详解

### 整体框架
基于InternVL 2.5初始化。输入：图像（连续特征，通过vision encoder）+ 文本指令（tokenizer）。输出：当需要分割时，MLLM输出visual tokens，VQ-VAE解码为mask图像。理解和生成任务共享统一的prediction head。

### 关键设计

1. **统一的Visual Token预测**
   - 做什么：让MLLM直接预测VQ-VAE codebook中的visual token IDs。
   - 核心思路：将VQ-VAE codebook（size=4096）中的tokens作为新"词汇"加入LLM词表。生成分割mask时，模型在遇到`<gen_start>`标记后，开始预测visual tokens。统一的classification head同时处理文本和visual token的预测，用cross-entropy loss监督（训练时GT visual tokens由VQ-VAE encoder获得）。
   - 设计动机：不用special token+外部decoder的方案，让LLM必须自己学习理解像素级信息才能预测正确的visual tokens。实验证明这是获得高精度的关键。

2. **Next-Scale Prediction加速**
   - 做什么：采用VAR的多尺度生成策略，每步并行生成整个scale的所有tokens。
   - 核心思路：使用VAR tokenizer将特征量化为K=10个尺度的token maps $(r_1, \ldots, r_{10})$。每一步生成当前尺度所有$h_k \times w_k$个tokens（并行），上一步的token map上采样后作为当前步的query。最终256×256图像用680个visual tokens表示，仅需10步自回归。
   - 设计动机：（a）粗到细的多尺度生成与分割的"先定位后细化"直觉一致；（b）比逐token生成快4×以上。

3. **训练策略：单阶段联合训练**
   - 做什么：在分割数据（402K）和理解数据（1.25M）上联合SFT。
   - 核心思路：Vision encoder和VQ-VAE全程冻结，只训练LLM和projector。利用预训练的多模态理解能力快速收敛。分割数据仅402K，远少于HiMTok的2.91M。
   - 设计动机：冻结tokenizer确保LLM必须自己学习像素级信息，而非依赖可学习的decoder。

### 损失函数
- 统一使用交叉熵损失，覆盖文本token和visual token的预测

## 实验关键数据

### 主实验（Referring Segmentation - RefCOCO/+/g cIoU）

| 方法 | 范式 | RefCOCO val | RefCOCO+ val | RefCOCOg val | 训练数据量 |
|------|------|------------|-------------|-------------|-----------|
| LISA-7B (ft) | 专用head | 74.9 | 65.1 | 67.9 | - |
| PSALM | 专用head | 83.6 | 72.9 | 73.8 | - |
| HiMTok-8B | 生成式(专用tokenizer) | 81.1 | 77.1 | 75.8 | 2.91M |
| HiMTok-8B (ft) | 同上 | 85.0 | 79.7 | 80.0 | 2.91M |
| **ARGenSeg** | **生成式(通用VQ-VAE)** | **82.2** | **77.9** | **78.4** | **402K** |
| **ARGenSeg (ft)** | 同上 | **86.3** | **82.3** | **81.7** | **402K** |

### 推理速度对比

| 方法 | 推理时间/图 | 加速比 |
|------|-----------|--------|
| HiMTok | ~4x baseline | 1× |
| UniGS (扩散) | ~10x | 0.4× |
| **ARGenSeg** | **~1x** | **4×+** |

### 关键发现
- **无额外分割头即达SOTA**：ARGenSeg是首个无需任何分割head就超越所有专用head方法的统一框架。
- **数据效率极高**：用402K分割数据超越使用2.91M的HiMTok（RefCOCO val: 86.3 vs 85.0）。
- **直接输出visual tokens是关键**：消融中，如果用类似LISA的hidden state+decoder方案，性能明显下降。
- **多尺度生成提升鲁棒性**：不仅加速推理，粗到细的过程也提升了分割质量。
- **可扩展到图像生成**：少量额外训练数据就能解锁文本到图像生成能力，验证了框架的通用性。

## 亮点与洞察
- **分割 = 图像生成**：将分割重新定义为条件图像生成（生成的"图像"是mask），概念简洁又有效。这避开了所有专用分割头的设计，让MLLM端到端学习像素级理解。
- **通用VQ-VAE vs 专用tokenizer**：HiMTok需要训练专门的mask tokenizer，ARGenSeg使用通用VQ-VAE——更通用、可扩展到其他生成任务。
- **冻结tokenizer的重要性**：冻结VQ-VAE保证分割质量完全取决于MLLM的理解能力，这是与"理解驱动分割"理念一致的关键设计。

## 局限性 / 可改进方向
- 输出分辨率固定为256×256，高分辨率分割可能需要更多尺度。
- VQ-VAE重建质量是性能上限——更好的tokenizer可能进一步提升。
- 实例分割和全景分割的评测不如referring segmentation充分。
- 训练仍需冻结tokenizer，端到端联合训练tokenizer+LLM是否更好未探索。

## 相关工作与启发
- **vs LISA/PSALM**: 这些方法用special token embedding驱动SAM/Mask2Former，LLM只提供语义信息不处理像素。ARGenSeg让LLM直接预测像素级tokens。
- **vs HiMTok**: 同为生成式范式，但HiMTok用专用mask tokenizer且需2.91M数据，ARGenSeg用通用VQ-VAE+402K数据更优。
- **vs Janus/Emu3**: 这些是理解+生成统一框架但不做分割，ARGenSeg证明统一框架可以扩展到像素级感知。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "分割=图像生成"范式转换 + 无分割头SOTA是突破性贡献
- 实验充分度: ⭐⭐⭐⭐ Referring/generalized/reasoning segmentation均有评测
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机论证充分
- 价值: ⭐⭐⭐⭐⭐ 为MLLM统一框架的像素级感知开创了新范式
