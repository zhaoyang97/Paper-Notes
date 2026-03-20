# MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding

**会议**: ICCV 2025  
**arXiv**: [2411.17762](https://arxiv.org/abs/2411.17762)  
**代码**: 无  
**领域**: 多模态VLM / 统一理解与生成  
**关键词**: unified VLM, visual tokenizer, semantic discrete encoding, autoregressive, image generation  

## 一句话总结
提出语义离散编码（SDE）视觉tokenizer，在VQGAN基础上加入SigLIP语义特征约束，使离散视觉token与语言token语义对齐，构建统一的自回归VLM（MUSE-VL），在仅用24M数据的条件下理解性能比Emu3提升4.8%，超过LLaVA-NeXT 34B专用理解模型3.7%，同时支持图像生成。

## 研究背景与动机
1. **领域现状**：统一多模态理解和生成是VLM的重要方向。现有统一模型（如Chameleon、Show-o、Emu3）使用VQGAN等视觉tokenizer将图像转为离散token，与文本token一起做next-token prediction。但VQGAN只关注底层像素信息（重建损失），生成的离散token缺乏语义信息。
2. **现有痛点**：(1) VQGAN token与语言token间有很大的语义鸿沟，导致统一模型的理解性能远低于专用模型；(2) Chameleon等模型需要从零训练LLM，数据需求和计算成本极高；(3) VILA-U尝试同时训练对比损失和重建损失来对齐语义，但存在严重的损失冲突导致收敛困难。
3. **核心矛盾**：视觉token需要同时满足两个矛盾需求——保留底层像素信息用于图像重建/生成，同时承载高层语义信息用于理解。VQGAN只顾前者，CLIP只顾后者，直接组合（如VILA-U）会冲突。
4. **本文要解决什么**：设计一种视觉tokenizer，使离散码既保留图像重建能力又包含丰富语义信息，从而降低统一VLM的训练难度。
5. **切入角度**：不直接对比学习（VILA-U的问题），而是用预训练SigLIP的图像编码器提取语义特征，将其融合到编码过程中，同时用语义解码器（重建SigLIP特征）和图像解码器（重建像素）两个分支约束。
6. **核心idea一句话**：将SigLIP语义特征融合到视觉量化过程中，用双分支（语义重建+图像重建）训练视觉tokenizer，使离散码自然包含语义信息。

## 方法详解

### 整体框架
SDE Tokenizer将图像编码为16×16或27×27的离散token序列 → 与文本token一起送入自回归Transformer（基于Qwen2.5/Yi-1.5 LLM） → 用标准cross-entropy next-token prediction训练。理解任务：视觉token在prompt中，输出文本回复。生成任务：文字描述在prompt中，输出视觉token后用图像解码器解码。

### 关键设计

1. **语义离散编码（SDE）Tokenizer**:
   - 做什么：将图像离散化为既包含像素信息又包含语义信息的visual tokens。
   - 核心思路：图像先经过初始化自SigLIP的编码器得到特征$z$，同时冻结的SigLIP提取语义特征$T$。将$T$和$z$相加后进行向量量化：$z_q = \text{Quant}(T + z)$。量化后的特征分别送入两个解码器：(1) 语义解码器（Transformer）重建SigLIP特征，损失为$L_{\text{sem}} = 1 - \cos(Dec_s(z_q), T)$；(2) 图像解码器（ConvNet）重建原始像素，损失为$L_{\text{img}} = \ell_2 + L_P + \lambda_G L_G$（像素+感知+对抗）。总损失$L = L_{\text{sem}} + L_{\text{img}} + L_{\text{vq}}$。
   - 设计动机：通过特征融合（而非对比学习）将语义注入量化码，避免了VILA-U的损失冲突。语义解码器确保量化码"记住"语义信息，图像解码器确保保留像素信息。codebook大小32768，维度8。
   - 与VILA-U的关键区别：VILA-U用文本编码器提取语义+对比学习，会冲突；SDE直接用图像编码器（SigLIP）的特征做融合+重建，更稳定。

2. **统一自回归建模**:
   - 做什么：将视觉离散token和文本token在同一个自回归模型中统一处理。
   - 核心思路：在现有LLM（Qwen2.5/Yi-1.5）的embedding层扩展32768维（codebook大小），用`<soi>`和`<eoi>`标记视觉token的开始和结束。训练目标纯next-token prediction，不需要修改LLM架构。
   - 设计动机：因为SDE token已经与语言对齐，无需额外的适配器或架构修改，可以直接利用预训练LLM的语言能力。这大大降低了训练复杂度。

3. **两阶段训练**:
   - **预训练**：在图文对数据上计算所有token的loss，学习视觉token的embedding并对齐。
   - **指令微调**：理解任务用SFT数据，只在回复部分计算loss；生成任务用反转的图文对（文字→图像），只在视觉token部分计算loss。

### 数据效率
MUSE-VL仅使用24M图文对，远少于Show-o (35M)和VILA-U (720M)，但理解性能更强。

## 实验关键数据

### 主实验（多模态理解）

| 模型 | LLM | Token类型 | MMBench | MMStar | SEED | MMMU | SQA-I | AI2D | MathVista | AVG |
|------|-----|---------|---------|--------|------|------|-------|------|-----------|-----|
| Chameleon-7B | 7B from scratch | Discrete | 31.1 | 31.1 | 30.6 | 25.4 | 46.8 | 46.0 | 22.3 | 33.3 |
| Emu3-8B | 8B from scratch | Discrete | 58.5 | 46.6 | 68.2 | 31.6 | 89.2 | 70.0 | 47.6 | 58.8 |
| LLaVA-NeXT-7B | Vicuna-7B | Continuous | 67.4 | 37.6 | 70.2 | 35.8 | 70.1 | 66.6 | 34.6 | 54.6 |
| LLaVA-NeXT-34B | Yi-34B | Continuous | 79.3 | 51.6 | 75.9 | 51.1 | 81.8 | 78.9 | 46.5 | 66.4 |
| **MUSE-VL-7B** | Qwen2.5-7B | Discrete | **72.1** | **49.6** | **69.1** | **39.7** | **93.5** | **69.8** | **51.3** | **63.6** |
| **MUSE-VL-32B** | Qwen2.5-32B | Discrete | **81.8** | **56.7** | 71.0 | 50.1 | **95.0** | **79.9** | **55.9** | **70.1** |

Tokenizer对比（同LLM Yi-1.5-9B，同数据）:

| Tokenizer | MMBench | SEED | MMStar | AVG |
|-----------|---------|------|--------|-----|
| VQGAN | 32.0 | 42.7 | 29.1 | 34.6 |
| SEED | 63.1 | 57.8 | 39.1 | 53.3 |
| LaVIT | 63.3 | 59.5 | 40.3 | 54.4 |
| **SDE (ours)** | **70.6** | **68.1** | **43.8** | **60.8** |

### 消融实验

| 图像分支 | 语义分支 | rFID↓ | MMBench | SEED | MMStar | AVG |
|---------|---------|-------|---------|------|--------|-----|
| ✓ | ✗ | 2.63 | 42.8 | 48.5 | 38.1 | 43.1 |
| ✗ | ✓ | - | 72.5 | 67.5 | 48.1 | 62.7 |
| ✓ | ✓ | 2.26 | 72.1 | 69.1 | 49.6 | 63.6 |

### 关键发现
- **语义是关键差距**：纯VQGAN tokenizer的VLM理解AVG只有34.6%，加入语义约束后飙升到60.8%（+26.2%）。这直接证明了离散token缺乏语义信息是统一VLM理解性能差的主因。
- **SDE比VQGAN理解提升20.5%，图像重建质量相当**（rFID 2.26 vs 2.63），说明语义约束不仅没有损害生成能力反而改善了。
- **数据效率极高**：24M数据 > VILA-U的720M数据，因为语义对齐大幅降低了LLM学习视觉token的难度。
- **良好的缩放特性**：从7B到32B，性能持续提升（63.6→70.1 AVG）。
- MUSE-VL-7B在SQA-I (93.5)和MathVista (51.3)上甚至超过了LLaVA-NeXT-34B，说明离散token方法在推理任务上有优势。
- 视觉码可视化显示同一语义概念（如猫耳朵、草莓）会被分配相同的code ID。

## 亮点与洞察
- **SDE的设计非常优雅**：通过"语义特征融合+双分支解码"替代"对比学习"，避免了VILA-U的损失冲突问题。背后的洞察是——SigLIP图像编码器的特征已经隐式包含了与文本对齐的语义信息，不需要显式做对比学习，只需要在量化过程中"融入"并"保持"这些语义。
- **统一模型的范式简洁性**：不改LLM架构、不加额外适配器、不需要diffusion model，纯next-token prediction。证明了只要tokenizer够好，统一模型可以极其简洁。
- **对统一VLM研究方向的启示**：问题不在LLM架构，而在视觉tokenizer。好的tokenizer可以让简单方法超越复杂系统。

## 局限性 / 可改进方向
- 图像生成质量虽然超过之前的统一模型但仍不如SD-XL等专用扩散模型。
- 分辨率受限（256或384），更高分辨率可能需要更大codebook或多尺度方案。
- SDE的語義編碼器（SigLIP）是冻结的，联合训练可能进一步提升但会增加复杂度。
- 仅用LLamaGEN的卷积decoder做图像重建，更强的decoder可能提升生成质量。

## 相关工作与启发
- **vs Emu3**: Emu3需要从零训练8B LLM，MUSE-VL直接用预训练LLM+24M数据就超过了Emu3 (+4.8%)。关键差异在于SDE tokenizer使visual token与text token天然对齐。
- **vs Chameleon**: Chameleon用标准VQGAN，理解AVG只有33.3%，MUSE-VL的63.6%是其2倍。
- **vs VILA-U**: VILA-U用对比+重建双损失对齐，需要720M数据且收敛困难。SDE用特征融合+语义重建，只需24M数据。
- **vs Janus**: Janus用双编码器分离理解和生成，增加了模型复杂度。MUSE-VL用单一SDE编码器兼顾两者。

## 评分
- 新颖性: ⭐⭐⭐⭐ SDE tokenizer的设计新颖且解决了VQ tokenizer的核心语义缺失问题
- 实验充分度: ⭐⭐⭐⭐⭐ 7个理解benchmark + 2个生成benchmark，tokenizer对比，消融，缩放实验全面
- 写作质量: ⭐⭐⭐⭐ 清晰，问题定义准确
- 价值: ⭐⭐⭐⭐⭐ 为统一VLM提供了一条简洁高效的路径，SDE tokenizer可作为通用组件复用
