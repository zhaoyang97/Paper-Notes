# UniCode: Learning a Unified Codebook for Multimodal Large Language Models

**会议**: ECCV 2024
**arXiv**: [2403.09072](https://arxiv.org/abs/2403.09072)
**代码**: 无
**领域**: 多模态VLM
**关键词**: 统一码本, 视觉量化, 多模态生成, 向量量化, 图像生成

## 一句话总结

UniCode提出学习一个统一的codebook来同时tokenize视觉和文本信号，通过language-driven iterative training范式将视觉tokenizer的码本与LLM的词表渐进对齐，并引入in-context image decompression预训练任务提升图像生成质量，使MLLM无需额外对齐模块即可实现多模态理解与生成。

## 研究背景与动机

1. **领域现状**：当前MLLM主要通过轻量级projector将视觉信号映射到LLM的文本空间，但这种范式只能生成文本，不能生成图像等非语言内容。
2. **现有痛点**：
   - **范式A**（vis enc + text tok，如LLaVA）：用projector对齐视觉和文本，但只能输出文本
   - **范式B**（vis tok + text tok，如Unified-IO 2）：将视觉codebook拼接到文本词表中，但扩大词表导致参数暴增，且面临"codebook collapse"问题（模型过度依赖少数code）。此外跨模态的分布差异难以弥合
3. **核心矛盾**：要让MLLM具备多模态生成能力，需要一个能同时表示视觉和文本的token空间。但视觉codebook与文本词表的分布差异很大，如何统一是关键挑战。
4. **本文解决的问题**：能否学习一个unified codebook，直接让LLM的词表能够量化视觉信号？
5. **切入角度**：不扩大codebook，而是让视觉tokenizer的codebook与LLM的现有词表逐步对齐收敛，共享同一套code。
6. **核心idea**：通过EMA平滑地将视觉codebook向LLM词表靠拢（language-driven iterative training），辅以in-context image decompression任务增强生成能力。

## 方法详解

### 整体框架

Pipeline: 图像 → Visual Encoder → 特征图 $Z_0$ → Vector Quantization（使用统一码本） → code map $M$ → Stacked Quantization压缩 → 聚合嵌入 → LLM → 输出（文本 or 视觉token）→ Visual Decoder → 重建图像

两阶段训练：Stage I（统一码本学习）→ Stage II（多模态指令微调）

### 关键设计

1. **Visual Tokenization与Stacked Quantization**:
   - 做什么：将图像压缩为离散token序列，使其可被LLM处理
   - 核心思路：使用VQ-VAE框架，encoder $\mathbb{E}$ 提取特征图 $Z_0 \in \mathbb{R}^{h \times w \times c}$，然后对每个向量 $z$ 在codebook $\mathbb{C}$ 中找最近邻：$Q(z;\mathbb{C}) = \arg\min_{k} \|z - e(k)\|_2^2$
   - 为解决code map分辨率过高导致LLM序列过长的问题，采用stacked quantization（如HQ-VAE）将 $h \times w$ 的code map压缩为 $\hat{h} \times \hat{w} \times D$ 的多层堆叠码图，每个位置的最终嵌入为D层量化向量的聚合：$\hat{z}_{ij} = \mathcal{F}_{d=1}^{D} e(\hat{M}_{i,j,d})$
   - 设计动机：直接使用 $256 \times 256$ 分辨率的code map会导致LLM需要处理过长序列。Stacked quantization在保持信息保真度的同时大幅减少token数量

2. **Language-Driven Iterative Training（核心创新）**:
   - 做什么：学习一个同时适用于视觉量化和文本处理的统一码本
   - 核心思路：交替训练visual tokenizer和LLM，通过EMA将visual tokenizer的codebook平滑地向LLM词表靠拢：
     $$\mathbb{C}' = \lambda \mathbb{C} + (1-\lambda) \mathbb{C}_L$$
     其中 $\mathbb{C}_L$ 是LLM的词表嵌入，$\lambda$ 是衰减率。关键在于：只用LLM的codebook去更新visual tokenizer的codebook，而不反向更新LLM
   - 设计动机：
     - **Frozen LLM codebook方案**会导致重建质量差，因为缺乏encoder/decoder与frozen codebook的同步机制
     - **Dual alternative training方案**会导致LLM语言能力崩溃，因为视觉codebook变化速率远大于文本codebook，频繁替换会破坏LLM内部一致性
     - **Language-driven方案**兼顾两端：EMA保证了visual codebook缓慢向LLM词表收敛，同时不扰动LLM自身训练
   - 消融验证：dual方案VQA性能崩溃（VQA-v2从53.1→9.3），frozen方案生成质量差（FID 34.45 vs 6.72），language-driven方案两者兼优

3. **In-Context Image Decompression预训练任务**:
   - 做什么：将压缩的视觉嵌入"解压"为多层code map，作为预训练任务增强LLM理解视觉token的能力
   - 核心思路：输入压缩后的量化嵌入 $\hat{Z} \in \mathbb{R}^{\hat{h} \times \hat{w}}$，自回归地预测展开的code序列 $\{u_1, u_2, ..., u_{\hat{h} \times \hat{w} \times D}\}$：
     $$\max_\theta \sum_{l=1}^{\hat{h} \times \hat{w} \times D} \log P_\Theta(u_l | u_{<l}; \hat{Z})$$
     将图像分段为 $T$ 个片段，利用multi-turn对话格式进行in-context learning
   - 设计动机：stacked quantization导致聚合嵌入与LLM词表的对齐更复杂。decompression任务迫使LLM学会理解压缩视觉表示的内部结构，避免过早收敛

### 损失函数 / 训练策略

- **Stage I（统一码本学习）**：交替训练visual tokenizer（图像重建任务，使用LCS-558K）和LLM（文本指令数据）。每隔一定步数用LLM的codebook通过EMA更新visual codebook
- **Stage II（多模态指令微调）**：冻结visual encoder和decoder，仅微调LLM。使用Mixed-665K + CC3M（text-to-image）+ image decompression数据的组合
- 损失函数：标准NLL目标 $\mathcal{L}(\Theta) = -\sum_{j=1}^{L} \log P_\Theta(y_j | \mathcal{I}, \hat{y}_{1:j-1})$，仅计算answer tokens的loss

## 实验关键数据

### 主实验

| Benchmark | 指标 | UniCode | UniCode+ | LLaVA-1.5 | Emu | 说明 |
|-----------|------|---------|----------|-----------|-----|------|
| VQA-v2 | Acc | 53.1 | **56.2** | 79.1 | 52.0 | 超越同类生成模型Emu |
| VizWiz | Acc | 46.2 | **47.1** | 47.8 | 34.2 | 接近LLaVA-1.5 |
| ScienceQA | Acc | 62.9 | **65.4** | 68.4 | - | 有竞争力 |
| POPE | Acc | 71.8 | **77.6** | 86.4 | - | 差距最大的benchmark |
| ImageNet | FID↓ | **6.72** | - | - | - | 类条件生成SOTA |
| LSUN-Cat | FID↓ | **8.07** | - | - | - | 无条件生成强劲 |
| LSUN-Church | FID↓ | **6.96** | - | - | - | 超越StyleGAN2 (3.86) |
| CC3M | FID↓ | **11.54** | - | - | - | 文本条件生成 |

UniCode是同时具备理解和生成能力的模型，用更少参数（104M visual tokenizer vs Emu的1B）和更少数据即可达到同类模型最优。

### 消融实验

| 配置 | VQA-v2 | VizWiz | POPE | ImageNet FID↓ | 说明 |
|------|--------|--------|------|---------------|------|
| vis enc+text tok | 52.3 | 45.4 | 69.7 | - | 只能理解不能生成 |
| vis tok+text tok | 49.0 | 44.5 | 65.4 | 9.82 | 分离码本，性能差 |
| **unified tok** | **53.1** | **46.2** | **71.8** | **6.72** | 统一码本，两端最优 |
| frozen codebook | 44.2 | 35.1 | 63.9 | 34.45 | 码本冻结导致生成崩溃 |
| dual training | 9.3 | 5.2 | 13.2 | 8.87 | 双向更新导致VQA崩溃 |
| **iterative (ours)** | **53.1** | **46.2** | **71.8** | **6.72** | 我方最佳范式 |
| w/o ImgDecomp | - | - | - | 7.08 | 无decompression |
| **w/ ImgDecomp** | - | - | - | **6.72** | 有decompression，-0.36 |

### 关键发现

- **统一codebook优于分离codebook**：在VQA和图像生成上全面胜出，因为共享token空间让视觉token与LLM更自然地交互
- **Codebook学习范式至关重要**：dual training会导致灾难性遗忘（VQA-v2: 53.1→9.3），frozen方案生成质量极差（FID 34.45 vs 6.72）
- **Visual tokenizer质量直接影响终端性能**：从VQ-GAN→RQ-VAE→HQ-VAE逐步升级，VQA-v2从49.1→49.8→53.1，说明UniCode框架可随视觉tokenizer进步而持续提升
- **Image decompression任务**有效降低FID（ImageNet 7.08→6.72），通过增加训练复杂度防止过早收敛
- **分辨率对齐很重要**：训练256×256但测试320×320时性能显著下降，因为code map每个元素代表的图像区域不一致

## 亮点与洞察

- **统一码本是新范式**：与扩大词表不同，UniCode证明了可以让视觉和文本共享同一码本，无需额外对齐模块。这是multi-modal I/O的一个优雅解法
- **Language-driven更新方向的洞察**：关键insight是视觉codebook变化速度远快于LLM词表，因此只能单向从LLM向visual同步，不能反向
- **EMA的妙用**：将EMA从传统的batch统计更新（BN）类比到cross-modal codebook同步，是一个简洁有力的技术贡献
- **Image decompression作为预训练任务**：将视觉token的解压重构转化为LLM的in-context learning任务，巧妙地让LLM学习"看懂"压缩的视觉表示
- **框架的可扩展性**：兼容VQ-GAN、RQ-VAE、HQ-VAE等多种视觉量化方案，升级tokenizer即可提升整体性能

## 局限性 / 可改进方向

- VQA性能与LLaVA-1.5差距较大（POPE: 71.8 vs 86.4），主要受限于visual tokenizer的质量和训练数据规模
- Visual tokenizer仅用558K图像训练，远少于CLIP的400M——扩大数据量有巨大提升空间
- 图像重建质量在LCS-558K（多样场景）上明显不如ImageNet，泛化性需要加强
- 训练/测试分辨率必须一致，灵活性不足
- 统一码本的大小有隐式限制——LLM词表通常32K，是否足以覆盖视觉语义尚待验证
- 仅支持图像生成，未扩展到视频或音频等其他模态

## 相关工作与启发

- **vs Emu**：Emu用1B参数visual encoder + 82M预训练数据做多模态生成，UniCode仅用104M参数 + 无额外预训练数据即超越其VQA性能
- **vs Unified-IO 2**：Unified-IO 2扩大词表拼接视觉和文本code，需要1B image-text对训练；UniCode共享同一词表，资源需求远低
- **vs LQAE**：LQAE用frozen BERT词表做视觉量化但重建质量差；UniCode通过iterative training保持重建质量的同时实现统一
- **vs SPAE**：SPAE用多层pyramid tokenizer对齐frozen LLM，需指数级token数；UniCode用stacked quantization + decompression任务更高效

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一码本的概念新颖，language-driven iterative training和image decompression都是原创贡献
- 实验充分度: ⭐⭐⭐⭐☆ 消融实验非常系统（三种范式、三种codebook学习方式、多种visual tokenizer），但与LLaVA-1.5等SOTA差距大，说服力受限
- 写作质量: ⭐⭐⭐⭐☆ 方法描述清晰，三种paradigm和三种codebook学习方式的对比图很直观
- 价值: ⭐⭐⭐⭐☆ 指出了一条有前途的研究方向（统一多模态码本），但当前性能还需大幅提升才能实用化
