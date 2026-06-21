---
title: >-
  [论文解读] SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention
description: >-
  [ICLR 2026][计算生物][单细胞 RNA 测序] SAVE 把单细胞表达谱里的几千个基因按 LLM 语义相似度聚成若干"基因块"，在块粒度上做 Transformer 注意力 + 变分自编码压缩 + 潜空间 Flow Matching 生成，并用 AdaLN 注入条件、用条件掩码统一生成与迁移任务，在条件生成、批次校正、扰动预测三类任务上、尤其在低资源和未见条件组合下显著超过现有方法。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "单细胞 RNA 测序"
  - "基因块注意力"
  - "条件生成"
  - "Flow Matching"
  - "批次校正"
---

# SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=l7QEoK4uDP](https://openreview.net/forum?id=l7QEoK4uDP)  
**代码**: https://github.com/fdu-wangfeilab/sc-save  
**领域**: 计算生物学 / 单细胞生成 / 条件生成 / Flow Matching  
**关键词**: 单细胞 RNA 测序, 基因块注意力, 条件生成, Flow Matching, 批次校正

## 一句话总结
SAVE 把单细胞表达谱里的几千个基因按 LLM 语义相似度聚成若干"基因块"，在块粒度上做 Transformer 注意力 + 变分自编码压缩 + 潜空间 Flow Matching 生成，并用 AdaLN 注入条件、用条件掩码统一生成与迁移任务，在条件生成、批次校正、扰动预测三类任务上、尤其在低资源和未见条件组合下显著超过现有方法。

## 研究背景与动机

**领域现状**：单细胞 RNA 测序（scRNA-seq）让人能在细胞粒度上观测基因表达，研究者希望用生成模型来模拟不同条件（细胞类型、疾病状态、药物扰动、测序批次等）组合下的细胞状态，从而省下昂贵的湿实验。目前主流是两类：以 scVI 为代表的 VAE（用零膨胀负二项似然建模、把协变量编进隐空间），以及 scGPT / scBERT / Geneformer 这类把每个基因当一个 token、用掩码建模训练的 Transformer "基础模型"。

**现有痛点**：这两类方法都有结构性缺陷。VAE 架构简单，难以建模多个外部条件之间的组合交互；Transformer 基础模型则把基因看成**扁平、token 级**的独立单元，完全忽略了基因模块 / 通路这种生物学高阶结构，而且它们大多只关注非零表达值、丢掉了 scRNA-seq 里富含信息的零膨胀（zero inflation），更关键的是它们大多只做表示编码、并没有被组织成一个**能从学到的条件分布里采样**的生成框架。

**核心矛盾**：基因表达数据高维、稀疏、且**无天然空间顺序**。要建模高阶依赖，最自然的想法是像 ViT 切 patch 一样做"粗粒度"表示；但和像素不同，基因没有空间邻近性可以用来分组，缺乏天然的局部结构。怎么给无序的基因找到一个有生物学意义的"分块"方式，是把粗粒度建模搬到单细胞上的核心障碍。

**本文目标**：(1) 给基因找到一种语义合理的分块表示，捕捉块间高阶关系；(2) 把它装进一个真正能采样的条件生成框架；(3) 让模型能泛化到训练中没见过的条件组合。

**切入角度**：作者借鉴 MaskGIT 等视觉里的掩码生成建模——无序数据可以用 image patch 这类粗粒度表示有效建模。既然基因无序、可以聚类，那就用在海量文本上预训练的 LLM 去读 NCBI 基因功能描述、抽语义特征，把语义相似的基因聚成"基因块"，在块上做注意力。

**核心 idea**：用"LLM 语义聚成的基因块"代替"单基因 token"做粗粒度 Transformer，配 VAE 压缩 + 潜空间 Flow Matching + AdaLN 条件注入 + 条件掩码，统一解决多条件单细胞的生成与迁移。

## 方法详解

### 整体框架
SAVE（Single-cell gene block Attention-based Variational gEnerative framework）是一个潜空间 Flow Matching（LFM）框架，整体分三大模块串行：**先把扁平的表达谱重组成"基因块"张量**，**再用带基因块注意力的 VAE 把细胞压成 latent**，**最后在 latent 上训练一个条件 Flow Matching 网络从噪声生成新细胞**，生成的 latent 再过 VAE 解码器还原成基因表达谱。

具体地：输入是 scRNA-seq 矩阵 $Y \in \mathbb{R}^{N \times G}$（$N$ 个细胞、$G$ 个基因）。第一步 Gene Block Construction 把 $G$ 个基因按语义聚成 $L$ 个等大的块（每块 $K$ 个基因），把 $Y$ 重排成结构化张量 $X \in \mathbb{R}^{N \times L \times K}$。第二步 VAE 编码器在块粒度上做 Transformer 注意力，输出潜分布参数 $\mu, \sigma^2$，采样得到 latent；解码器镜像还原。第三步条件 Flow Matching 网络学一个时间相关的速度场 $v_\theta$，把先验噪声 $x_0$ 沿直线路径搬到数据 latent $x_1$，条件（细胞类型 / 疾病 / 批次等）通过 AdaLN 注入。训练时用**条件掩码**：把条件随机替换成 `[MASK]`，让生成与迁移任务统一、并泛化到未见条件组合。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入 scRNA-seq<br/>Y (N×G)"] --> B["基因块构建<br/>LLM 语义嵌入<br/>+ 最优传输聚类"]
    B --> C["块张量 X (N×L×K)"]
    C --> D["基因块注意力 VAE<br/>块粒度 Transformer 压缩"]
    D --> E["潜表示 latent"]
    E --> F["条件 Flow Matching<br/>噪声→latent 速度场"]
    G["AdaLN 条件注入<br/>+ 条件掩码"] -->|条件 s, [MASK]| F
    F --> H["生成 latent"]
    H --> I["VAE 解码器还原<br/>重建 scRNA-seq"]
```

### 关键设计

**1. Gene Block 构建：用 LLM 语义把无序基因聚成等大的功能块**

这一步直接针对"基因没有天然局部结构、无法像像素那样分块"的核心障碍。SAVE 分两阶段构块。第一阶段是 LLM 嵌入构建：从 NCBI Gene 数据库取每个基因的 "Summary" 功能描述文本，清洗掉无信息内容（清洗后平均约 73 词），再用 LLM（`text-embedding-ada-002`）编码成 1536 维嵌入向量 $g_i$，让每个基因带上丰富的生物学语境。第二阶段是基于最优传输（Optimal Transport）的迭代聚类：把"将 $G$ 个嵌入分到 $L$ 个等大、互不重叠的块"建模成一个带均匀边际约束的传输问题，迭代地算欧氏代价矩阵 $C_{ij} = \|g_i - c_j^{(t)}\|_2^2$、在约束 $T\mathbf{1}_L = a,\ T^\top \mathbf{1}_G = b$ 下求平衡传输计划、更新块质心 $c_j^{(t)}$，直到收敛成结构均衡的功能模块。最终把 $Y$ 重排成 $X \in \mathbb{R}^{N \times L \times K}$。用最优传输而非普通 K-means 的好处是能**强制每块大小相等**，让后续注意力的张量形状规整、计算均衡；用 LLM 语义而非空间邻近来分组，则让块对应到真实的生物功能模块。

**2. Gene Block Attention：把注意力从单基因搬到块粒度，省算力又抓高阶依赖**

把基因当 token 直接做注意力时，序列长度等于基因数（上万），算力爆炸；SAVE 改成在 $L$ 个块上做标准 Transformer。每个块先经可学习 MLP $W^{in}$ 投到 $e$ 维隐空间，再走前置 LayerNorm 的标准块：

$$h_0 = X W^{in},\quad h_{t'} = h_t + \mathrm{Attention}(\mathrm{LayerNorm}(h_t)),\quad h_{t+1} = h_{t'} + \mathrm{FeedForward}(\mathrm{LayerNorm}(h_{t'}))$$

因为序列长度从基因数 $G$ 降到块数 $L$（如 $G$ 上万、$L$ 只有 6），自注意力的二次复杂度被大幅压缩——消融里相对 $K=1$（每基因一个 token）的朴素注意力实现了约 **191×** 训练提速（12.5 vs 2391.2 分钟）。同时块表示天然聚合了语义相关基因，注意力学到的是**模块间**的高阶转录依赖，而不是噪声很大的单基因两两关系。

**3. AdaLN 条件注入 + 条件掩码：统一生成与迁移、泛化到未见条件组合**

多条件单细胞的难点是要同时解耦并建模多个协变量（批次、细胞类型、疾病阶段等）。SAVE 把所有条件编成矩阵 $S \in \mathbb{R}^{N \times d_s}$、每个类别条件查可学习嵌入得到 $S_E$，再用 Adaptive Layer Normalization 把条件信号注入 Transformer：从 $S_E$ 线性导出缩放/偏移参数 $\alpha,\beta,\gamma$，让条件去调制每层归一化后的表示，

$$h_{t'} = h_t + \alpha_1 \cdot \mathrm{Attention}(\mathrm{AdaLN}(h_t, \gamma_1, \beta_1)),\quad \mathrm{AdaLN}(h,\gamma,\beta) = \frac{h - \mathbb{E}[h]}{\sqrt{\mathrm{Var}[h]+\epsilon}}\cdot\gamma + \beta$$

在此基础上加**条件掩码**：训练时把 $S$ 的每个元素以固定概率 $p$（实验取 0.6）独立替换成 `[MASK]`。这一步是泛化能力的关键——它一方面模拟"缺失条件信息"，逼模型在部分条件未知时也能稳健生成；另一方面把"生成"（给 Flow Matching 掩条件）和"迁移/批次校正"（给 VAE 掩条件）两类任务统一进同一个训练目标，从而能外推到训练里没出现过的条件组合。

**4. 潜空间 Flow Matching + VAE：可采样的生成内核**

前三个设计解决了"怎么表示"，这一设计解决"怎么真正采样生成"。VAE 部分给 latent 加高斯先验，编码器输出 $\mu,\sigma^2$、用重参数化采样，并加 KL 正则 $L_p = D_{KL}(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1))$，解码器镜像重建、损失为 $L_{recon} = -\log L(\hat X | X)$。生成内核是条件 Flow Matching：在噪声 $x_0$ 与数据 latent $x_1$ 间用仿射概率路径做线性插值 $x_t = (1-t)x_0 + t x_1$，目标速度场就是其时间导数 $u_t = x_1 - x_0$，网络 $v_\theta(x_t, t, s)$ 用 MSE 回归它，$L_{FM} = \mathbb{E}\,\|v_\theta(x_t,t,s) - u_t\|^2$。推理时从噪声出发解概率流 ODE $\mathrm{d}x_t/\mathrm{d}t = v_\theta(x_t,t,s)$ 到 $t=1$ 得新样本，并用 Classifier-Free Guidance $\hat v_\theta = (1-w)v_\theta(x_t,t) + w\,v_\theta(x_t,t,s)$ 调控多样性与条件保真度的权衡。相比 scDiffusion 那种细粒度扩散，仿射路径的 Flow Matching 训练目标更简单、采样更高效。

### 损失函数 / 训练策略
总训练目标由三部分组成：VAE 的 KL 正则 $L_p$、重建损失 $L_{recon}$、Flow Matching 的 MSE 速度回归 $L_{FM}$。预处理上，每个细胞表达归一化到总计数 $10^4$ 并取 log，再做 max-absolute 归一化缩到 $[0,1]$。默认块大小 $K=3200$、条件掩码比 0.6、AdamW（学习率 $1\times10^{-4}$、权重衰减 $2.5\times10^{-5}$），单张 RTX 3090 24GB 即可训练，所有任务共用同一套超参。

## 实验关键数据

### 主实验

条件生成上，用 Wasserstein Distance（WD↓）和 MMD↓ 衡量生成细胞与真实细胞的分布相似度。

| 任务 / 数据集 | 指标 | SAVE | 次优方法 | 说明 |
|--------------|------|------|----------|------|
| 单条件 · Dentate gyrus | WD / MMD | **9.16 / 0.17** | 21.55 / 1.12 (CFGen) | WD、MMD 都比次优低一半多 |
| 单条件 · Tabula Muris | MMD | **0.04** | 0.19 (CFGen) | 与真实全局均值高度对齐 |
| 双条件 · Heart | WD / MMD | **8.30 / 0.63** | 12.57 / 0.66 (CFGen) | 批次+细胞类型双协变量 |
| 双条件 · PBMC | WD / MMD | **5.37 / 0.29** | 11.38 / 0.48 (scDiffusion) | — |
| 双条件 · Lung Atlas | WD / MMD | **4.37 / 1.14** | 13.89 / 1.71 (scDiffusion) | CFGen 在此退化到接近 scVI |
| 多条件 · Lung Cancer（未见组合 WD） | WD | **4.63** | 5.29 (scDiffusion) | 5 条件映成 27 类，留 13/24 类做外推 |

批次校正（biological conservation Bio.↑ / batch correction Batch.↑ / scIB↑），对比 Scanorama / Harmony / scVI / trVAE：SAVE 在三个数据集都拿到最高 Bio.，并在两个数据集拿到最佳 Batch，整体 scIB 领先（Lung Atlas 0.81、PBMC 0.83），说明它能更好地把生物变异和技术噪声解耦。

扰动预测（PBMC-IFN，IFN-β 药物扰动，PCC↑ / R²↑）：SAVE 平均 PCC > 0.95、R² 0.86，超过所有基线；强基线 CellOT、scGEN 平均 PCC 约 0.94，trVAE 居中，MFM 最差（平均 PCC 0.71，甚至不如"未扰动对照 vs 扰动"的基线相关 0.86）。在 control 与 stimulated 差异大的 CD14+Mono、FCGR3A+Mono 上 SAVE 优势尤其明显。

### 消融实验

| 配置 | WD ↓ | MMD ↓ | 说明 |
|------|------|-------|------|
| SAVE（完整，Heart） | 8.30 | 0.63 | 完整模型 |
| SAVE w/o 基因块注意力 | 8.89 | 0.65 | 去掉后 WD 明显变差 |

基因块大小 $K$ 的影响（生成质量 + 训练效率）：

| $K$ | 块数 $L$ | WD ↓ | MMD ↓ | 训练时间 (min) |
|-----|---------|------|-------|----------------|
| 1（朴素逐基因注意力） | 19112 | — | — | 2391.2 |
| 1600 | 12 | 9.64 | 0.65 | 16.4 |
| 3200 | 6 | **8.30** | 0.63 | 12.5 |
| 5600 | 4 | 8.41 | 0.63 | 9.0 |

### 关键发现
- **基因块注意力是核心贡献**：去掉它 WD 从 8.30 升到 8.89；而把它从逐基因（$K=1$）换成块粒度（$K=3200$）带来约 191× 训练提速（2391.2→12.5 分钟），生成质量反而更好——说明粗粒度块既省算力又抓住了更有意义的结构。
- **块大小有甜点区**：$K=3200$（$L=6$）在 Heart 上 WD/MMD 最优；块太小（$K=600$）质量和效率都差，块太大（$K=5600$）质量略降但更快。
- **优势集中在复杂与未见场景**：在简单的 PBMC3K 上 CFGen 也有竞争力，但数据复杂度一升高、尤其在双/多条件和留出的未见条件组合下，SAVE 的鲁棒性拉开差距（多条件未见集 WD 4.63 仍最低）。

## 亮点与洞察
- **把 NLP 的语义嵌入当"先验知识注入"用得很巧**：基因没有空间结构，作者绕道用 LLM 读 NCBI 功能描述来度量基因语义相似度，相当于把人类积累的生物学文献知识压进了分块结构里——这是把基础模型常识迁到表格型组学数据的一个可复用范式。
- **用最优传输做等大聚类**：相比 K-means，OT 的均匀边际约束保证每块大小相等，让后续张量化和注意力计算规整高效，是个值得借鉴的工程+建模一体化技巧。
- **条件掩码统一生成与迁移**：同一个掩码机制，掩 Flow Matching 的条件就变成生成、掩 VAE 的条件就变成迁移/批次校正，一套训练目标覆盖多任务、还顺带获得对未见条件的外推能力。
- **"粗粒度 token"的思路可迁移**：任何高维、无序、稀疏的表格/组学数据（蛋白组、代谢组、空间转录组），只要能找到语义相似度就能照搬"语义聚块 + 块注意力"，既降算力又引入结构先验。

## 局限与展望
- **依赖文献/数据库标注的完备性**：LLM 块构建对研究充分的基因能给出丰富语境，但对注释稀疏的冷门或新发现基因，嵌入可能噪声大、信息少。
- **易受历史文献偏差影响**：相比 MSigDB / KEGG 这类人工策展、严格验证的通路库，纯文本驱动的无监督分组更容易继承文献里的历史偏见。
- **改进方向**：作者建议把结构化生物知识图谱或策展过的基因调控网络与 LLM 嵌入结合，给分块提供更可靠的监督。
- **自己的观察**：分块在训练前固定、不随任务自适应；不同任务/组织是否需要不同的块划分、能否让块结构端到端可学，文中未探索。块大小甜点区也依赖数据集，换数据可能要重新搜 $K$。

## 相关工作与启发
- **vs scVI（VAE 基线）**：scVI 用 ZINB 似然 + 协变量进隐空间，擅长表示学习与批次校正，但架构简单、难建模多条件组合交互；SAVE 在 VAE 基础上叠了块注意力 Transformer + Flow Matching 生成，多条件与未见组合下大幅领先。
- **vs scGPT / scBERT / Geneformer（Transformer 基础模型）**：它们把基因当扁平 token、靠 rank 或离散化表示、主要做编码；SAVE 改成语义块粒度、保留连续表达、并真正接成可采样的生成框架。
- **vs scDiffusion / CFGen（生成基线）**：两者都做条件生成但依赖细粒度扩散，CFGen 在双条件复杂度下退化到接近 scVI；SAVE 用更简单的仿射路径 Flow Matching + 块粒度 latent，保真度和可扩展性都更强。
- **vs scGen / trVAE / scDisInFact（条件迁移基线）**：scGen 假设隐空间线性平移、trVAE 用 MMD 对齐，多限于单一条件；scDisInFact 用多编码器解耦但需预定义因子、难扩到组合条件空间；SAVE 用条件掩码统一处理多因子、扰动预测平均 PCC > 0.95 超过它们。

## 评分
- 新颖性: ⭐⭐⭐⭐ "LLM 语义聚块 + 最优传输等大聚类 + 块注意力"组合新颖，把基础模型常识迁到单细胞建模有想法
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖条件生成/批次校正/扰动预测三大任务、多个数据集，含块大小与模块消融、效率对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰、图文配合好；个别表格编号（如正文提到的 Table 15）与正文略有错位
- 价值: ⭐⭐⭐⭐ 单卡可训、统一多任务、对未见条件外推强，对虚拟细胞合成与单细胞分析有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](../../ICML2026/computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](../../ICML2026/computational_biology/scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICLR 2026\] A Foundation Model with Multi-Variate Parallel Attention to Generate Neuronal Activity](a_foundation_model_with_multi-variate_parallel_attention_to_generate_neuronal_ac.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/computational_biology/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)

</div>

<!-- RELATED:END -->
