# Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2501.05179](https://arxiv.org/abs/2501.05179)  
**代码**: [https://github.com/xuyang-liu16/GlobalCom2](https://github.com/xuyang-liu16/GlobalCom2)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: Token压缩, 高分辨率VLM, 动态裁剪, 即插即用加速, 全局-局部引导  

## 一句话总结

提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

## 背景与动机

高分辨率LVLM（如LLaVA-NeXT、InternVL3）普遍采用**动态裁剪**策略：将高分辨率图像拆解为一张全局缩略图+多张局部裁剪图，分别用ViT编码后拼接。这确实提升了细粒度理解能力，但代价是视觉token数量暴增（LLaVA-NeXT为5×576，LLaVA-OV可达10×729），导致LLM推理的二次复杂度成为瓶颈。

现有token压缩方法（FastV、SparseVLM、PruMerge等）主要为**单视图（single-view）**VLM设计，直接应用于动态裁剪的HR-LVLM时存在三个关键问题：
1. **忽略全局上下文**：不利用缩略图的全局信息来评估各crop的重要性
2. **信息丰富度中差异不敏感**：不同crop的语义密度差异巨大（如上半区有球员，下半区只有草地），但现有方法一视同仁地均匀压缩
3. **位置偏差（positional bias）**：FastV等基于LLM注意力的方法会系统性地给后面位置的crop分配更多token，与实际内容重要性无关——在极端压缩下甚至导致严重的多模态幻觉（POPE掉14.8分）

## 核心问题

**如何在动态裁剪的层级化视觉结构中，实现内容感知的差异化token压缩？** 核心挑战在于：缩略图和crop扮演不同角色（前者提供整体语境、后者提供细节），不同crop信息密度差异显著，现有方法要么不区分要么因位置偏差反而压缩错了区域。

## 方法详解

### 整体框架

GlobalCom²的设计理念是"**全局到局部**"（global-to-local）的层级压缩，灵感来自人类视觉的"先抓大意、再看细节"过程。整个框架在ViT编码后、送入LLM前执行，是一个**vision encoding阶段的即插即用模块**，由两条路径组成：

- **蓝色路径**（缩略图压缩）：基于[CLS]注意力分数对thumbnail token做TopK保留
- **黄色路径**（crop压缩）：全局引导的两阶段crop压缩
  - (a) **自适应压缩调整**：根据每个crop在全局视角下的信息丰富度，动态分配不同的保留比例
  - (b) **综合token评估**：结合全局和局部两个视角评估每个token的重要性，执行TopK保留

### 关键设计

1. **缩略图压缩（Thumbnail Compression）**：利用ViT最后一层[CLS] token与所有patch token的注意力分数作为重要性分数$s_i^G$，保留Top-$k$（$k=R \times N$）个token。这一步比较常规，关键在于缩略图的注意力分布会被后续复用来引导crop压缩。

2. **自适应压缩调整（Adaptive Compression Adjustment）**：这是GlobalCom²的核心创新。将每个crop对应到缩略图上的区域，累加该区域内的[CLS]注意力分数得到crop级别的**信息丰富度分数**$s_j^G = \sum_{i \in \text{crop}_j} s_i^G$。然后通过softmax归一化（温度$\tau=10$）得到相对重要性权重$\sigma_j$，每个crop的保留比例调整为：
   $$r_j = R \times (1 + \sigma_j - \frac{1}{n})$$
   这确保了信息丰富的crop保留更多token，冗余crop被更激进地压缩，且所有crop的总token数仍满足预设比例$R$。

3. **综合token评估（Holistic Token Evaluation）**：对每个crop中的每个token，综合两个视角的重要性分数：
   - **局部分数** $s_{j,i}^L$：crop内部[CLS]与patch的注意力分数，捕捉局部显著性
   - **全局分数** $\hat{s}_{j,i}^G$：将缩略图的1D注意力scores reshape为2D并双线性插值到原始分辨率，取对应crop区域作为全局重要性
   - 综合分数：$s_{j,i} = \alpha \cdot \hat{s}_{j,i}^G + (1-\alpha) \cdot s_{j,i}^L$（$\alpha=0.5$）

4. **无[CLS]模型适配（针对SigLIP等）**：对于LLaVA-OneVision使用的SigLIP（无[CLS] token），提出用**负余弦相似度**替代：计算所有token的全局均值向量$\mathbf{g}$，与全局均值相似度低的token信息独特性更强（$s_i = -\cos(\mathbf{x}_i, \mathbf{g})$）。实验验证其效果接近[CLS]方案。

5. **视频理解扩展**：将"全局缩略图→局部crop"的逻辑类比为"全局视频表示→各帧"，通过全局平均池化得到全局表示，按帧自适应分配压缩强度。

### 损失函数 / 训练策略

**不涉及训练**。整体方法是training-free的即插即用方案，只有两个超参数：温度$\tau=10$（控制crop间分配的锐度）和全局-局部混合系数$\alpha=0.5$。实验表明对这两个参数不敏感。

## 实验关键数据

**LLaVA-NeXT-7B 主结果（Table 1）**：

| 保留比例 | 方法 | GQA | VQAT | POPE | MME | MM-Vet | 均分(%) |
|---------|------|-----|------|------|-----|--------|--------|
| 100% | 原始 | 64.2 | 64.9 | 86.5 | 1519.0 | 43.9 | 100.0% |
| 50% | FastV | 61.8 | 59.6 | 85.5 | 1490.3 | 37.6 | 95.5% |
| 50% | **GlobalCom²** | **63.9** | **62.3** | **88.1** | **1552.9** | **40.4** | **98.5%** |
| 25% | SparseVLM | 59.9 | 58.3 | 85.0 | 1465.9 | 38.5 | 94.6% |
| 25% | **GlobalCom²** | **61.5** | **60.9** | **87.6** | **1493.5** | **40.7** | **96.7%** |
| 10% | FastV | 55.9 | 55.7 | 71.7 | 1282.9 | 27.2 | 85.4% |
| 10% | FasterVLM | 56.9 | 56.5 | 83.6 | 1359.2 | 35.0 | 89.9% |
| 10% | **GlobalCom²** | **57.1** | **58.4** | **83.8** | **1365.5** | **36.4** | **91.6%** |

**LLaVA-OneVision**（Figure 6）：R=10%时保持90.5%性能，仅消耗35.4%原始GPU内存。

**效率分析（Table 4，R=10%）**：

| 方法 | TFLOPs | Memory | Throughput | 性能 |
|------|--------|--------|-----------|------|
| 原始 | 41.7 | 23.0GB | 3.8 samples/s | 100% |
| SparseVLM | 5.4(↓87%) | 24.2(↑5.2%) | 5.9(1.6×) | 85.7% |
| FasterVLM | 3.8(↓91%) | 13.6(↓40%) | 6.7(1.8×) | 89.5% |
| **GlobalCom²** | **3.8(↓91%)** | **13.9(↓40%)** | **6.7(1.8×)** | **90.8%** |

注意SparseVLM由于需要显式注意力矩阵，不兼容FlashAttention，内存反而上升。

**与question-aware方法组合（R=10%）**：
- +FastV → 均分提升5.3%，POPE提升8.2
- +SparseVLM → 均分提升5.2%，POPE提升4.5

### 消融实验要点

- **自适应压缩调整策略**（Table 2）：Softmax(sum) > Softmax(max) > n_top-k > Uniform，最优策略比均匀压缩提升1.4%均分。说明基于crop整体信息量的分配优于只看最强token。
- **token评估来源**（Table 3）：Global+Local(96.7%) > Local only(95.6%) > Global only(94.7%)。单独用局部分数在细粒度任务(VQAT, POPE)好，单独用全局分数在通用感知(MME, SQA)好，两者互补。
- **无[CLS]替代方案**（Table 5）：$s_i^{sim}$（负余弦相似度）效果接近$s_i^{[CLS]}$（95.8% vs 96.4%），远优于负patch注意力$s_i^{attn}$（93.2%）。
- **超参数鲁棒性**（Figure 10）：$\tau \in [5, 20]$和$\alpha \in [0.3, 0.7]$范围内表现稳定。

## 亮点

- **"全局到局部"的压缩哲学**：这是一个非常直觉的设计——先看全局再决定局部压缩力度，模拟人类视觉的从粗到精。简单有效，insight清晰。
- **诊断了位置偏差bug**：系统性揭示了FastV等方法在HR-LVLM上的positional bias问题，无论crop输入顺序如何（正序/倒序），后面位置的token总是获得更高注意力分数。这一发现本身就很有价值。
- **真正的即插即用**：在ViT编码后、LLM前操作，不需要修改模型结构，不需要训练，兼容FlashAttention——实用性极强。
- **扩展性好**：在无[CLS]模型和视频理解上都展示了适配方案，通用性较强。
- **极端压缩下优势突出**：R=10%时相比其他方法优势最大（比第二名高1.7%均分），在别人"崩溃"的场景下保持鲁棒。

## 局限性 / 可改进方向

- **仅限vision encoding阶段压缩**：虽然这保证了FlashAttention兼容性，但也意味着无法利用text query信息做question-aware的压缩。作者通过组合实验（+FastV/SparseVLM）缓解了这一限制，但两阶段并行还有优化空间。
- **超参数依赖**：$\tau$和$\alpha$虽然不太敏感，但对不同模型/任务的最优值可能不同，且$\alpha=0.5$的等权混合显得粗糙——可以学一个task-adaptive的$\alpha$。
- **仅评估了LLaVA系列**：没有在InternVL、Qwen2-VL等更多主流HR-LVLM上系统评测（Qwen2-VL仅在视频任务做了），普适性有待验证。
- **crop粒度的自适应偏粗**：自适应调整是在crop级别而非更细粒度（如sub-region级别），当单个crop内部信息密度差异大时（如crop左半有文字、右半是背景），无法做进一步差异化。
- **没有与token merging方法对比或结合**：与ToMe等merging策略的交互未探讨，直接丢弃不重要token可能损失信息，合并是更soft的策略。
- **全局均值替代[CLS]的理论基础薄弱**：$s_i = -\cos(\mathbf{x}_i, \mathbf{g})$虽然实验有效，但缺乏理论解释——为什么偏离均值就等于信息丰富？这和信息瓶颈理论可能有更深的联系。

## 与相关工作的对比

| 方法 | 压缩阶段 | 感知dynamic cropping | 训练 | FlashAttn兼容 | R=10%性能 |
|------|---------|---------------------|------|-------------|-----------|
| FastV (ECCV'24) | LLM pre-filling | ✗ | Free | ✗ | 85.4% |
| SparseVLM (ICML'25) | LLM pre-filling | ✗ | Free | ✗ | 86.1% |
| FasterVLM (2024.12) | Vision encoding | ✗ | Free | ✓ | 89.9% |
| PruMerge (ICCV'25) | Vision encoding | ✗ | Free | ✓ | 80.6% |
| **GlobalCom² (AAAI'26)** | **Vision encoding** | **✓** | **Free** | **✓** | **91.6%** |

与FasterVLM对比最为典型：两者都在vision encoding阶段用[CLS]注意力，但FasterVLM均匀压缩所有crop，GlobalCom²引入全局引导的差异化压缩，在R=10%下高出1.7%均分。在LLaVA-NeXT-13B上，从R=25%到R=10%时，FasterVLM的MMB下降5.9而GlobalCom²仅下降2.9，退化鲁棒性显著更好。

与FastV/SparseVLM的本质区别在于压缩阶段和位置偏差问题——后两者在LLM内部用注意力评估token重要性，导致位置偏差和FlashAttention不兼容。

## 启发与关联

**与ideas/目录的关联**：

1. **[Cross-Layer Token Budget Allocation](../../../ideas/model_compression/20260318_cross_layer_token_budget_allocation.md)**：该idea关注跨层的token预算分配，而GlobalCom²关注跨crop的预算分配。两者高度互补——GlobalCom²的自适应调整可以作为"层间"分配策略的"层内"子模块。**具体组合方案**：在每一层使用GlobalCom²的crop-level budgeting，同时跨层使用learned budget allocation，形成"层间+crop间"的双重自适应。

2. **[Task-aware Token Compression](../../../ideas/model_compression/20260316_task_aware_token_compression.md)**：GlobalCom²的全局-局部混合系数$\alpha$是固定的，而该idea提出多目标token重要性评分。可以将$\alpha$替换为task-adaptive的混合机制——理解任务增大全局权重、检测/grounding任务增大局部权重。

3. **[Adaptive Multi-Granularity KV Compress](../../../ideas/model_compression/20260318_adaptive_multi_granularity_kv_compress.md)**：GlobalCom²在token维度压缩，该idea在KV cache维度压缩。两者可以级联——先用GlobalCom²在vision encoding阶段减少token数，再用KV cache压缩进一步减少内存占用。

**新启发**：
- GlobalCom²的"负余弦相似度"方案（$s_i = -\cos(\mathbf{x}_i, \mathbf{g})$）本质上是在衡量token的**信息独特性/不可替代性**。这和信息瓶颈理论中的"信息量=编码到目标的互信息"有关联——可以考虑用信息瓶颈框架严格化这个直觉，为ideas/中的IB-based token compression提供理论支撑。
- 位置偏差的发现提示：任何在LLM内部做question-aware压缩的方法都可能受此影响，应该在评估时加入"crop顺序对照实验"作为标准sanity check。

## 评分

- 新颖性: ⭐⭐⭐⭐ 全局引导局部的分层压缩思路清晰且创新，但[CLS]注意力作为importance的使用并非新颖（FasterVLM已有）；核心贡献在于"为HR-LVLM量身定制"的系统化设计
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型（7B/13B/0.5B）、多比例（75%/50%/25%/10%）、消融完善（策略/分数来源/超参数/组合性/视频/效率），可视化也做得很好
- 写作质量: ⭐⭐⭐⭐ 分析部分（Section 3）做得很扎实，先观察再设计的叙事逻辑清楚；发现positional bias这个bug的分析很有说服力
- 价值: ⭐⭐⭐⭐ 实用价值高——training-free、plug-and-play、兼容FlashAttention，工业界可直接用；学术上为HR-LVLM的token压缩开辟了"结构感知"的新方向
