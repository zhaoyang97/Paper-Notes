# Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**会议**: CVPR2025  
**arXiv**: [2603.13057](https://arxiv.org/abs/2603.13057)  
**代码**: [GitHub](https://github.com/litelightlite/VTON-IQA)  
**领域**: human_understanding  
**关键词**: virtual try-on, image quality assessment, human feedback, cross-attention, benchmark

## 一句话总结

提出 VTON-IQA，一个无参考的虚拟试穿图像质量评估框架，通过大规模人类标注基准 VTON-QBench（62,688 张试穿图 + 431,800 条标注）和 Interleaved Cross-Attention 模块实现与人类感知对齐的图像级质量预测。

## 研究背景与动机

1. **虚拟试穿评估困境**：真实场景中，同一人穿着目标服装的 ground-truth 图像通常不可用，reference-based 指标（SSIM、LPIPS）无法适用
2. **分布级指标的局限**：FID、KID 等指标衡量数据集级别统计特性，无法反映单张生成图像的感知质量
3. **现有方法缺陷**：VTONQA 数据集规模有限（748 对 vs 本文 13,153 对），VTON-VLLM 聚焦文本批评而非定量预测，VTBench 未从大规模人类标注中直接学习统一质量模型
4. **质量评估的特殊性**：虚拟试穿质量评估不同于传统单图 IQA，需要同时验证服装保真度和非目标区域（人物身份、背景等）的保持
5. **跨图像交互需求**：评估质量需要建模生成试穿图与输入服装图、人物图之间的交叉关系，传统 IQA 方法无法实现
6. **缺乏可复现基准**：现有方法缺乏公开实现和标准化评估基准，难以进行可复现的评估

## 方法详解

### 整体框架

三分支 Transformer 架构处理服装图 $I_G$、人物图 $I_P$ 和生成试穿图 $I_V$，前半层独立特征提取，后半层引入 Interleaved Cross-Attention (ICA) 进行跨图像交互建模。

### VTON-QBench 数据集构建

- **数据增强**：基于 FLUX.1-dev 生成合成服装-人物对，覆盖 casual/street/formal/minimal/vintage 风格，对数从 6,981 扩增至 13,153（约 1.9×）
- **伪三元组构建**：用强模型 Nano Banana Pro 生成伪 ground-truth，构建 $(I_G, I_P, I_R)$ 三元组，支持与 reference-based 指标的比较
- **14 个 VTON 模型生成**：涵盖 GAN-based（VITON-HD、HR-VITON、SD-VITON）、U-Net diffusion（IDM-VTON、CatVTON、OOTDiffusion）、DiT diffusion（FitDit、CatVTON-FLUX）、专有编辑模型（Nano Banana Pro、GPT-Image-1.5）
- **标注协议**：三级顺序量表（Unnatural / Slightly unnatural / Completely natural），13,838 名合格标注者提供 431,800 条标注
- **数据清洗**：两阶段过滤——dummy 任务一致性检查 + Krippendorff's α 阈值过滤（α ≤ 0.4 的问卷丢弃），α 从 0.286 提升至 0.550

### Interleaved Cross-Attention (ICA) 模块

- 在后 L/2 层标准 Transformer 块中，在 self-attention 和 MLP 之间插入 cross-attention 层
- **非对称交互设计**：试穿图表示聚合来自服装和人物的贡献 $\hat{X}_V = \tilde{X}_V + C_{V \leftarrow G} + C_{V \leftarrow P}$，而服装/人物分支仅从试穿分支获取信息
- 避免不必要的 G↔P 耦合，强调质量判断以生成试穿图为中心

### 评分模块

- 提取各分支 [CLS] token 作为紧凑全局表示 $c_G, c_P, c_V$
- 中间关系分数：可学习权重 $\alpha$ 平衡服装一致性和非目标区域保持的余弦相似度
- $\tilde{s} = \alpha \frac{c_G^\top c_V}{\|c_G\|\|c_V\|} + (1-\alpha) \frac{c_P^\top c_V}{\|c_P\|\|c_V\|}$
- 最终分数：$\hat{s} = \tanh(a\tilde{s}+b)$，可学习仿射变换 + tanh 约束至 $[-1,1]$

### 损失函数

- **配对偏好项**：Bradley-Terry 模型建模配对偏好，soft-label cross-entropy 对齐预测和人类偏好分布
- **分数回归项**：L2 loss 强制预测分数与人类评分的一致性
- 联合优化兼顾相对排序和绝对分数对齐

## 实验关键数据

| 方法 | ρ_SRCC↑ | ρ_PLCC↑ | R²↑ | A_macro↑ | A_micro↑ |
|------|---------|---------|------|----------|----------|
| SSIM | – | 0.135 | – | 0.596 | 0.593 |
| LPIPS | – | 0.387 | – | 0.701 | 0.695 |
| DINOv3 (zero-shot) | – | 0.261 | – | 0.637 | 0.641 |
| VTON-IQA w/o ICA | 0.617 | 0.615 | 0.372 | 0.722 | 0.747 |
| **VTON-IQA** | **0.750** | **0.751** | **0.553** | **0.781** | **0.790** |
| Human | 0.760 | 0.762 | 0.536 | 0.782 | 0.791 |

**关键发现**：
- ICA 模块带来显著提升（SRCC: 0.617→0.750），验证跨图像交互建模的有效性
- 在配对精度 (A_macro/A_micro) 上模型接近人类水平（0.781 vs 0.782）
- 相关性指标仍有提升空间（0.750 vs 0.760），说明细粒度感知对齐尚需改进
- 14 个 VTON 模型 benchmark 中，Nano Banana Pro 在 Dress Code 和 VITON-HD 上综合得分最高
- GAN-based 方法（VITON-HD、HR-VITON、SD-VITON）在 VTON-IQA 评分中显著落后于 diffusion 方法
- 定性分析表明 VTON-IQA 对姿态和缩放变化具有鲁棒性，而 SSIM/LPIPS 会过度惩罚全局变换

## 亮点

1. **规模空前的人类标注基准**：VTON-QBench 是目前最大的虚拟试穿人类主观评估数据集，标注者数量（13,838）和标注条数（431,800）远超前作
2. **非对称 ICA 设计**：精准建模 V↔G 和 V↔P 交互而避免冗余 G↔P 耦合，与虚拟试穿质量评估的本质对齐
3. **接近人类的配对精度**：在 pairwise ranking 任务上达到与人类可比的一致性
4. **全面 benchmark**：涵盖 14 个代表性 VTON 模型的系统评估，为社区提供标准化参考

## 局限性

1. 相关性指标与人类表现仍有差距（SRCC 0.750 vs 0.760），细粒度感知对齐待提升
2. 标注量表仅三级，粒度较粗，可能限制质量分数的区分度
3. 训练和评估均基于 VTON-QBench，泛化到全新 VTON 模型或极端场景的能力待验证
4. 仅评估上半身/全身试穿，未覆盖配饰、鞋类等细分场景
5. 骨干网络基于 DINOv3 ViT-L/16，推理成本较高（需处理三张图的三分支前向传播）

## 相关工作

- **虚拟试穿**：从 GAN-based 两阶段流程（VITON-HD、HR-VITON）→ U-Net diffusion（IDM-VTON、CatVTON）→ DiT（FitDit、Any2AnyTryon）→ 专有编辑模型（Nano Banana Pro、GPT-Image-1.5）
- **质量评估**：VTONQA（有限规模标注训练评估器）、VTON-VLLM（文本批评导向）、VTBench（层次化 benchmark 但无统一质量模型）
- **通用 IQA**：Q-Align、CLIP-IQA（单图 IQA，未建模跨图像交互）

## 评分

- 新颖性: ⭐⭐⭐⭐ — ICA 非对称交互设计针对性强，大规模众包标注数据集构建流程完善
- 实验充分度: ⭐⭐⭐⭐⭐ — 14 模型 benchmark、消融实验、与人类对比全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ — 为虚拟试穿社区提供了急需的标准化评估工具和大规模 benchmark
