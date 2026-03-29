# The Platonic Universe: Do Foundation Models See the Same Sky?

**会议**: NeurIPS 2025
**arXiv**: [2509.19453](https://arxiv.org/abs/2509.19453)
**代码**: 无（使用公开模型和Multimodal Universe数据集）
**领域**: Physics / Foundation Models / Representation Learning
**关键词**: Platonic Representation Hypothesis, 基础模型, 天文学, 表征对齐, 跨模态收敛

## 一句话总结
在天文学场景下验证柏拉图表征假说（PRH）：使用JWST、HSC、Legacy Survey和DESI光谱数据，测量6种基础模型（ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer）的表征对齐度，发现模态内和跨模态MKNN分数随模型规模一致增加（p=3.31×10⁻⁵），支持不同架构和模态向共享表征收敛的假说。

## 研究背景与动机

1. **领域现状**：天文学正经历"第四波"AI应用——基础模型的涌入。多个团队分别探索对比学习、生成式、自回归等不同路线构建天文基础模型，但缺乏共识哪种架构最优。
2. **现有痛点**：天文观测本质上是同一宇宙物理的不同投影（光学成像、红外成像、光谱），但每种模态的模型通常独立设计和训练，跨模态知识利用不足。
3. **核心矛盾**：天文社区是否需要从零训练专用基础模型，还是可以复用通用视觉模型已投入的巨额GPU计算？
4. **本文要解决什么？** 在天文学场景下定量验证PRH——不同神经网络是否在足够数据和算力下收敛到一致的表征空间。
5. **切入角度**：天文学是PRH的理想测试场——不同观测模态是同一物理的数学投影，如果PRH成立，模型应该学到相似的表征。
6. **核心idea一句话**：即使在自然图像上预训练的通用视觉模型，其天文数据嵌入的跨模态对齐度也随模型规模显著增加，天文专用模型并无优势。

## 方法详解

### 整体框架
4种天文数据集（HSC/Legacy/JWST图像 + DESI光谱）× 6种模型架构 → 提取嵌入 → 模态内和跨模态MKNN对齐度测量 → 分析规模-对齐度关系。

### 关键设计

1. **数据模态选择**：
   - HSC：地面光学成像（z/r/g波段），作为参考基线
   - DESI Legacy Survey：不同地面光学巡天策略
   - JWST NIRCam：空间红外成像（F444W/F277W/F090W），最极端的成像测试
   - DESI光谱：1D光谱数据，与图像完全不同的模态
   - 使用Multimodal Universe (MMU)进行跨模态匹配

2. **模型架构覆盖**：
   - 监督分类：ViT (Base/Large/Huge), ConvNeXtv2 (Nano/Tiny/Base/Large)
   - 自监督KD：DINOv2 (Small/Base/Large/Giant)
   - 自监督预测：IJEPA
   - 天文专用自回归：AstroPTv2 (Small/Base/Large)，在DESI Legacy Survey上预训练
   - 天文光谱Transformer：Specformer，处理1D光谱

3. **MKNN对齐度度量**：
   - $\text{MKNN}(\mathbf{z}_1, \mathbf{z}_2) = k^{-1} |N_k(\mathbf{z}_1) \cap N_k(\mathbf{z}_2)|$
   - 模态内测试：同模态、同架构不同规模的嵌入对齐度
   - 跨模态测试：同架构同规模在不同模态上的嵌入对齐度
   - PRH预测：两者都应随模型规模增大而提升

## 实验关键数据

### 主实验 — 模态内对齐度（部分）

| 模型对 | JWST | Legacy | HSC |
|--------|------|--------|-----|
| AstroPTv2 S vs B | 49.7% | 8.1% | 10.3% |
| AstroPTv2 B vs L | 56.2% | 10.0% | 13.5% |
| DINOv2 L vs G | 40.2% | 10.2% | 10.9% |
| ViT L vs H | 32.6% | 4.4% | 5.0% |

### 统计检验

| 对齐类型 | 增长比例 | 二项检验p值 |
|---------|---------|------------|
| 模态内 | 14/18 (78%) | p = 1.54×10⁻² |
| 跨模态 | 28/33 (85%) | p = 3.31×10⁻⁵ |

### 关键发现
- **跨模态对齐显著增长**：28/33次跨模态比较中MKNN随模型规模增加，统计高度显著
- **通用模型≈天文专用模型**：AstroPTv2（天文专用）的对齐度并不显著优于DINOv2或ViT（自然图像预训练）
- **最极端跨模态也有效**：自然图像预训练模型在DESI光谱的Specformer嵌入上也显示出对齐增长趋势
- **JWST对齐度最高**：与HSC配对时，JWST的MKNN分数系统性高于Legacy Survey

## 亮点与洞察
- **"拿来主义"的科学论据**：天文社区不需要从零训练专用基础模型——复用ML社区已投入GPU-centuries的预训练模型再微调即可，大幅降低计算和碳排放成本
- **天文学作为PRH自然测试场**：不同观测模态是同一物理的数学投影，比自然图像域的测试更有物理根基
- **实用建议明确**："focus less on astronomy-specific architectures and more on scale and data diversity"

## 局限性 / 可改进方向
- 部分跨模态匹配数据量小（JWST vs HSC仅1.67K天体），可能不够代表性
- MKNN仅是一种对齐度量，未使用CKA、互信息等补充指标
- 未测试LLM、扩散模型等更多架构类型
- 对齐度量非因果推断——高MKNN不能直接证明模型"理解了相同物理"

## 相关工作与启发
- **vs PRH原文 (Huh et al., ICML 2024)**：原文是通用视觉的position paper，本文首次在科学数据（天文多模态）上定量验证
- **vs AstroLLaMA等天文LLM**：这些模型追求domain-specific训练，本文结果暗示这可能不必要
- **vs Multimodal Universe (MMU)**：本文依赖MMU的跨模态匹配基础设施，进一步证明了MMU平台的价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在科学领域系统验证PRH，视角独特
- 实验充分度: ⭐⭐⭐ 6种架构×4种模态覆盖广，但数据量有限
- 写作质量: ⭐⭐⭐⭐⭐ 叙述优美，从柏拉图洞穴寓言到天文观测的类比精彩
- 价值: ⭐⭐⭐⭐ 对天文基础模型社区的策略方向有直接指导意义
