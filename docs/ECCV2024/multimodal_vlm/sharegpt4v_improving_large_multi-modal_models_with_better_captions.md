# ShareGPT4V: Improving Large Multi-Modal Models with Better Captions

**会议**: ECCV 2024  
**arXiv**: [2311.12793](https://arxiv.org/abs/2311.12793)  
**代码**: [https://ShareGPT4V.github.io](https://ShareGPT4V.github.io)  
**领域**: 多模态VLM  
**关键词**: 高质量caption, 数据增强, 多模态对齐, GPT4-Vision, 预训练数据

## 一句话总结

ShareGPT4V 构建了一个120万条高质量描述性caption数据集（由GPT4-Vision生成100K种子 + Share-Captioner扩展至1.2M），通过在预训练和SFT两阶段使用该数据集训练LLaVA架构的模型ShareGPT4V-7B，在11个多模态benchmark中9个取得最优，证明了高质量caption是LMM模态对齐的关键瓶颈。

## 研究背景与动机

1. **领域现状**：当前大型多模态模型（LMM）遵循"预训练对齐 + SFT微调"的双阶段范式，在多模态理解方面取得了显著进展。
2. **现有痛点**：主流image-text数据集中的caption普遍简短、只聚焦于显著目标（如COCO-Caption平均仅52字符），导致视觉中丰富的细粒度语义信息被大量压缩丢失。
3. **核心矛盾**：视觉模态天然包含丰富信息（世界知识、物体属性、空间关系、美学评价等），但现有caption的信息量远不足以支撑有效的模态对齐。LLaVA-Instruct虽使用GPT4但模型并未"看到"图像，依赖人工标注+想象，不可避免导致幻觉。
4. **本文解决的问题**：如何大规模、低成本地获取高质量图像描述来改善LMM各训练阶段的模态对齐效果。
5. **切入角度**：先用GPT4-Vision直接对图像生成100K高质量caption（平均942字符），再训练一个caption模型（Share-Captioner）扩展至1.2M规模。
6. **核心idea**：高质量caption是多模态对齐的银弹——即使不改架构，仅替换3.5%的SFT数据为高质量caption就能显著提升性能。

## 方法详解

### 整体框架

Pipeline: 多来源图像 → 数据特定prompt设计 → GPT4-Vision生成100K种子caption → 训练Share-Captioner → 扩展生成1.2M caption → 用于LMM的预训练和SFT阶段

模型架构沿用LLaVA-1.5：CLIP-Large (336×336) → 2层MLP Projector → Vicuna-v1.5 (7B)

### 关键设计

1. **数据特定的Prompt工程**:
   - 做什么：为不同数据来源设计专用prompt，引导GPT4-Vision生成高内容关联性描述
   - 核心思路：基础prompt（物体属性、外观、空间关系）+ 数据源专用prompt（如地标图→地理位置和名称，名人图→身份信息）+ 美学评价prompt
   - 设计动机：不同来源的图像关注点不同，通用prompt无法捕获domain-specific知识（如埃菲尔铁塔不应只被描述为"高铁塔"）

2. **Share-Captioner（caption模型训练与数据扩展）**:
   - 做什么：用100K GPT4-Vision caption微调一个caption模型，替代昂贵的GPT4-Vision进行大规模caption生成
   - 核心思路：在100K多样化高质量caption上微调后，Share-Captioner可以用统一instruction生成高质量caption，不再需要数据源特定prompt
   - 设计动机：GPT4-Vision成本高昂，需要一个本地化的caption模型实现规模化扩展。人工评估表明Share-Captioner与GPT4-Vision质量相当（38.2% vs 35.3%偏好，26.5%持平）
   - 扩展规模：1.2M图像，约44 A100 GPU天

3. **预训练阶段的Vision Encoder解冻策略**:
   - 做什么：在预训练阶段同时微调vision encoder、projector和LLM
   - 核心思路：由于caption质量足够高，解冻vision encoder后半部分（从block 12开始）能让encoder学习生成与caption细节对应的视觉embedding，实现双向对齐
   - 设计动机：之前的LMM工作不会在预训练阶段微调vision encoder，因为低质量caption可能导致视觉知识退化。但高质量caption下解冻更有利。
   - 关键发现：解冻后半（12层起）效果最优，MME提升52.2分，MMB提升2.2%

### 损失函数 / 训练策略

- **预训练阶段**：使用ShareGPT4V-PT (1.2M caption)，学习率 $2e{-5}$，batch size 256，约4700 steps。同时微调vision encoder后半、projector和LLM。
- **SFT阶段**：沿用LLaVA-1.5的665K SFT数据，仅替换其中23K detailed description为ShareGPT4V的高质量caption。冻结vision encoder，微调projector和LLM。学习率 $2e{-5}$，batch size 128，约5200 steps。

## 实验关键数据

### 主实验

| Benchmark | 指标 | ShareGPT4V-7B | LLaVA-1.5-13B | Qwen-VL-Chat-7B | 提升 |
|-----------|------|---------------|---------------|-----------------|------|
| MME-P | 感知得分 | **1567.4** | 1531.3 | 1487.5 | +36.1 vs 13B |
| MME-C | 认知得分 | **376.4** | 295.4 | 360.7 | +15.7 vs Qwen |
| MMBench | 准确率 | **68.8%** | 67.7% | 60.6% | +1.1% |
| SEED-I | 准确率 | **69.7%** | 68.2% | 58.2% | +1.5% |
| MM-Vet | GPT评分 | **37.6** | 35.4 | - | +2.2 |
| LLaVA-Wild | GPT评分 | **72.6** | 70.7 | - | +1.9 |
| VQA-v2 | 准确率 | **80.6%** | 80.0% | 78.2% | +0.6% |
| VizWiz | 准确率 | **57.2%** | 53.6% | 38.9% | +3.6% |

7B模型在11个benchmark中9个超越所有竞争对手（包括13B模型和使用14亿训练样本的Qwen-VL-Chat）。

### 消融实验

| 配置（预训练/SFT使用ShareGPT4V） | MME-P | MMBench | SEED-I | 说明 |
|----------------------------------|-------|---------|--------|------|
| ✗/✗ (baseline LLaVA-1.5) | 1510.7 | 64.3% | 66.2% | 基线 |
| ✗/✓ | 1542.1 | 66.8% | 66.7% | SFT单独使用，+31.4 |
| ✓/✗ | 1557.2 | 67.4% | 68.5% | 预训练单独使用，+46.5 |
| ✓/✓ | **1567.4** | **68.8%** | **69.7%** | 两阶段均使用，最优 |

| Vision Encoder解冻起始层 | 显存占用 | MME-P | MMBench | SEED-I |
|--------------------------|----------|-------|---------|--------|
| 24（全冻结） | 49.6 GB | 1515.2 | 66.6% | 68.1% |
| 12（后半解冻） | 56.7 GB | **1567.4** | **68.8%** | **69.7%** |
| 0（全解冻） | 63.6 GB | 1545.7 | 68.5% | 69.2% |

### 关键发现

- 仅替换**3.5%**的SFT数据为高质量caption就能在多个LMM上带来一致提升（LLaVA: MME+222.8, LLaVA-1.5-13B: MME+22.0, Qwen-VL-Chat: MME+22.3）
- 预训练阶段的caption质量影响更大：使用相同图像，ShareGPT4V-PT caption比BLIP caption在MME多提升18.2分
- 高质量caption使模态对齐可以以相对轻量级的数据规模实现——100K数据就有显著提升，1000K后逐渐饱和
- 解冻后半vision encoder（而非全部或完全冻结）是最优策略

## 亮点与洞察

- **数据质量 > 模型架构**：本文的核心message是"在简单架构和少量参数下，仅靠高质量数据就能超越大模型"，这是一个具有普遍指导意义的洞察
- **Share-Captioner的蒸馏思路**：用GPT4-Vision的100K数据训练一个替代caption模型，是一种经典的"强模型蒸馏 → 弱模型扩展"范式
- **数据源多样化prompt设计**：针对不同图像来源设计prompt是提升caption覆盖度的有效手段
- **Vision Encoder解冻策略**：在高质量数据场景下，解冻后半encoder是一个可迁移的training trick

## 局限性 / 可改进方向

- Caption生成依赖GPT4-Vision，初始100K的成本依然不低
- 仅在7B规模验证，未探索更大模型或更大数据规模的scaling law
- Share-Captioner的质量上界受限于GPT4-Vision，无法超越teacher模型
- 数据集虽然多样，但仍以自然图像为主，对文档、图表等场景覆盖有限
- 未探索caption与其他类型训练数据（如grounding、OCR）的交互效果

## 相关工作与启发

- **vs LLaVA-Instruct**：LLaVA让GPT4"想象"图像内容生成caption，不可避免产生幻觉；ShareGPT4V让GPT4-Vision直接看图描述，准确性更高
- **vs LaCLIP/VeCLIP**：这些工作尝试用LLM重写/融合caption来增强CLIP训练，但受限于原始caption质量和LLM幻觉；ShareGPT4V从源头产生高质量描述
- **vs BLIP-LCS**：BLIP生成的caption平均54字符，ShareGPT4V平均942字符，信息密度差异巨大

## 评分

- 新颖性: ⭐⭐⭐☆☆ 方法本身并不复杂（用强模型生成好数据），但"数据质量决定模态对齐上限"的洞察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 11个benchmark全面评估，消融实验系统（数据阶段、数据量、encoder解冻层数、caption质量对比），非常详尽
- 写作质量: ⭐⭐⭐⭐☆ 逻辑清晰、图表丰富，故事线完整
- 价值: ⭐⭐⭐⭐⭐ ShareGPT4V数据集已成为后续多个LMM工作的标准预训练数据，实际影响力巨大
