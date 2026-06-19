---
title: >-
  [论文解读] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling
description: >-
  [AAAI 2026][3D视觉][Text-to-CAD] 首次提出基于NURBS表面表示的文本到CAD生成框架NURBGen，通过微调LLM将自然语言描述转换为结构化的NURBS参数JSON，并引入混合表示（untrimmed NURBS + 解析原语）和大规模partABC数据集，在几何保真度和尺寸精度上显著超越现有方法。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "Text-to-CAD"
  - "NURBS"
  - "LLM"
  - "BRep"
  - "3D生成"
---

# NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling

**会议**: AAAI 2026  
**arXiv**: [2511.06194](https://arxiv.org/abs/2511.06194)  
**代码**: 即将公开  
**领域**: 3D视觉  
**关键词**: Text-to-CAD, NURBS, LLM, BRep, 3D生成

## 一句话总结

首次提出基于NURBS表面表示的文本到CAD生成框架NURBGen，通过微调LLM将自然语言描述转换为结构化的NURBS参数JSON，并引入混合表示（untrimmed NURBS + 解析原语）和大规模partABC数据集，在几何保真度和尺寸精度上显著超越现有方法。

## 研究背景与动机

### 问题定义

CAD建模在现代工程和产品设计中至关重要，但创建详细的CAD模型通常需要专业软件知识（如Onshape、AutoCAD），且极为耗时。Text-to-CAD技术旨在让设计师通过自然语言描述3D物体，无需专业建模技能。

### 现有方法的局限

**设计历史依赖**：几乎所有现有方法（如DeepCAD、Text2CAD、CAD-LLaMA）都依赖设计历史表示（design-history-based），其中形状通过参数化操作序列（挤出、2D草图）构造。虽然直观且可编辑，但训练数据集（如DeepCAD-170k）规模小、复杂度低（大多为长方体、圆柱体），限制了泛化能力。

**ABC数据集未被充分利用**：ABC数据集包含超100万个3D CAD模型，但有两个关键限制：(a) 以BRep（边界表示）形式存储，缺乏设计历史；(b) 缺少高质量文本描述。

**NURBS建模困难**：BRep中的解析表面最常用NURBS表示，但NURBS在深度生成研究中很少被探索，原因包括：高效表示的挑战、节点向量的不可微性、参数变异性高、修剪复杂性等。

### 本文动机

- 将NURBS表面视为语言对齐的对象，将每个表面编码为包含控制点、度数、权重和节点向量的JSON token序列
- 将text-to-CAD问题转化为语言建模任务
- 利用ABC数据集的大规模和几何多样性优势

## 方法详解

### 整体框架

NURBGen的整体流程：(1) 从ABC数据集提取零件级CAD模型 → (2) 将每个零件用混合格式编码（NURBS + 解析原语），并用VLM生成高质量描述 → (3) 微调Qwen3-4B将文本描述映射到结构化的混合CAD表示 → (4) 输出的JSON可直接转换为BRep模型。

### 关键设计

#### 1. **CAD表示（NURBS参数提取）**

- **归一化**：将几何体归一化到以原点为中心的 $2 \times 2 \times 2$ 边界框内
- **NURBS转换**：使用PythonOCC的`BRepBuilderAPI_NurbsConvert`将每个面转换为untrimmed NURBS表示，统一所有底层表面
- **参数提取**：对每个面提取控制点（poles）、两个参数方向的节点向量、节点重数、u和v方向的度数、有理权重和周期性标志
- **精确重建**：有了这些参数，原始表面可通过`Geom_BSplineSurface`构造器精确重建

NURBS曲面的数学定义为：

$$\mathbf{S}(u,v) = \frac{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}\mathbf{P}_{ij}}{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}}$$

#### 2. **混合表示（Hybrid Representation）**

核心创新之一。并非所有表面都能被untrimmed NURBS鲁棒表示——特别是孔洞或圆角附近的薄区域常引入几何伪影。

- **退化检测**：比较每个重建表面 $f_n$ 与真实表面 $f_{gt}$ 之间的Chamfer Distance：$CD(f_n, f_{gt}) \leq \epsilon$，阈值 $\epsilon = 6 \times 10^{-4}$
- **回退策略**：当NURBS近似不可接受时，保留原始的解析原语（lines、circles、B-splines、ellipses、parabolas、hyperbolas）
- **统计**：实践中约70%的面用NURBS建模，30%回退到解析原语
- **优势**：比纯NURBS格式更具表现力和紧凑性，减少参数量，产生更短、更token高效的输入

#### 3. **自动标注流水线**

解决ABC数据集缺少文本描述的问题：

- **多视角渲染**：每个BRep先转为三角网格，在Blender中从6个视角以512×512分辨率渲染，启用Freestyle渲染器叠加轮廓边
- **元数据引导**：提取VLM无法直接获取的几何元数据——长宽高、表面积、体积、拓扑孔洞数（通过Euler-Poincaré公式计算genus $g = 0.5 \times (2 - \chi)$）
- **描述生成**：使用InternVL3-13B多视角VLM，输入6个渲染视图和元数据增强的标注提示，生成shape-centric描述
- **质量验证**：GPT-4o对1000个随机样本验证准确率约85%

### 数据集构建（partABC）

- 从ABC的20万个模型中提取零件级子结构，获得300万个零件级CAD实例
- **复杂度过滤**：使用加权评分函数 $w(B) = l_1 \times \text{token\_count} + l_2 \times \text{through\_holes} + l_3 \times \frac{\text{surface\_area}}{\text{volume}} + l_4 \times \text{bbox\_diag}$
- 将零件分为simple（≤0.12）、moderate（0.12-0.23）、complex（>0.23）三级
- 保留10% simple + 50% moderate + 40% complex，最终约30万高质量样本

### 训练策略

- **基座模型**：Qwen3-4B
- **优化器**：AdamW，学习率 $5 \times 10^{-5}$，线性warmup
- **LoRA**：rank=64，$\alpha=128$
- **训练**：180k步，batch size=1，4×H200 GPU，3天
- **上下文窗口**：训练8192，推理14k
- **温度**：0.3
- **生成速度**：RTX 3090上约800 tokens/s
- **数据处理**：控制点坐标保留6位小数；权重用(value, frequency)压缩

## 实验关键数据

### 主实验

| 模型 | User偏好(1k)↑ | GPT偏好↑ | 无效率IR↓ | CD↓ | HD↓ | JSD↓ | MMD↓ |
|------|---------|--------|-------|------|------|------|------|
| GPT-4o | 1.5 | 1.9 | 0.17 | 7.2 | 0.36 | 72.87 | 4.17 |
| DeepCAD | 5.6 | 6.1 | 0.32 | 10.28 | 0.45 | 89.77 | 4.43 |
| Text2CAD | 26.1 | 27.2 | 0.05 | 9.66 | 0.42 | 85.27 | 4.54 |
| **NURBGen** | **64.1** | **61.6** | **0.018** | **4.43** | **0.25** | **57.94** | **2.14** |

注：CD、JSD、MMD均乘以 $10^2$。NURBGen在所有指标上大幅领先。

### 消融实验

| 配置 | 人类偏好↑ | GPT-4o偏好↑ | 说明 |
|------|----------|-----------|------|
| NURBS-only | 28% | 21% | 仅使用untrimmed NURBS，无解析原语回退 |
| **Hybrid（完整）** | **72%** | **79%** | 混合NURBS+解析原语 |

NURBS-only模型在孔洞、尖锐过渡和NURBS拟合不精确的区域出现明显几何伪影和重建错误。

### 关键发现

1. NURBGen在7500个测试样本上的CD仅为4.43（×$10^2$），比第二名Text2CAD（9.66）低54%
2. 人类评估中64.1%的top-1偏好率，远超Text2CAD的26.1%
3. 无效率仅0.018，表明生成的BRep具有很强的几何正确性
4. 混合表示相比纯NURBS在人类评估中提升44个百分点

## 亮点与洞察

1. **NURBS即语言**：将NURBS表面参数序列化为JSON token，将CAD生成优雅地转化为语言建模任务，这是一个重要的范式转移
2. **混合表示的实用性**：70% NURBS + 30%解析原语的混合策略在鲁棒性和token效率之间取得了很好的平衡
3. **自底向上的数据工程**：从零件提取→复杂度过滤→自动标注的完整流水线设计，使得利用大规模未标注的ABC数据集成为可能
4. **极低的无效率**：0.018的无效率说明LLM生成的结构化参数具有很强的几何一致性

## 局限与展望

1. **复杂提示**：对于复杂描述（如"有山墙屋顶的两层房子"），NURBGen难以捕捉精细结构
2. **几何伪影**：少数情况下出现自相交或拓扑不一致
3. **文字雕刻**：无法重建包含雕刻文字的提示
4. **上下文窗口限制**：当前训练限制在8192 tokens，未来可探索长上下文训练以处理更复杂的装配体
5. 仅处理了ABC数据集的20万个模型（共100万），未来可扩展

## 相关工作与启发

- **与NeuroNURBS的区别**：NeuroNURBS用非自回归transformer VAE学习untrimmed NURBS的潜在编码，但不支持语言条件生成且无法处理修剪问题
- **与LLaMA-Mesh的对比**：LLaMA-Mesh微调LLaMA生成mesh顶点和面的纯文本，而NURBGen生成的是结构化、可编辑的NURBS参数
- **启发**：结构化符号表示（vs 潜在编码）在LLM驱动的3D生成中可能是更有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个NURBS-based text-to-CAD框架，混合表示设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 多指标评估+人类评估，但消融研究较少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术细节完整
- 价值: ⭐⭐⭐⭐⭐ — 开创了NURBS作为设计历史方法替代的新方向，partABC数据集有很大价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LATTICE: Democratize High-Fidelity 3D Generation at Scale](../../CVPR2026/3d_vision/lattice_democratize_high-fidelity_3d_generation_at_scale.md)
- [\[ICML 2026\] RelaxFlow: Text-Driven Amodal 3D Generation](../../ICML2026/3d_vision/relaxflow_text-driven_amodal_3d_generation.md)
- [\[CVPR 2026\] CoSMo3D: Open-World Promptable 3D Semantic Segmentation through LLM-Guided Canonical Spatial Modeling](../../CVPR2026/3d_vision/cosmo3d_open-world_promptable_3d_semantic_segmentation_through_llm-guided_canoni.md)
- [\[CVPR 2026\] Text-Driven 3D Hand Motion Generation from Sign Language Data](../../CVPR2026/3d_vision/text-driven_3d_hand_motion_generation_from_sign_language_data.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)

</div>

<!-- RELATED:END -->
