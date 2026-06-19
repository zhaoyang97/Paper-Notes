---
title: >-
  [论文解读] PhysX-3D: Physical-Grounded 3D Asset Generation
description: >-
  [NeurIPS 2025 Spotlight][3D视觉][物理属性3D生成] PhysX提出了首个端到端的物理属性驱动3D资产生成范式，包括PhysXNet（首个系统标注了绝对尺度、材料、功能可供性、运动学和功能描述五个维度的物理3D数据集，含26K+对象）和PhysXGen（双分支前馈生成框架，将物理知识注入预训练的3D结构空间中）。
tags:
  - "NeurIPS 2025 Spotlight"
  - "3D视觉"
  - "物理属性3D生成"
  - "3D数据集"
  - "物理标注"
  - "铰接体建模"
  - "具身AI"
---

# PhysX-3D: Physical-Grounded 3D Asset Generation

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2507.12465](https://arxiv.org/abs/2507.12465)  
**代码**: [项目主页](https://physx-3d.github.io/)  
**领域**: 3D视觉  
**关键词**: 物理属性3D生成, 3D数据集, 物理标注, 铰接体建模, 具身AI

## 一句话总结

PhysX提出了首个端到端的物理属性驱动3D资产生成范式，包括PhysXNet（首个系统标注了绝对尺度、材料、功能可供性、运动学和功能描述五个维度的物理3D数据集，含26K+对象）和PhysXGen（双分支前馈生成框架，将物理知识注入预训练的3D结构空间中）。

## 研究背景与动机

近年来3D资产生成取得了巨大进展，但现有方法几乎**完全聚焦于几何和纹理，忽视了物理属性**。然而真实世界的物体天然具有丰富的物理和语义特征：

**绝对尺度**：物体的真实物理尺寸

**材料**：材料名称、杨氏模量、泊松比、密度

**功能可供性（Affordance）**：各部件被触摸/抓取的优先级

**运动学**：关节类型、运动范围、运动方向、父子部件关系

**功能描述**：基础/功能/运动描述文本

现有数据集的覆盖非常碎片化（见表1对比）：
- PartNet-Mobility仅2.7K对象且只有运动学标注
- ABO有材料和尺寸但只有对象级标注（非部件级）
- Objaverse规模大但无任何物理标注

这种物理属性的缺失严重阻碍了3D资产在仿真、机器人和具身AI中的实际应用。PhysX的核心动机是**从上游数据标注到下游生成模型，建立完整的物理化3D资产流水线**。

## 方法详解

### 整体框架

PhysX包含两个核心组件：
- **PhysXNet数据集**：26K+物理3D对象 + PhysXNet-XL的600万程序化扩展
- **PhysXGen生成模型**：双分支前馈框架，基于预训练的TRELLIS 3D生成模型

### 关键设计

1. **Human-in-the-Loop标注流水线**：分为两个阶段：

    - **初步数据获取**：首先对每个部件进行Alpha合成渲染（目标部件红色，其他灰色），以获得最佳视觉提示并最小化遮挡干扰。然后使用GPT-4o进行自动标注，获取基础物理属性（材料、密度、部件名称、功能可供性、功能描述）。人工候选者检查并修正VLM输出。
    - **运动学参数确定**：对所有有约束运动的部件（非自由或刚性连接），计算子-父网格对的接触区域，进行平面拟合以获取运动轴候选方向，生成候选位置（对旋转关节额外用K-means确定旋转轴位置），最后由人工选择最佳候选并确定运动学参数。
   
   在预处理中，对PartNet中过于细粒度的部件进行合并（面积≤0.2或面数≤100且面积≤0.06），并人工验证合并结果。运动学类型包括五种：A（自由）、B（棱柱/平移关节）、C（旋转关节）、D（铰关节）、E（刚性连接），以及组合类型CB。

2. **PhysXGen双分支生成架构**：包含两阶段：

    - **Physical 3D VAE**：构建物理VAE编码器 $\mathcal{E}_{phy}$ 和解码器 $\mathcal{D}_{phy}$，将物理属性（尺度 $P_{dim} \in \mathbb{R}^{N \times 1}$、可供性 $P_{aff} \in \mathbb{R}^{N \times 1}$、密度 $P_\rho \in \mathbb{R}^{N \times 1}$、运动学参数 $P_{mov} \in \mathbb{R}^{N \times 11}$）拼接为 $P_{phy} \in \mathbb{R}^{N \times 14}$，连同CLIP编码的功能描述 $P_{sem} \in \mathbb{R}^{N \times 768 \times 3}$，编码到物理潜空间 $P_{plat} \in \mathbb{R}^{N \times 8}$。结构分支使用预训练的DINOv2特征编码。关键是通过残差连接建立 $\mathcal{D}_{phy}$ 到 $\mathcal{D}_{aes}$ 的信息通路，利用物理-结构的相关性。
   
    - **物理潜在生成**：基于Transformer的扩散模型，使用条件流匹配（CFM）目标。物理分支有14个Transformer块（少于结构分支的24块以降低计算量）。结构分支通过可学习的skip-connection层向物理分支提供指导。损失为 $\mathcal{L}_{diff} = \mathcal{L}_{aes} + \mathcal{L}_{phy}$。

3. **PhysXNet-XL程序化扩展**：通过类内和跨类组合规则，从PhysXNet出发程序化生成600万+物理3D对象。类内组合覆盖柜子、桌子、瓶子等9类；跨类组合识别抽屉和门作为可灵活集成的模块化组件。确保结构和物理一致性。

### 损失函数 / 训练策略

**VAE损失：**

$$\mathcal{L}_{vae} = \mathcal{L}_{aes}^{color} + \mathcal{L}_{aes}^{geometry} + \mathcal{L}_{phy} + \mathcal{L}_{sem} + \mathcal{L}_{kl} + \mathcal{L}_{reg}$$

- $\mathcal{L}_{aes}^{color}$: L2 + LPIPS
- $\mathcal{L}_{aes}^{geometry}$: mask + normal + depth
- $\mathcal{L}_{phy}$/$\mathcal{L}_{sem}$: 归一化L2
- AdamW优化器，lr=1e-4，8×A100 GPU
- 24K训练 / 1K验证 / 1K测试

## 实验关键数据

### 主实验

**PhysXGen vs 基线方法：**

| 方法 | PSNR↑ | CD↓ | F-Score↑ | 绝对尺度↓ | 材料↑ | 可供性↑ | 运动COV↑ | 运动MMD↓ | 描述↑ |
|------|-------|-----|----------|---------|------|--------|---------|---------|------|
| TRELLIS | 24.31 | 13.2 | 76.9 | – | – | – | – | – | – |
| TRELLIS+PhysPre | 24.31 | 13.2 | 76.9 | 13.21 | 8.63 | 7.23 | 0.24 | 0.12 | 6.55 |
| **PhysXGen** | **24.53** | **12.7** | **77.3** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | **10.11** |

**与GPT基线对比（TRELLIS+PartField+GPT-4o）：**

| 方法 | 绝对尺度↓ | 材料↑ | 可供性↑ | 运动COV↑ | 运动MMD↓ | 描述↑ |
|------|---------|------|--------|---------|---------|------|
| TRELLIS+PartField+GPT | 8.81 | 7.95 | 6.73 | 0.09 | 0.24 | **14.31** |
| **PhysXGen** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | 10.11 |

PhysXGen在绝对尺度、材料、运动学和可供性上分别提升24%、64%、28%和72%。

### 消融实验

**双分支架构的效果：**

| Dep-VAE | Dep-Diff | PSNR↑ | 绝对尺度↓ | 材料↑ | 可供性↑ | 运动COV↑ | 描述↑ |
|---------|---------|-------|---------|------|--------|---------|------|
| ✗ | ✗ | 24.31 | 13.21 | 8.63 | 7.23 | 0.24 | 6.55 |
| ✗ | ✓ | 24.31 | 12.01 | 10.69 | 8.95 | 0.26 | 7.71 |
| ✓ | ✗ | 24.32 | 10.57 | 9.86 | 9.32 | 0.28 | 7.54 |
| ✓ | ✓ | **24.53** | **7.24** | **13.01** | **11.30** | **0.33** | **10.11** |

### 关键发现

- **物理-结构相关性是核心**：联合建模两者后，不仅物理属性大幅提升，几何质量也有改善（CD从13.2降到12.7）
- VAE和扩散模型中的双分支设计都不可或缺，缺少任一均会显著降低物理属性生成质量
- GPT-4o在功能描述上更擅长（得益于语言能力），但在结构化物理属性上远不如端到端学习
- 绝对尺度预测面临长尾分布挑战（1-1000cm），线性和对数归一化都不理想
- 运动学是最有挑战性的属性，因为需要同时准确预测离散的部件层级关系和连续的运动参数

## 亮点与洞察

- **填补重要空白**：首次系统定义并标注了3D物体的完整物理属性谱，对具身AI社区价值巨大
- **标注流水线的可扩展设计**：GPT-4o + 人工校验的组合既高效又可靠，且可复用到新数据集
- **物理-结构相关性的利用**很聪明：物理属性（如材料→密度→运动特性）本身就与几何形状相关，联合建模是自然且有效的
- PhysXNet-XL的600万级程序化扩展为大规模物理3D数据提供了可行路径

## 局限与展望

- 物理属性生成可能产生空间不一致的伪影（如相邻区域材料/可供性不连续）
- 运动学参数的回归预测难以准确确定部件数量和父子层级关系
- 数据集受限于PartNet的室内/CAD模型分布，缺少户外和真实扫描数据
- 功能描述依赖CLIP的编码器，其不可逆性限制了embedding到文本的解码能力
- 仅使用了4种物理属性用于生成，未包含摩擦系数等更细粒度的物理量

## 相关工作与启发

- **TRELLIS**是结构化3D生成的基础，PhysXGen通过在其潜空间上叠加物理分支来扩展
- 与PartNet-Mobility相比，PhysXNet在标注维度和规模上都有质的飞跃
- 标注流水线中的"部件级视觉隔离+VLM标注"策略可推广到其他需要细粒度标注的场景
- 对机器人操作和物理仿真的下游应用有直接价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次定义物理属性3D生成问题，数据集贡献突出
- **实验充分度**: ⭐⭐⭐⭐ 与多种基线对比，消融验证了核心设计，但测试集仅限于室内CAD模型
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，标注流水线描述详细
- **价值**: ⭐⭐⭐⭐⭐ 数据集对具身AI和机器人社区有巨大推动作用，方向意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image](../../CVPR2026/3d_vision/physx-anything_simulation-ready_physical_3d_assets_from_single_image.md)
- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](../../CVPR2025/3d_vision/kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](../../ICCV2025/3d_vision/gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[ICCV 2025\] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](../../ICCV2025/3d_vision/dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)
- [\[NeurIPS 2025\] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)

</div>

<!-- RELATED:END -->
