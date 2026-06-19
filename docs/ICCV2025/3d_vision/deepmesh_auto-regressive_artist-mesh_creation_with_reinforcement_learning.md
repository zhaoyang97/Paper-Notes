---
title: >-
  [论文解读] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning
description: >-
  [ICCV 2025][3D视觉][3D网格生成] 提出 DeepMesh 框架，通过改进的高效mesh tokenization算法（72%压缩率）和首次将DPO强化学习引入3D网格生成来实现人类偏好对齐，能够生成最高3万面的高质量Artist-like三角网格。 Artist-like Mesh的重要性：- 三角网格是3…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D网格生成"
  - "自回归模型"
  - "强化学习"
  - "DPO"
  - "网格tokenization"
  - "点云条件生成"
---

# DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning

**会议**: ICCV 2025  
**代码**: [https://zhaorw02.github.io/DeepMesh/](https://zhaorw02.github.io/DeepMesh/)  
**领域**: 3D视觉  
**关键词**: 3D网格生成, 自回归模型, 强化学习, DPO, 网格tokenization, 点云条件生成

## 一句话总结

提出 DeepMesh 框架，通过改进的高效mesh tokenization算法（72%压缩率）和首次将DPO强化学习引入3D网格生成来实现人类偏好对齐，能够生成最高3万面的高质量Artist-like三角网格。

## 研究背景与动机

**Artist-like Mesh的重要性：**

- 三角网格是3D资产的基础表示，广泛应用于VR、游戏和动画
- 艺术家手工创建的网格具有优化的拓扑结构，便于编辑、变形和纹理映射
- Marching Cubes等自动化方法生成的网格虽然几何精度高，但拓扑结构不规则、过于密集

**自回归网格生成的两大挑战：**

**预训练效率问题**：现有的mesh tokenization方法产生过长的序列（增加计算成本），且低质量mesh导致训练不稳定（loss spike）

**缺乏人类偏好对齐**：现有方法无法保证生成结果符合人类的审美标准，且常出现几何缺陷（空洞、缺失部分、冗余结构）

**BPT的局限**：虽然达到约74%的压缩率，但仅在低分辨率（128）下有效工作，更高分辨率下词表大小剧增（40960），导致训练困难。

## 方法详解

### 整体框架

DeepMesh = 改进的tokenization + 高效预训练策略 + DPO后训练。核心模型是自回归transformer，包含自注意力和交叉注意力层。

### 1. 改进的Tokenization算法

基于BPT进行改进，保持约72%的压缩率同时大幅缩小词表：

**核心步骤：**
1. **局部面遍历（Local Face Traversal）**：按连通性将mesh面划分为局部patch，最小化冗余
2. **排序和量化**：对每个面的顶点坐标排序、量化，按XYZ顺序展平
3. **三级层次化块索引**：将坐标系分为三个层次的块，用偏移量索引量化坐标
4. **相同索引合并**：相邻顶点往往共享相同的偏移索引，合并以进一步压缩

**关键优势**：在512分辨率下，压缩率0.28、词表大小4736（vs BPT的0.26/40960），训练效率大幅提升。

### 2. 预训练策略

**数据筛选（Data Curation）**：基于几何结构和视觉质量过滤低质量mesh，有效缓解训练中的loss spike。

**截断训练（Truncated Training）**：将token序列分割为固定大小的上下文窗口，采用滑动窗口机制逐步训练。

**数据打包（Data Packaging）**：按面数分类mesh，分配面数相近的mesh到同一batch，确保更好的负载均衡。

**模型架构**：基于Hourglass Transformer，节省50%显存；使用基于Michelangelo的perceiver encoder处理点云条件。模型规模从5亿到10亿参数。

### 3. DPO后训练——首次在3D Mesh生成中应用

**评分标准（Score Standard）：**
- **几何完整性**：使用Chamfer Distance衡量生成mesh与GT之间的相似度
- **视觉美观度**：招募志愿者进行主观比较，捕捉传统指标无法衡量的审美判断

**偏好对构建（Preference Pair Construction）：**
1. 对每个输入点云，模型生成两个不同的mesh
2. 首先用Chamfer Distance筛选：如果两个mesh的CD都较差则丢弃
3. 如果一个优于另一个，直接选择；如果两者都合格，由志愿者判断审美偏好
4. 共收集5000个偏好对

**DPO损失函数：**

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(c,y^+,y^-) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+|c)}{\pi_{\text{ref}}(y^+|c)} - \beta \log \frac{\pi_\theta(y^-|c)}{\pi_{\text{ref}}(y^-|c)} \right) \right]$$

同样采用截断训练策略处理DPO中的长token序列。

## 实验关键数据

### 主实验：点云条件生成的定量比较

| 指标 | MeshAnythingv2 | BPT | Ours (w/o DPO) | **Ours (w DPO)** |
|------|---------------|-----|----------------|------------------|
| C.Dist. ↓ | 0.1249 | 0.1425 | 0.1001 | **0.0884** |
| H.Dist. ↓ | 0.2991 | 0.2796 | 0.1861 | **0.1708** |
| User Study ↑ | 10% | 19% | 34% | **37%** |

- DeepMesh在几何精度上大幅超越基线（Chamfer Distance降低29.2% vs MeshAnythingv2）
- DPO后训练进一步提升了几何质量和用户偏好（CD从0.1001→0.0884）
- 用户研究中37%的志愿者偏好DeepMesh的结果

### 消融实验：Tokenization算法比较

| 指标 | AMT | EdgeRunner | BPT | **Ours** |
|------|-----|-----------|-----|----------|
| 压缩率 ↓ | 0.46 | 0.47 | 0.26 | **0.28** |
| 词表大小 ↓ | 512 | 512 | 40960 | **4736** |
| 时间 (s) ↓ | 816 | - | 540 | **480** |

- 在512分辨率下，DeepMesh的tokenization在压缩率和词表大小之间取得**最佳平衡**
- 词表仅为BPT的11.6%，训练效率最高
- 小词表使得模型更容易学习，训练更稳定

### DPO后训练效果

定性分析显示：
- DPO前后的模型都能生成良好的几何结构
- DPO后的结果在视觉上更美观，线框更规则，表面细节更丰富
- 量化指标也确认DPO提升了与GT的相似度

### 关键发现

1. 模型可生成高达30k面的高精度mesh，远超基线方法
2. 5000个偏好对就足以通过DPO显著提升生成质量
3. 数据筛选策略有效缓解了训练中的loss spike问题
4. 截断训练 + 数据打包策略使得训练效率提升明显

## 亮点与洞察

1. **首次将RLHF/DPO引入3D Mesh生成**：从LLM领域的成功经验迁移到3D生成，展示了跨模态的方法论价值
2. **Tokenization算法的工程创新**：在保持高压缩率的同时将词表从40960缩小到4736，这是使高分辨率mesh生成变得可行的关键
3. **评分标准的双重设计**：结合客观度量（Chamfer Distance）和主观评价（人类偏好），比单纯使用任一维度都更全面
4. **系统性的预训练优化**：数据筛选、打包和截断训练共同构成了稳健的训练pipeline
5. **多样性能力**：相同点云输入可生成多种不同外观的mesh，对设计应用非常有价值

## 局限性

1. **生成速度**：自回归生成30k面的mesh需要预测大量token，推理时间较长
2. **仅支持点云条件**：图像条件生成需要先转为点云（通过TRELLIS），是间接方案
3. **DPO数据规模有限**：仅5000对偏好数据，更大规模的对齐数据可能带来更好效果
4. **纹理缺失**：目前仅生成几何网格，不包含纹理信息

## 相关工作与启发

- **MeshGPT → MeshAnything → BPT → DeepMesh**：自回归mesh生成的快速演进
- **DPO在不同模态的应用**：LLM → VLLM → 3D Mesh，模态迁移的成功案例
- **启发**：RLHF/DPO在其他3D生成任务中的潜力（如纹理生成、场景生成）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — DPO在3D Mesh领域的首次应用，tokenization改进实用
- **实验**: ⭐⭐⭐⭐ — 定量定性全面，包含用户研究和多维度消融
- **写作**: ⭐⭐⭐⭐ — 结构清晰，但部分拼写错误（如"poineer"）
- **价值**: ⭐⭐⭐⭐⭐ — 推动了高质量Artist Mesh自动生成的前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BuildingGPT: Auto-Regressive Building Wireframe Reconstruction Model with Reinforcement Learning](../../CVPR2026/3d_vision/buildinggpt_auto-regressive_building_wireframe_reconstruction_model_with_reinfor.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](auto-regressively_generating_multi-view_consistent_images.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)

</div>

<!-- RELATED:END -->
