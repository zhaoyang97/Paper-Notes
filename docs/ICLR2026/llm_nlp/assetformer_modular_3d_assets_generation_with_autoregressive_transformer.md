---
title: "AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer"
authors: "Junhao Chen, Xiang Li, Jian Yang, et al."
affiliations: "HKU, LIGHTSPEED Studios"
venue: "ICLR 2026"
arxiv: "2602.12100"
code: "https://github.com/Advocate99/AssetFormer"
tags: ["3D generation", "autoregressive transformer", "modular assets", "UGC", "Llama", "text-to-3D"]
rating:
  novelty: 4
  experiments: 3
  writing: 4
  value: 4
---

# AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer

## 一句话总结

提出 AssetFormer，基于 Llama 架构的自回归 Transformer，将模块化 3D 资产（由 primitive 序列组成）建模为离散 token 序列，通过 DFS/BFS 图遍历重排序和联合词汇表解码实现从文本描述生成可直接用于游戏引擎的模块化 3D 资产。

## 研究动机

3D 资产生成是 UGC（用户生成内容）和游戏行业的核心需求之一。当前主流的 3D 生成方法（如基于 NeRF、3D Gaussian Splatting 或网格直接生成的方法）虽然在视觉质量上取得了长足进步，但存在几个关键问题：

1. **与现有工作流程不兼容**：生成出的 3D 内容往往是整体性的网格或隐式表示，难以直接导入游戏引擎进行编辑、拆分或重组，用户需要大量后处理工作。
2. **缺乏模块化结构**：真实游戏开发中，3D 场景和物体通常由标准化的模块化组件（如积木块）组合而成，现有方法难以生成这种结构化的资产。
3. **离散属性建模困难**：模块化资产中的每个 primitive 具有类别（class）、旋转（rotation）、位置（position）等混合属性，这些属性是离散且有结构约束的，传统的连续生成方法不擅长处理。

AssetFormer 的出发点是：**既然模块化 3D 资产本质上是一组带有离散属性的 primitive 序列，那为什么不用自然语言领域成熟的自回归 Transformer 来直接建模呢？**

## 核心问题

如何将模块化 3D 资产的生成问题转化为一个序列到序列的建模问题，使得自回归 Transformer 能够有效地从文本描述生成由离散 primitive 组成的合法 3D 资产？

## 方法详解

### 模块化 3D 资产表示

模块化 3D 资产由一系列 **primitive** 组成，每个 primitive 具有三个属性：
- **类别 (class)**：从一个预定义的 primitive 库中选取（如不同形状的积木块），共 $K$ 种类型
- **旋转 (rotation)**：离散化为有限组旋转角度，每种 primitive 根据其对称性有不同数量的有效旋转
- **位置 (position)**：在 3D 网格上的坐标 $(x, y, z)$，离散化为整数坐标

一个完整的 3D 资产可以表示为：
$$
A = \{(c_i, r_i, p_i)\}_{i=1}^{N}
$$
其中 $N$ 是 primitive 数量，$c_i, r_i, p_i$ 分别为第 $i$ 个 primitive 的类别、旋转和位置。

### 无损离散 Tokenization

与许多需要训练 VQ-VAE codebook 的方法不同，AssetFormer 采用**无损离散化 (lossless tokenization)**：

- 每个 primitive 的属性直接映射为离散 token，无需额外的编码-解码器
- 类别、旋转、位置分别占用固定数量的 token
- 这保证了 token 化过程的无损性——解码出的 token 可以无歧义地还原为精确的 3D 资产

这种设计避免了 codebook 训练带来的信息损失和训练不稳定性问题。

### Token 重排序策略

将 3D 资产展开为 1D token 序列时，primitive 的排列顺序对自回归建模的效果至关重要。作者观察到 primitive 之间存在空间邻接关系，可以构建一个**邻接图 (adjacency graph)**：

- 节点：每个 primitive
- 边：空间上相邻（共享面或边）的 primitive 对

在此图上，作者对比了两种遍历策略：
- **BFS（广度优先搜索）**：按层次展开，生成的序列中相邻 token 倾向于在同一"层"上
- **DFS（深度优先搜索）**：沿着一条路径深入，生成的序列中相邻 token 倾向于在同一"分支"上

实验表明 **DFS 排序略优于 BFS**，原因是 DFS 产生的序列中相邻 token 之间的空间关系更加局部化，有利于自回归模型捕捉短程依赖。此外，DFS 产生的 token 序列在属性值（尤其是坐标）上的变化更平滑，有利于训练时的归一化（normalization）。

### Token Set Modeling

一个关键挑战是：每个 primitive 的三种属性（class, rotation, position）各自有不同的合法取值范围。简单地将所有可能的 token 放入同一个词汇表会导致大量非法组合。

AssetFormer 提出 **Token Set Modeling**：
- 构建一个**联合词汇表 (joint vocabulary)**，包含所有属性类型的所有可能取值
- 在解码时使用 **filtered decoding（过滤解码）**：根据当前应该生成的属性类型，动态屏蔽不属于该类型的 token，确保每一步只能生成合法的属性值
- 这实际上是一种**结构化约束解码**，在不改变模型架构的前提下保证了输出的合法性

### 模型架构

AssetFormer 基于 **Llama 架构**：
- 使用标准的 decoder-only Transformer
- 文本 prompt 通过预训练文本编码器编码后作为前缀条件
- 3D 资产的 token 序列作为目标进行自回归生成
- 采用 RoPE 位置编码
- 模型规模未追求极大，证明在适当规模下即可有效工作

### Classifier-Free Guidance (CFG)

借鉴扩散模型中 Classifier-Free Guidance 的成功经验，AssetFormer 将 CFG 适配到自回归生成框架中：

- 训练时以一定概率随机丢弃文本条件（替换为空 prompt）
- 推理时同时计算有条件和无条件的 logits，通过线性外推增强文本引导效果：
$$
\text{logits}_{\text{guided}} = \text{logits}_{\text{uncond}} + \lambda \cdot (\text{logits}_{\text{cond}} - \text{logits}_{\text{uncond}})
$$
其中 $\lambda > 1$ 为引导强度。这有效提升了生成结果与文本描述的一致性。

### 数据集构建

数据是本文的一个重要贡献：
- **真实用户数据**：从在线 UGC 平台（类似 Roblox 的游戏创作平台）收集真实用户创建的模块化 3D 资产
- **PCG 程序化数据**：使用程序化内容生成（Procedural Content Generation）扩充数据集
- **文本标注**：将 3D 资产从多个视角渲染为 2D 图像，然后使用 **GPT-4o** 生成对应的文本描述
- 这种半自动的数据构建流程有效解决了模块化 3D 资产缺乏文本标注的问题

## 实验关键数据

### 生成质量

- AssetFormer 生成的模块化资产在视觉质量和文本一致性上明显优于简单的基线方法
- 与端到端 3D 网格生成方法相比，AssetFormer 的优势在于生成结果**天然可编辑、可在游戏引擎中直接使用**
- 定量指标包括 FID、CLIP Score 等

### DFS vs BFS

- DFS 排序在多个指标上略优于 BFS
- 作者分析原因是 DFS 序列的坐标变化更加连续，有助于模型学习空间位置的局部模式
- 这一发现对模块化资产这类具有空间拓扑结构的序列建模任务有参考意义

### CFG 效果

- CFG 的引入显著提升了文本-3D 一致性
- 引导强度 $\lambda$ 存在一个最优区间，过大会导致多样性下降

### 与其他 3D 生成方法的对比

- 对比了基于扩散模型和其他自回归方法的 3D 生成器
- AssetFormer 在模块化资产这一特定任务上具有明显的结构优势
- 其他方法难以直接生成可编辑的模块化结构

## 亮点

1. **问题形式化巧妙**：将模块化 3D 资产生成转化为离散 token 序列建模，充分利用了大语言模型技术栈的成熟度
2. **无损 tokenization**：避免了 VQ-VAE 等方法的信息压缩损失，token 与原始属性一一对应
3. **图遍历重排序**：利用 primitive 之间的空间拓扑关系构建邻接图，通过 DFS/BFS 遍历找到更适合自回归建模的序列顺序，是一个简洁有效的设计
4. **Token Set Modeling + Filtered Decoding**：联合词汇表 + 动态过滤解码，优雅地解决了多属性类型的合法性约束问题
5. **实用导向**：生成结果可直接导入游戏引擎，对 UGC 平台和游戏开发有真实应用价值
6. **数据构建流程可复制**：GPT-4o 标注 + PCG 扩充的数据管线具有推广价值

## 局限性 / 可改进方向

1. **Primitive 库固定**：当前方法依赖预定义的 primitive 集合，无法生成全新类型的基本组件，限制了生成多样性
2. **纹理和材质**：论文主要关注几何结构的生成，未涉及纹理、材质、光照等视觉属性
3. **规模限制**：由于 token 序列长度与 primitive 数量成正比，超大规模的 3D 场景可能导致序列过长
4. **评估指标有限**：3D 模块化资产的评估标准尚不成熟，论文中的定量评估可能不能完全反映实际使用体验
5. **数据来源单一**：训练数据主要来自特定的 UGC 平台，可能存在风格偏差
6. **DFS vs BFS 差异较小**：虽然 DFS 略优，但两者差距不大，说明序列顺序可能不是最关键的瓶颈

## 思考与启发

- **"万物皆序列"的又一例证**：继代码生成、分子生成之后，3D 模块化资产也被成功地序列化并用自回归模型处理，表明自回归 Transformer 的范式在结构化离散数据上具有广泛的适用性
- **领域知识的融入方式**：通过图遍历排序和 filtered decoding，将 3D 资产的结构先验（空间邻接、属性类型约束）优雅地融入到通用的 Transformer 框架中，而非设计专用架构
- **与 CAD 生成的联系**：本文的方法思路与 CAD 模型生成（如 DeepCAD）有相通之处，都是用序列模型建模离散的几何操作序列；不同之处在于本文的 primitive 更加标准化
- **Classifier-Free Guidance 的通用性**：CFG 从图像扩散模型到 3D 自回归生成的成功迁移，说明这项技术在条件生成任务中具有范式级的通用价值
- **游戏行业的实际需求**：来自 LIGHTSPEED 工作室的合作背景使得这篇论文更具工程落地潜力，也表明工业界对 AI 辅助内容创建的真实需求
