# PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data

**会议**: ICLR 2026
**arXiv**: [2509.21965](https://arxiv.org/abs/2509.21965)
**代码**: https://czvvd.github.io/PartSAMPage/
**领域**: 3D 视觉
**关键词**: 3D part segmentation, SAM, prompt-based, native 3D data, open-world

## 一句话总结
提出首个在大规模原生 3D 数据上训练的可提示部件分割模型 PartSAM，采用 triplane 双分支编码器（冻结 SAM 先验 + 可学习 3D 分支）和 SAM 风格解码器，通过模型在环标注流程构建 500 万+形状-部件对，在开放世界设置下单次点击即超越 Point-SAM 90%+。

## 研究背景与动机

1. **领域现状**：3D 部件分割是 CV 经典问题。早期方法在 ShapeNet-Part/PartNet 等闭集数据上训练，无法泛化到开放世界。近期方法（SAMPart3D、PartField）利用 SAM 的 2D 先验做多视图 lift。
2. **现有痛点**：(1) 2D→3D lift 丢失内部结构信息，只能理解表面；(2) 聚类方法（PartField）缺乏交互可控性；(3) 训练数据瓶颈——缺乏大规模 3D 部件标注；(4) 对网格连接高度依赖，AI 生成形状上性能崩溃。
3. **核心矛盾**：如何在缺乏大规模 3D 部件标注的情况下，训练既能灵活交互、又能理解 3D 内部结构的模型？
4. **本文要解决什么？** 构建大规模原生 3D 部件数据（500万+对），设计同时利用 2D 先验和 3D 知识的新架构，实现 SAM 风格的交互+自动分割。
5. **切入角度**：双通道设计——冻结的 SAM 通道保留 2D 知识，可学习通道适应原生 3D 标注。
6. **核心 idea 一句话**：用双分支 triplane 编码器 + SAM 解码器在百万级原生 3D 部件数据上训练，首次实现真正理解 3D 内部结构的可提示部件分割。

## 方法详解

### 整体框架
输入为点云 $P_{in} \in \mathbb{R}^{N \times 9}$（坐标+法向+RGB）和提示点 $P_{prompt}$，编码器提取 triplane 特征场并采样得 patch embedding $F_c$，解码器结合提示 embedding $F_p$ 通过双向 Transformer 生成分割 mask。支持交互模式（用户点击）和自动模式（Segment Every Part）。

### 关键设计

1. **双分支 Triplane 编码器**:
   - 做什么：同时保留 SAM 的 2D 先验和学习原生 3D 表示
   - 核心思路：两个并行分支，均用 PVCNN+Transformer 构建 triplane 特征场。冻结分支保留 PartField 预训练的 SAM 对比学习特征；可学习分支通过零卷积接受法向/RGB 额外输入，在原生 3D 标注上训练
   - 设计动机：冻结分支防止 2D 知识遗忘，可学习分支适应新的 3D 监督信号

2. **SAM 启发的提示引导解码器**:
   - 做什么：从提示和特征生成分割 mask
   - 核心思路：引入输出 token $T_{out}$ 和 IoU token $T_{iou}$，双向 CrossAttn 交互 $F_c' = \text{CrossAttn}(F_c \leftrightarrow [F_p; T_{out}; T_{iou}])$，单提示时并行解码 3 个候选 mask，IoU token 预测质量分数选最优
   - 设计动机：SAM 的并行解码和 IoU 预测处理部件边界歧义

3. **模型在环数据标注**:
   - 做什么：从碎片化 Objaverse 资产扩展训练数据
   - 核心思路：第一阶段从场景图/连通分量提取 180k 形状 2200 万部件；第二阶段用 PartField 生成伪标签（K=10/20/30 聚类），PartSAM 10 轮交互验证，IoU@1>60% 或 IoU@10>90% 则接受，最终 500k 形状 5500 万部件
   - 设计动机：高 IoU@1 代表内在明确的部件，低 IoU@1 但高 IoU@10 代表可交互完善的部件

### 损失函数
$\mathcal{L} = \mathcal{L}_{focal} + \alpha \mathcal{L}_{dice} + \mathcal{L}_{IoU} + \lambda \mathcal{L}_{triplet}$

## 实验关键数据

### 主实验（交互分割）

| 数据集 | 方法 | IoU@1 | IoU@5 | IoU@10 |
|--------|------|-------|-------|--------|
| PartObjaverse-Tiny | Point-SAM | 29.4 | 68.7 | 73.9 |
| | **PartSAM** | **56.1** | **84.1** | **87.6** |
| PartNetE | Point-SAM | 35.9 | 75.1 | 79.2 |
| | **PartSAM** | **59.5** | **86.5** | **89.9** |

### 自动分割

| 方法 | PartObjaverse-Tiny | PartNetE |
|------|-------------------|----------|
| PartField | 51.5 | 59.1 |
| **PartSAM** | **69.5** | **72.4** |

### 关键发现
- **IoU@1 提升 91%**：单次点击即准确分割，Point-SAM 高度依赖迭代补正
- **内部结构理解**：能分割被遮挡的手袋内物品、汽车内座椅等，SAMesh 无法做到
- **AI 生成形状泛化**：在 Hunyuan3D 生成的不规则网格上保持良好性能

## 亮点与洞察
- **范式转变**：从"2D lift → 聚类"到"原生 3D + 交互解码"，首次真正理解 3D 内部结构
- **模型在环标注解决了鸡生蛋问题**：通过 PartField 伪标签 + PartSAM 交互验证的双层筛选
- **双分支设计精妙**：冻结保留 2D 边界先验，可学习适应粗糙 3D 信号，相加融合简洁有效

## 局限性 / 可改进方向
- 交互推理成本较聚类方法更高
- 500k 形状相比 2D 基础模型的十亿级数据仍有差距
- 极小部件（按钮级别）因点采样分辨率限制性能下降

## 相关工作与启发
- **vs PartField**: 聚类后处理是 bottleneck，prompted 解码绕过了这个问题，且不依赖网格连接
- **vs Point-SAM**: 数据和架构都更大（5x 数据、双分支编码），性能跃升
- 启示：3D foundation model 需要真正大规模的原生 3D 数据

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个原生 3D 训练的可提示部件分割模型
- 实验充分度: ⭐⭐⭐⭐ 多基准全面评估，但消融细节在附录
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述系统，可视化出色
- 价值: ⭐⭐⭐⭐⭐ 开启 3D 版 SAM 的新方向
