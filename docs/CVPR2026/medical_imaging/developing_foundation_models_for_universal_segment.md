# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**会议**: CVPR 2026  
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)  
**代码**: [GitHub](https://github.com/YichiZhang98/SegAnyPET)  
**领域**: 医学图像分割 / 基础模型 / PET 成像  
**关键词**: PET segmentation, foundation model, promptable segmentation, 3D architecture, universal model  

## 一句话总结
构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11041 例扫描、59831 个掩模），并提出 SegAnyPET 基础模型，实现基于 prompt 的 3D 全身 PET 通用可交互分割，在多中心、多示踪剂、多疾病场景下展现强 zero-shot 泛化能力。

## 背景与动机
PET 成像可视化放射性示踪剂分布以量化代谢过程，在肿瘤学和神经学中不可替代。但用于 PET 定量分析的深度学习模型发展远落后于 CT/MRI：原因包括 PET 缺乏解剖对比度（低分辨率、部分容积效应）、数据采集和标注成本高、公开数据集少且局限于特定肿瘤任务。现有通用分割基础模型（SAM-Med3D、SegVol、SAT 等）主要在结构影像（CT/MRI）上训练，直接迁移到 PET 效果极差——特别是文本提示模型（如 SAT）在 PET 器官分割上 DSC 接近 0。

## 核心问题
如何为功能性 PET 影像构建一个真正通用的分割基础模型——能够灵活分割训练中见过的器官和未见过的病灶/器官，支持多示踪剂和多中心场景，并融入临床 human-in-the-loop 工作流？

## 方法详解
### 整体框架
PETWB-Seg11K 数据集构建 → SegAnyPET 3D 基础模型训练 → 多层次评估（in-distribution 器官/病灶 + out-of-distribution 跨中心/跨示踪剂/跨疾病）→ 下游临床应用验证（标注效率 + 代谢网络分析）。

### 关键设计
1. **PETWB-Seg11K 数据集**：整合 AutoPET-FDG（1014 例肿瘤）+ UDPET（1371 例多器官）+ PETS-A（5731 例）+ PETS-B（2925 例），涵盖多中心、多扫描仪、多疾病，附带内部验证集（886 例）和外部验证集（1551 例，包含 PSMA-PET 前列腺癌等全新示踪剂）。
2. **SegAnyPET 架构**：全 3D Transformer 架构（非 2D+adapter 的妥协方案），包含 Image Encoder（3D 特征提取 + 绝对位置编码）、Prompt Encoder（支持 sparse/dense prompt, 点和粗掩模）、Mask Decoder（3D Transformer blocks + 转置卷积上采样 + MLP 预测头），实现端到端体积分割。
3. **迭代评估循环**：模拟交互式分割流程——自动从当前预测与 GT 的差异中生成新 prompt（FN 区域采正点、FP 区域采负点），累积 prompt 反馈给模型，驱动分割逐步收敛到 GT。
4. **SegAnyPET-Lesion 变体**：在 SegAnyPET 基础上用病灶数据微调 200 epoch，增强对小的异质性肿瘤病灶的敏感度和边界精度。

### 损失函数 / 训练策略
- Dice Loss + Cross-Entropy（sigmoid 激活 + squared predictions）
- AdamW 优化器，Image Encoder lr=8e-5，Prompt/Mask Decoder lr=8e-6，weight decay=0.1
- 8 × A100 80GB，500 epoch，global batch size=96，patch size=128³
- MultiStepLR（120/180 epoch 各降 10×）
- 每次迭代随机采样 1-20 个 click，增强交互适应性
- 增强：随机轴向/冠状/矢状翻转 + 自适应裁剪/填充到 128³

## 实验关键数据
- **vs 任务特定模型**（nnUNet/STUNet/SwinUNETR/SegResNet）：SegAnyPET（5p）在多数器官上 DSC 可比或超过，如肝脏 0.9494 vs nnUNet 0.9379，且无需 per-task 重训练
- **vs 通用分割基础模型**（SAM-Med3D/SAT/VISTA3D/SegVol/nnIteractive）：所有现有模型在 PET 上表现极差——SAT（文本提示）器官分割 DSC≈0；SAM-Med3D 最好也仅 0.61；SegAnyPET（1p）达到 0.84
- **跨中心/跨示踪剂泛化**（外部验证）：UMD-PETCT 13 器官、UMD-PETMR 多器官、AutoPET-PSMA 前列腺癌，SegAnyPET 展现强 zero-shot 能力，如 PSMA-PET 前列腺癌 DSC=0.576（SegAnyPET-Lesion），远超其他基础模型
- **临床标注效率**：SegAnyPET 辅助交互式工作流比纯人工标注节省 82.4%/83.0% 时间（两位专家）
- **代谢网络分析**：SegAnyPET 输出具有高生物学保真度，可构建可靠的全身代谢协方差网络

### 消融实验要点
- 1-point → 3-point → 5-point prompt 下 DSC 持续提升（如肝脏 0.926→0.939→0.949），验证了交互式精炼的有效性
- 通用模型 vs 病灶特化模型（SegAnyPET-Lesion）：后者在淋巴瘤和肺癌分割上有明显提升

## 亮点
- 填补了 PET 影像分割基础模型的空白——此前所有"通用"医学分割基础模型实际上都对 PET 无效
- 数据规模创纪录：11K+ 全身 PET 扫描远超现有 PET 数据集
- 全 3D 架构设计直接利用体积空间信息，避免了 2D slice-by-slice 的碎片化问题
- 开源代码和模型，有利于社区推动 PET AI

## 局限性 / 可改进方向
- 点 prompt 对弥漫性多病灶（如淋巴瘤全身散布）效率低，需逐个病灶 prompt——未来可引入文本 prompt 或自动语义分割
- 训练数据以 FDG 示踪剂为主，PSMA 等其他示踪剂数据有限
- 病灶分割指标（DSC ~0.1-0.15 基线很低）仍有很大改进空间
- 需要更多前瞻性临床研究验证实际工作流集成效果

## 与相关工作的对比
- vs **SAM / MedSAM**（2D）：需要 N 倍 prompt 工作量（每个切片），推理时间增加 10-30 倍，且缺乏 3D 空间连续性
- vs **SAM-Med3D**（3D 点 prompt）：PET 上性能远逊于 SegAnyPET（如肝脏 0.702 vs 0.926），因训练数据为结构影像
- vs **SAT**（文本 prompt）：在 PET 上完全失效（DSC≈0），说明结构影像训练的文本-视觉对齐无法迁移到 PET 的代谢分布模式
- vs **nnUNet**（任务特定）：SegAnyPET 一个模型替代多个 nnUNet 任务特定模型，且能泛化到未见目标

## 启发与关联
- 不同模态需要专属基础模型——通用 ≠ 万能，PET 的功能性成像特征与结构影像存在根本性 domain gap
- "prompt 工程"在 3D 医学影像中的设计原则值得系统研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 PET 专属分割基础模型，数据集和模型均有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 多中心、多示踪剂、多模型对比、下游应用验证，评估极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表丰富，数据展示规范
- 价值: ⭐⭐⭐⭐ 填补重要空白，开源推动社区发展
