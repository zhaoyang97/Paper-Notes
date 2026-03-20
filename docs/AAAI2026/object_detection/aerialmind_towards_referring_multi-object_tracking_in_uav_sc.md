# AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios

**会议**: AAAI 2026  
**arXiv**: [2511.21053v2](https://arxiv.org/abs/2511.21053v2)  
**代码**: [有 (数据集)](https://github.com/shawnliang420/AerialMind)  
**领域**: 目标检测 / 多目标跟踪 / 视觉-语言  
**关键词**: RMOT, 无人机, 视觉-语言融合, 多目标跟踪, 基准数据集  

## 一句话总结
构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

## 背景与动机
RMOT 任务通过自然语言指令实现对视频中特定目标的检测与跟踪，是智能机器人系统的基础能力。然而，现有 RMOT 研究几乎全部局限于地面视角（如 Refer-KITTI、Refer-BDD），无法覆盖无人机航拍的广域监控需求。无人机凭借俯瞰视角和高机动性，在大范围监控和具身智能中越来越重要，但航拍场景带来了目标外观剧变、复杂空间关系、动态场景变化、语义表达多样性等独特挑战。现有 RMOT 数据集和方法无法直接适配这些挑战。

## 核心问题
1. **数据集缺失**：无人机场景下缺乏大规模 RMOT 基准，限制了航拍 VL 感知的研究
2. **标注成本高**：RMOT 需要同时标注时序轨迹和自然语言描述，传统人工标注费时费力
3. **视觉-语言融合效率低**：现有早期融合/晚期融合范式存在模态鸿沟或"语言信号稀释"问题
4. **小目标感知困难**：航拍场景中高分辨率特征图的有效感受野有限，小尺度目标容易被背景噪声淹没

## 方法详解

### 整体框架
HETrack 基于 Deformable DETR 架构，使用 ResNet50 做视觉骨干、RoBERTa 做语言编码器。关键创新在于编码器和解码器之间插入了两个新模块：
1. **Co-evolutionary Fusion Encoder (CFE)**：在编码阶段实现视觉和语言的双向协同进化
2. **Scale Adaptive Contextual Refinement (SACR)**：在编码器输出和解码器之间增强小目标感知
3. 解码器使用 Semantic Guidance Module 进行语义引导的查询增强

### 关键设计
**CFE（共进化融合编码器）**：
- 核心思想：视觉特征的结构化过程和语言信息的引导过程不应独立，而应深度交织、相互促进
- 堆叠 $N_e$ 个 block，每个 block 包含：
  - **双向融合层（BFL）**：通过多头注意力实现视觉→语言和语言→视觉的双向信息流动。视觉特征为语言概念提供具体锚点，语言概念为视觉特征的筛选与增强提供引导
  - **可变形编码层（DEL）**：对融合后的特征做高效的空间关系建模
- 最终用句子级全局特征 $\mathbf{T}_s$ 对编码输出做整体调制，赋予模型对整体指代意图的把控

**SACR（尺度自适应上下文精炼）**：
- 在最高分辨率特征图上用并行空洞卷积（dilation rate = {6, 12, 18}）捕获多尺度上下文，不损失空间分辨率
- 自适应通道重校准：GAP → 1D 卷积（核大小由通道维度自适应确定：$k = |\log_2(C) + b / \gamma|_{\text{odd}}$）→ Sigmoid → 通道加权，抑制背景噪声、强调小目标关键通道

**Semantic Guidance Module**：检测查询与词级特征做交叉注意力后拼接跟踪查询送入解码器

**COALA 标注框架**（数据集构建的创新）：
- 四阶段 Agent 协作：场景理解提示生成（SUP-Agent）→ 半自动目标标注（SOL-Agent，标注者只需两次点击定义时间边界）→ 一致性检查（CC-Agent，跨模态时空逻辑推理验证）→ 表达扩展（EE-Agent，语义等价的多样化改写）

### 损失函数 / 训练策略
- 总损失 = $\lambda_{cls}\mathcal{L}_{cls} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{giou}\mathcal{L}_{giou} + \lambda_{ref}\mathcal{L}_{ref}$
- $\mathcal{L}_{cls}$：focal loss，$\mathcal{L}_{L1}$：L1 回归损失，$\mathcal{L}_{giou}$：GIoU 损失
- 权重配置：$\lambda_{cls}=2, \lambda_{L1}=5, \lambda_{giou}=2, \lambda_{ref}=2$
- AdamW 优化器，初始学习率 $1\times10^{-4}$，第 40 epoch 衰减 10 倍，共 100 epoch
- 8×A100 GPU，batch size 1，300 个 object queries
- 推理阶段得分阈值 0.5，referring 匹配阈值 $\beta_{ref}=0.4$
- 模型 51.4M 可训练参数，单 RTX 4080 推理 15.6 FPS

## 实验关键数据

**AerialMind 数据集规模**：93 视频序列，24.6K 表达式，293.1K 实例，46.14M bbox 标注 — 远超 Refer-KITTI-V2（9.8K 表达式）

**In-domain（VisDrone 测试集）**：

| 方法 | HOTA | DetA | AssA | HOTA_S | HOTA_M |
|------|------|------|------|--------|--------|
| TransRMOT | 23.54 | 13.18 | 42.24 | 27.21 | 24.05 |
| TempRMOT | 26.24 | 13.06 | 53.22 | 28.14 | 23.77 |
| MGLT | 26.16 | 14.83 | 46.47 | 26.39 | 26.10 |
| **HETrack** | **31.46** | **21.57** | **46.23** | **34.37** | **31.12** |

**Cross-domain（UAVDT 测试集）**：HETrack HOTA 31.60、DetA 21.35、LocA 83.98 — 均为最佳

**Refer-KITTI-V2（地面场景）**：HOTA 35.40，与 HFF-Track（36.18）接近，验证方法的通用性

### 消融实验要点
- 去掉 CFE+SACR：HOTA 从 31.46→26.41（-5.05），说明两个模块共同贡献巨大
- 仅去 CFE：HOTA 28.27（-3.19），CFE 贡献更大，视觉-语言协同融合是核心
- 仅去 SACR：HOTA 29.89（-1.57），SACR 对小目标检测有效但贡献相对较小
- 融合方式对比：CFE 的双向融合优于 Concat（28.88）、Add（30.39）、Cross-Attn（30.52）
- SACR 内部消融：仅空洞卷积 29.70，仅通道重校准 29.13，两者协同才能达到 31.46
- 引用阈值 $\beta_{ref}=0.4$ 为最优，过高过低均降性能
- 属性级分析：HETrack 在 Low Resolution（38.49%）、Fast Motion（35.41%）、Night（35.4%）场景优势明显

## 亮点
1. **首个无人机 RMOT 基准**：填补了航拍场景下语言引导跟踪的数据空白，数据规模远超现有 RMOT 数据集
2. **COALA 标注框架创新**：四阶段 Agent 协作将人工标注简化为"两次点击 + 审核"的模式，显著降低标注成本
3. **首次引入属性级评估**：逐帧标注 8 种挑战属性（夜间、遮挡、低分辨率、视角变化、尺度变化、快速运动、旋转、低分辨率），提出 HOTA_S 和 HOTA_M 指标
4. **CFE 的"共进化"思想**：不是简单的早融合或晚融合，而是让视觉结构化和语言引导同步迭代演进
5. **跨域泛化有趣发现**：跨域测试 HOTA 反而更高，作者分析原因是 UAVDT 仅有车辆类别，语义空间更简单

## 局限性 / 可改进方向
1. **未利用 LLM 推理能力**：当前架构基于传统 VL 融合范式，未引入大语言模型的高级推理
2. **部署效率不足**：51.4M 参数、15.6 FPS，难以在资源受限的无人机平台实时运行
3. **数据集依赖已有标注**：基于 VisDrone/UAVDT 扩展，继承了原始数据集的少量标注错误
4. **检测精度与定位精度的权衡**：HETrack 提升 DetA 的同时 LocA 略降（82.77 vs 其他方法 83+）
5. **目标类别有限**：训练集 10 类目标，跨域测试集仅车辆，缺乏对更丰富类别的验证

## 与相关工作的对比
- **vs TransRMOT/TempRMOT**：这些是 RMOT 的开创性工作但仅面向地面场景，HETrack 在 AerialMind 上 HOTA 提升约 5-8 个点
- **vs iKUN**：iKUN 通过不需要重训练的方式做 RMOT，但在 Refer-KITTI-V2 上性能很低（10.32 HOTA）
- **vs HFF-Track**：AAAI 2025 的工作，在 Refer-KITTI-V2 上 HFF-Track（36.18）略优于 HETrack（35.40），但 HETrack 在检测召回率上更强（41.16 vs 36.86）
- **数据集对比**：AerialMind 的表达式数量（24.6K）、实例数（293.1K）、bbox 标注量（46.14M）均远超所有已有 RMOT 数据集

## 启发与关联
- 与 ideas 中的 [预测即感知 (PAP) 小目标检测](../../ideas/object_detection/20260317_prediction_perception_small_det.md) 相关：该 idea 同样使用 VisDrone/UAVDT 数据集，AerialMind 的 SACR 模块也关注小目标感知，两者可互相借鉴
- COALA 标注框架的 Agent 协作思想可迁移到其他视频理解任务的标注效率提升
- CFE 的"共进化"融合范式对任何需要跨模态对齐的任务（如 referring segmentation、VQA）都有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个无人机 RMOT 基准，方法设计合理但非颠覆性创新
- 实验充分度: ⭐⭐⭐⭐⭐ In-domain/cross-domain/ground-level 三维度评估 + 属性级分析 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集和方法的动机阐述充分
- 对我的价值: ⭐⭐⭐ 数据集标注框架和跨模态融合思路有参考意义
