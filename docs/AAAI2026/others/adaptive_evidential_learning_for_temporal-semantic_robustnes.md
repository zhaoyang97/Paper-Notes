# Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval

**会议**: AAAI 2026  
**arXiv**: [2512.00953v1](https://arxiv.org/abs/2512.00953v1)  
**代码**: [https://github.com/KaijingOfficial/DEMR](https://github.com/KaijingOfficial/DEMR) (有)  
**领域**: 视频理解 / 时刻检索 / 不确定性估计  
**关键词**: Moment Retrieval, Evidential Learning, 不确定性估计, 跨模态对齐, 去偏  

## 一句话总结
提出 DEMR 框架，将深度证据回归（DER）引入视频时刻检索任务，通过 Reflective Flipped Fusion 模块缓解模态不平衡、通过 Geom-regularizer 修复原始 DER 中不确定性估计的反直觉偏差，在标准和去偏数据集上均取得了显著提升。

## 背景与动机
视频时刻检索（Moment Retrieval, MR）要求根据自然语言查询在未剪裁视频中定位对应时间片段。现有方法主要基于预训练的 Transformer（如 CLIP-ViT），但存在两个核心痛点：
1. **确定性推理的局限**：主流方法采用确定性范式，对困难帧（如查询中提到的物体不在画面中）缺乏有效的应对策略，推理时只能靠 NMS 选择最高分 proposal，面对歧义场景容易过度自信。
2. **CLIP 特征的模态偏差**：CLIP 主要在静态图像-文本对上预训练，偏向捕捉物体级视觉特征，对动态动作和文字语义的细粒度理解不够，导致跨模态融合时过度依赖视觉信息。

作者尝试将深度证据回归（DER）直接引入 MR 作为 baseline，发现两个新问题：(a) 简单拼接多模态特征无法解决模态不平衡；(b) DER 原始正则项的梯度只与误差相关、与证据量无关，导致低误差样本的证据反而被过度抑制，高误差样本的不确定性反而偏低——这是反直觉的。

## 核心问题
如何在多模态时刻检索中实现可靠的不确定性建模？具体包含两个子问题：
1. 如何缓解视觉-文本模态不平衡，使不确定性估计对两个模态都敏感？
2. 如何修复 DER 正则项中"准确预测的证据反被抑制"的结构性缺陷？

## 方法详解

### 整体框架
DEMR 的整体流程：输入未剪裁视频和自然语言查询 → 使用冻结的 CLIP-ViT + SlowFast 提取视频特征，CLIP 提取文本特征 → 通过 **RFF 模块**进行渐进式跨模态对齐 → **MR Head** 预测时间边界 + **DER Head** 估计不确定性 → 训练分两阶段：先用 **QR 任务**增强文本敏感性，再联合训练 MR 和证据学习。推理时利用不确定性辅助选择 proposal。

### 关键设计
1. **Reflective Flipped Fusion (RFF) 模块**：采用双分支结构，在每一层中交替翻转视频和文本特征的角色（Query ↔ Key/Value），通过共享的交叉注意力 + 各自的自注意力实现渐进式跨模态对齐。这种"反射翻转"设计比简单拼接更充分地建模了双向模态交互，让视觉和文本分支都能获得足够的跨模态信息。
2. **Query Reconstruction (QR) 辅助任务**：在训练早期阶段，随机 mask 查询中的一个名词（名词是 CLIP 特征最擅长捕捉的语义单元），要求模型利用视频上下文和剩余文本 token 重建被 mask 的 token。这迫使模型学会从视频中提取文本相关的语义信息，增强文本分支的敏感性。QR 只在前 30 个 epoch 训练，之后冻结。
3. **Geom-regularizer**：针对原始 DER 正则项 $\mathcal{L}^R = \Delta \cdot \Phi$（误差×证据）的梯度 $-\nabla_\Phi \mathcal{L}^R = -\Delta$ 只依赖误差不依赖证据量的问题，提出基于几何约束的新正则项。核心思想是：将归一化后的误差 $\bar{\Delta}$ 和证据 $\bar{\Phi}$ 约束在直线 $\bar{\Phi} + \bar{\Delta} = 1$ 上，即 $\mathcal{L}^L = \|\bar{\Phi} + \bar{\Delta} - 1\|_2^2$。其梯度 $-\nabla_{\bar{\Phi}} \mathcal{L}^L = -2(\bar{\Delta} + \bar{\Phi} - 1)$ 同时依赖误差和证据，实现了"准确预测应有高证据、不准确预测应有低证据"的自适应调节。

### 损失函数 / 训练策略
- **总损失**: $\mathcal{L} = \mathcal{L}_{mr} + \lambda_{der} \cdot \frac{2}{N} \sum_i \mathcal{L}_i^e + \mathcal{L}_{qr}$
- 其中 $\mathcal{L}_{mr}$ 包含 Smooth L1 + GIoU loss（仅对前景 clip）
- $\mathcal{L}_i^e = \lambda_{NLL} \mathcal{L}_{NLL} + \lambda_{geom} \mathcal{L}^L$（NIG 负对数似然 + Geom 正则）
- **两阶段训练**：Stage 1 训练 QR 模块（30 epoch, lr=1e-5），Stage 2 训练 MR + DER（Geom 正则的梯度对 MR 分支 detach，专注优化不确定性）
- 关键超参: $\lambda_{geom}=10^{-2}$, $\lambda_{der}=10^{-3}$

## 实验关键数据
| 数据集 | 指标 | DEMR | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| QVHighlights (val) | mAP | 43.0 | 42.9 (CG-DETR) | +0.1 |
| QVHighlights (val) | R1@0.5 | 65.0 | 67.4 (CG-DETR) | -2.4 |
| Charades-STA | R1@0.5 | 60.2 | 58.4 (CG-DETR) | +1.8 |
| Charades-STA | mIoU | 51.6 | 50.1 (CG-DETR/UniVTG) | +1.5 |
| TACoS | R1@0.5 | 37.3 | 39.5 (CG-DETR) | -2.2 |
| QVHighlights (test, MLLM) | mAP@0.75 | 56.82 | 54.40 (LLaVA-MR) | +2.42 |
| Charades-CD | R1@0.3 IID-OOD gap | 3.29% | 12.00% (CM-NAT) | -8.71% |

### 消融实验要点
- **RFF 模块**：mAP 从 61.1 → 62.4（+1.3），并将视觉-文本不确定性方差差距 ΔVar 从 8.32 降至 7.03
- **QR 任务**：进一步提升至 63.8（+1.4），ΔVar 从 7.03 降至 0.98，模态平衡效果显著
- **Geom-regularizer**：完整模型 65.0（+1.2），且实现了"误差越大不确定性越高"的正确校准
- QR 最优设置：mask 1 个名词、训练 30 epoch、lr=1e-4
- $\lambda_{geom}$ 在 $10^{-2}$ 处最优，$\lambda_{der}$ 超过 $10^{-2}$ 性能明显下降

## 亮点
1. **首次将证据回归引入时刻检索**，并系统性分析了直接迁移的问题（模态不平衡+反直觉不确定性）
2. **Geom-regularizer 设计精巧**：用一条简单的几何约束线 $\bar{\Phi}+\bar{\Delta}=1$ 就解决了梯度场的结构缺陷，思路简洁而有效
3. **可解释性强**：不确定性可以直观地反映模型在 OOD 区域的低信心、在歧义查询上的高 epistemic uncertainty，为 MR 模型提供了可信度信号
4. **去偏泛化**：在 Charades-CD/ActivityNet-CD 上的 IID-OOD gap 极小（3.29%），远优于确定性方法

## 局限性 / 可改进方向
1. **骨干网络受限**：使用冻结的 CLIP-ViT/SlowFast，未利用更强的 VLM 骨干（如 InternVideo2、LanguageBind），论文自身也提到与 MLLM 结合是未来方向
2. **QR 的名词依赖**：QR 任务只 mask 名词，对动词/形容词等语义的增强不足，但 MR 中动作语义同样重要
3. **NMS 仍是必需的**：不确定性虽然提供了额外信号，但推理时仍依赖 NMS 做最终筛选，未能完全利用不确定性进行 proposal 选择
4. **计算开销**：RFF 的多层交叉注意力+DER 的 NIG 分布学习增加了训练/推理成本，论文未报告效率对比

## 与相关工作的对比
- **vs UniVTG / QD-DETR / CG-DETR**：这些是确定性 MR 方法，DEMR 的核心优势不在绝对性能（某些指标略低于 CG-DETR），而在于提供了不确定性估计能力和更好的去偏泛化
- **vs MomentDiff**：基于扩散的 MR 方法，DEMR 在所有数据集上优于它，且提供不确定性量化
- **vs 原始 DER (Amini 2020)**：DEMR 的 Geom-regularizer 修复了原始正则项的梯度场缺陷，是对 DER 框架的重要改进，可能对其他回归任务也有参考价值

## 启发与关联
- **Geom-regularizer 的通用性**：这个正则项的设计思路（把误差和证据约束在一条线上）不局限于 MR，可以迁移到任何使用 DER 的回归任务，如深度估计、姿态估计等
- **与 ideas 的关联**：[证据性学习与VLM结合的医学图像分割](../../ideas/medical_imaging/20260317_evidential_rejectable_med_seg.md) 同样使用证据性学习做不确定性估计，但面向分类（Dirichlet 分布）而非回归（NIG 分布）。DEMR 的 Geom-regularizer 思想（误差-证据几何约束）可能也能迁移到分类场景，用于改善 EDL 的 KL 正则项
- **不确定性引导的主动学习/标注**：DEMR 在 OOD 区域产生高 epistemic uncertainty 的特性，天然适合用于主动标注——优先标注模型最不确定的时间段

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次将 DER 引入 MR 并系统性解决迁移问题)
- 技术深度: ⭐⭐⭐⭐ (Geom-regularizer 的梯度分析和几何约束设计有深度)
- 实验充分度: ⭐⭐⭐⭐⭐ (标准+去偏数据集、丰富的消融和可视化)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，可视化丰富)
- 实用价值: ⭐⭐⭐⭐ (代码开源，不确定性估计对下游应用有价值)
