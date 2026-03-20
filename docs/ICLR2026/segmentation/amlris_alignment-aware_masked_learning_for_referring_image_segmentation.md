# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

**会议**: ICLR 2026  
**arXiv**: [2602.22740](https://arxiv.org/abs/2602.22740)  
**代码**: [GitHub](https://github.com/pipashu1/AMLRIS)  
**领域**: segmentation  
**关键词**: referring image segmentation, vision-language alignment, masked learning, cross-modal similarity  

## 一句话总结
提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。

## 背景与动机
1. 指称图像分割(RIS)需要根据自然语言表达精准分割图像中的目标对象，依赖跨模态精细对齐
2. RIS 训练中每个样本通常仅有一个标注目标，监督信号稀疏
3. 理解"离人最近的长颈鹿"等表达需要依赖视觉上下文中其他物体的空间关系
4. 现有方法(LAVT/CARIS/DETRIS)通过复杂融合模块增强对齐，但对全部像素施加损失会引入不可靠梯度
5. 在密集损失下，模型容易过拟合到与表达无关的区域
6. 数据增强方法(翻转/颜色抖动)易破坏指称表达的语义一致性

## 方法详解
**整体框架**: 两阶段训练(共享参数)——第一阶段前向计算对齐图并生成遮蔽，第二阶段在遮蔽图像上正常训练

**PatchMax Matching Evaluation (PMME)**:
- 将视觉特征 $V$ 和文本特征 $T$ 分别 $\ell_2$ 归一化
- 用随机高斯矩阵 $W_i, W_t$ 投影到公共 $D_a$ 维空间(Johnson-Lindenstrauss 保距)
- 计算 $S_{norm} = \text{SoftMax}(V'T'^{\top})$，每个 patch 取与最强匹配 token 的最大相似度

**Alignment-Aware Filtering Mask (AFM)**:
- 将 patch 级相似度双线性上采样到像素级
- 低于阈值 $\tau$ 的像素标记为弱对齐，随机保留 $1-\rho$ 比例(防过滤)
- 按 block 聚合遮蔽(任一像素弱对齐则整块遮蔽)，对输入图像 zero-out

**关键超参**: $\tau=0.4$, $\rho=0.25$, block $32\times32$, $D_a=2048$

**损失**: 标准交叉熵分割损失 $\mathcal{L}_{seg}$，无额外损失项

## 实验关键数据
| 方法 | RefCOCO val | RefCOCO+ val | RefCOCOg val | Avg mIoU |
|------|-------------|--------------|--------------|----------|
| CARIS* | 76.77 | 69.33 | 68.87 | 71.8 |
| MagNet | 77.43 | 70.10 | 68.53 | 72.1 |
| **AMLRIS** | **77.89** | **71.33** | **69.24** | **72.9** |

- oIoU 指标：RefCOCO val 75.45(+0.80 vs CARIS），RefCOCO+ val 67.37（+1.83)
- 全部 8 个 split 均 SOTA
- 跨数据集鲁棒性：仅在 RefCOCO+ 训练，在 7 种扰动场景下均优于 baseline
- 额外开销：仅增加 4.9% 显存和 17.2% 训练时间，推理无开销

## 亮点
- **即插即用**: 不修改模型架构，不增加推理成本，可集成到 DETRIS/CARIS/ReLA 等多种 RIS 框架
- **理论保证**: 用 JL 引理严格证明随机投影保持跨模态内积，alignment 度量有理论支撑
- **鲁棒性强**: 遮挡/噪声等扰动场景下性能优势更明显

## 局限性
- 阈值 $\tau$ 和 dropout 比例 $\rho$ 需要手动调节，对不同数据集可能需要不同设置
- 随机投影的对齐度量未必能捕捉深层语义对齐（仅基于初始特征相似度）
- 两阶段前向带来 17.2% 训练时间增加
- 仅在 RefCOCO 系列评估，未验证在开放词汇/更复杂场景下的泛化性

## 相关工作
- **CARIS/LAVT/DETRIS**: 本文 baseline 和对比，通过融合架构提升对齐
- **MaskRIS/NeMo/MagNet**: 数据增强路线，但仍用全像素损失
- **CRIS**: 基于 CLIP 的像素级适配方法

## 评分
- 新颖性: ⭐⭐⭐⭐ (对齐感知遮蔽思路简洁新颖)
- 实验充分度: ⭐⭐⭐⭐ (全 split SOTA + 鲁棒性 + 消融)
- 写作质量: ⭐⭐⭐⭐ (理论推导完整清晰)
- 价值: ⭐⭐⭐⭐ (通用训练策略，实用价值高)
