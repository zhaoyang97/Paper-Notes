# Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token

**会议**: CVPR 2026  
**arXiv**: [2603.19026](https://arxiv.org/abs/2603.19026)  
**代码**: [https://github.com/ANDYZAQ/SELF1E](https://github.com/ANDYZAQ/SELF1E)  
**领域**: 多模态VLM  
**关键词**: MLLM分割, 无解码器分割, 单token分割, Pixel-Unshuffle, 特征精化

## 一句话总结
提出 SELF1E，首次实现不依赖专用 mask 解码器且仅用单个 [SEG] token 的 MLLM 分割方法，通过 Residual Features Refilling (RFR) 和 Residual Features Amplifier (RFA) 恢复 pixel-shuffle 压缩造成的分辨率损失，在多个分割任务上达到与解码器方法竞争力相当的性能。

## 研究背景与动机
1. **领域现状**：MLLM 分割方法（LISA、GSVA、OMG-LLaVA 等）主要通过在 MLLM 上挂载专用 mask 解码器（SAM / Mask2Former）来生成分割掩码。
2. **现有痛点**：
   - 专用解码器引入额外参数和复杂结构，破坏方法的简洁性且依赖外部基础模型
   - UFO 尝试无解码器方案，但需要 16 个 [SEG] token 来补偿分辨率损失，增加计算成本
   - 问题根源：现代 MLLM 的 pixel-shuffle 下采样使视觉特征分辨率大幅降低（如 4 倍压缩），丢失了分割所需的细粒度空间信息
3. **核心矛盾**：pixel-shuffle 压缩是 MLLM 高效处理的必要手段，但压缩导致的空间信息丢失是无解码器分割的根本瓶颈。
4. **本文要解决什么**：证明单个 [SEG] token 足以实现高质量分割，瓶颈不在 token 数量而在特征分辨率。
5. **切入角度**：压缩前的图像编码器特征保有完整分辨率，可以作为"预压缩特征"保留；LLM 处理后的特征带有更精细的语义区分度；两者互补。
6. **核心idea**：保留编码器输出的未压缩特征+收集 LLM 各层的残差特征并上采样融合+用 Pixel-Unshuffle 进一步放大分辨率。

## 方法详解

### 整体框架
图像 → Vision Encoder → 分支1: pixel-shuffle+MLP 压缩 → LLM → [SEG] token + 压缩图像特征；分支2: 自复制保留未压缩特征 → RFR 融合残差 → RFA 进一步放大 → 点积生成高分辨率 mask。

### 关键设计

1. **Residual Features Refilling (RFR)**:
   - 保留编码器输出的未压缩特征 $F_{V_1}^{HQ} \in \mathbb{R}^{N_0 \times d}$（通过将每个 pixel 自复制 $\alpha$ 次后过同一 MLP 实现）
   - 收集 LLM 处理前后的残差：$F_R = F_{IMG} - F_{V_1}$
   - 上采样残差并融合：$F_{IMG}' = F_{V_1}^{HQ} + \mathcal{I}(F_R)$
   - 效果：将 LLM 学到的细粒度语义区分度注入到高分辨率特征中

2. **Residual Features Amplifier (RFA)**:
   - 对 $F_{V_1}$（LLM前）和 $F_{IMG}$（LLM后）分别施加 MLP + Pixel-Unshuffle 操作
   - 放大后残差 $F_{RFA} = f_{PUS}'(F_{IMG}) - f_{PUS}(F_{V_1})$
   - 最终融合 $F_{IMG}' = f_{PUS}(F_{V_1}^{HQ}) + \mathcal{I}(F_{RFA})$，分辨率达到 $\alpha N_0 \times d$
   - 设计动机：压缩特征的每个 embedding 隐含了 $\alpha$ 个像素的信息，Pixel-Unshuffle 可以恢复这些隐含信息
   - [SEG] token 也同样过 Pixel-Unshuffle 后取平均：$F_{SEG}' = \text{mean}(f_{PUS}'(F_{SEG}))$

3. **分割专用注意力掩码**:
   - 设计双感知路径：image-to-image（图像 token 间双向注意力）+ image-to-segmentation（图像 token 与 [SEG] token 双向交互）
   - 比标准因果注意力提供更丰富的像素间和像素-语义交互
   - 确保 [SEG] token 能充分感知所有图像位置的信息

### 损失函数 / 训练策略
基于 InternVL 系列训练。RFA 中的两个 Pixel-Unshuffle MLP 需要训练。

## 实验关键数据

### 主实验（Referring Expression Segmentation）

| 方法 | 无专用解码器 | 单token | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|:----------:|:------:|:-----------:|:------------:|:------------:|
| LISA-7B | ✗ | ✓ | 74.9 | 65.1 | 67.9 |
| u-LLaVA | ✗ | ✓ | 83.0 | 77.1 | 77.1 |
| UFO (16-token) | ✓ | ✗ | - | - | - |
| **SELF1E** | **✓** | **✓** | **~80+** | **~73+** | **~75+** |

### 消融实验

| 配置 | 关键效果 |
|------|---------|
| 压缩分辨率直接预测 | IoU 显著低（约低 10+%） |
| + RFR（仅残差填充） | IoU 大幅提升，证明高分辨率+语义残差有效 |
| + RFA（残差放大） | 进一步提升 2-3%，Pixel-Unshuffle 恢复隐含信息 |
| + 分割注意力掩码 | 额外提升 1-2%，双向交互有帮助 |

### 关键发现
- 首次证明：无专用解码器 + 单 token 的 MLLM 分割是可行的，性能接近带 SAM/Mask2Former 的方法
- RFR 贡献最大：恢复高分辨率特征是关键，而非增加 [SEG] token 数量
- 保持VQA能力：分割训练不会损害模型的通用 VQA 性能
- pixel-shuffle 压缩是分辨率瓶颈的根源，而非 [SEG] token 数量

## 亮点与洞察
- **挑战了"分割必须用解码器"的主流范式**：证明 MLLM 本身具备分割能力，只需恢复被压缩的空间信息即可
- **RFR/RFA 的设计哲学**：不增加新模块，而是巧妙利用 MLLM 中已有的信息（编码器特征、LLM 残差、pixel-shuffle 的逆操作），用"减法+加法"恢复丢失的信息
- **对 MLLM 架构设计的洞察**：pixel-shuffle 压缩虽然对 VQA 友好，但对像素级任务是根本性障碍，未来 MLLM 设计需要考虑如何在压缩中保留空间信息

## 局限性 / 可改进方向
- 当前性能仍略低于最强的带解码器方法（如 u-LLaVA），有提升空间
- RFA 中的 Pixel-Unshuffle MLP 引入了额外训练参数
- 分割注意力掩码需要修改 LLM 的注意力计算，不完全是 plug-and-play
- 开放词汇分割因为类别词汇的歧义性而更具挑战

## 相关工作与启发
- **vs LISA / GLaMM**：它们将 [SEG] token 送入 SAM 解码器生成 mask，依赖外部模型能力。SELF1E 完全自给自足
- **vs UFO**：UFO 也去掉了解码器但需 16 个 [SEG] token，本质上是用 token 数量弥补分辨率不足。SELF1E 直接解决分辨率问题，单 token 即可

## 补充分析
- 基于 InternVL 系列的 pixel-shuffle 比例 $\alpha$ 通常为 4，即压缩后分辨率降为 1/4
- 自复制操作（self-replication）将每个像素特征复制 $\alpha$ 次后过同一 MLP，模拟了邻近像素的 pre-shuffled 特征
- RFR 和 RFA 可以独立使用或组合，组合效果最优
- 分割注意力掩码的双感知路径允许图像 token 和 [SEG] token 双向交互，而标准因果注意力只允许单向
- 整个方法不引入外部分割基础模型（SAM/Mask2Former），真正实现了 MLLM-only 分割

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 挑战主流范式，首次实现无解码器单token分割
- 实验充分度: ⭐⭐⭐⭐ 多任务验证，消融充分
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，图示直观
- 价值: ⭐⭐⭐⭐ 简化了MLLM分割流水线，启发未来架构设计
