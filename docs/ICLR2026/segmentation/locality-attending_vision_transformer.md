# Locality-Attending Vision Transformer

**会议**: ICLR 2026  
**arXiv**: [2603.04892](https://arxiv.org/abs/2603.04892)  
**代码**: [GitHub](https://github.com/sinahmr/LocAtViT/)  
**领域**: segmentation / vision transformer  
**关键词**: ViT, locality, Gaussian attention, semantic segmentation, patch representation, dense prediction  

## 一句话总结
提出 LocAt，一个轻量级 ViT 插件，通过可学习高斯核调制自注意力偏向局部邻域(GAug)和无参数的 Patch 表征精炼(PRR)，在不改变训练范式的前提下为 ViT 带来 6%+ 的分割性能提升且不牺牲分类精度。

## 背景与动机
1. ViT 的全局自注意力在分类中表现出色，但会模糊密集预测所需的细粒度空间细节
2. 分类训练 ViT 中，patch token 逐渐丢失局部结构并向 [CLS] token 对齐
3. 现有改进(层级 ViT、窗口注意力)需要大幅修改架构，不适合基础模型
4. 分类目标未考虑密集预测需求，patch 位置的输出不受直接监督
5. GAP 聚合也存在问题：均匀梯度流让所有 patch 接收相同重要性
6. CLIP 等基础模型采用 vanilla ViT，增强 ViT 本身比设计新架构更有实用价值

## 方法详解
**Gaussian-Augmented Attention (GAug)**:
- 在注意力 logit 上加一个补充矩阵 $\mathbf{S}$：$\mathbf{Z} = \text{softmax}(\frac{\mathbf{q}\mathbf{k}^\top}{\sqrt{d}} + \mathbf{S})\mathbf{v}$
- $\mathbf{S}$ 由可学习高斯核生成：以每个 patch 为中心，方差从 query 预测
- $\boldsymbol{\Sigma} = f(\mathbf{q}_{sp} \mathbf{W}^\sigma)$，缩放系数 $\boldsymbol{\alpha} = \text{softplus}(\mathbf{q}_{sp}\mathbf{W}^\alpha)$
- 数据依赖的软局部性约束：$\alpha$ 小时接近标准全局注意力，$\alpha$ 大时强局部偏置
- [CLS] token 不受局部偏置影响

**Patch Representation Refinement (PRR)**:
- 在分类头前加一个无参数多头自注意力
- 将梯度路由到所有 patch token 位置
- 解决 ViT 分类训练中 patch 输出无直接监督的问题

**耦合关系**: PRR 将梯度传到最后一层的 GAug 参数使其能有效学习

## 实验关键数据
| 方法 (Tiny) | ADE20K mIoU | P-Context mIoU | COCO-Stuff mIoU | ImageNet Top-1 |
|---|---|---|---|---|
| ViT | 17.30 | 33.71 | 20.29 | 72.39 |
| **LocAtViT** | **23.47 (+6.17)** | **38.57 (+4.86)** | **26.15 (+5.86)** | **73.94 (+1.55)** |
| RegViT | 15.98 | 33.45 | 19.58 | 72.90 |
| **LocAt+RegViT** | **24.39 (+8.41)** | **39.90 (+6.45)** | **27.38 (+7.80)** | **74.08** |

- Base 规模在 ADE20K 上也提升 4%+
- 适用于 ViT/Swin/RegViT/RoPEViT/Jumbo 等多个基线
- FLOPs 增加可忽略（Tiny: 1.26→1.27G）
- 分割评估使用冻结主干 + 单层 MLP 解码器

## 亮点
- **极简设计**: 仅新增 $\mathbf{W}^\sigma$ 和 $\mathbf{W}^\alpha$ 两个小矩阵，PRR 完全无参数
- **不改变训练范式**: 用标准分类目标训练，segmentation-in-mind pretraining
- **通用性强**: 可插入任何 vanilla ViT，包括 CLIP 等大规模基础模型
- **分类不降反升**: 在多数模型上分类精度还有小幅提升

## 局限性
- 分割评估仅用冻结主干 + MLP，未在完整分割框架(如 UperNet)下充分验证
- 高斯核假设各向同性或二维独立，可能不适合所有场景
- 未在 CLIP-scale 基础模型上实测
- PRR 的无参数自注意力在更大分辨率下的计算开销未讨论

## 相关工作
- **Swin/PVT**: 层级 ViT，通过架构变化引入多尺度
- **DeiT/RegViT**: ViT 改进，Register token 吸收噪声但未解决梯度流问题
- **RoPE/RPE**: 位置编码引入空间感知，与 LocAt 正交互补
- **CLIP**: vanilla ViT 基础模型，LocAt 的潜在应用对象

## 评分
- 新颖性: ⭐⭐⭐⭐ (高斯注意力调制 + PRR 组合简洁新颖)
- 实验充分度: ⭐⭐⭐⭐ (5 种模型 × 3 种分割 benchmark + 分类)
- 写作质量: ⭐⭐⭐⭐⭐ (动机清晰，理论分析深入)
- 价值: ⭐⭐⭐⭐ (实用的 ViT 改进插件)
