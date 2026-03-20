# Intrinsic Lorentz Neural Network

**会议**: ICLR 2026  
**arXiv**: [2602.23981](https://arxiv.org/abs/2602.23981)  
**代码**: 待确认  
**领域**: 模型架构 / 双曲几何  
**关键词**: hyperbolic neural network, Lorentz model, intrinsic operations, batch normalization, geometric deep learning  

## 一句话总结
提出完全内禀（fully intrinsic）的双曲神经网络 ILNN，所有运算均在 Lorentz 模型内完成，消除了现有方法中混合欧几里得操作的几何不一致性，在图像分类、基因组学和图分类上取得 SOTA。

## 研究背景与动机
1. **领域现状**：双曲神经网络利用双曲空间的指数增长体积表示层次结构数据，Lorentz 模型因数值稳定性好于 Poincare 模型成为首选。
2. **现有痛点**：现有双曲网络在某些操作中"退回"欧几里得空间（如 tangent space 线性变换、欧几里得 BN），导致几何不一致。
3. **核心矛盾**：如何在保持所有运算在双曲流形上的同时，设计足够表达且数值稳定的组件？
4. **本文要解决**：构建一套完全内禀的 Lorentz 空间操作集。
5. **切入角度**：用 Lorentz 超平面有符号距离替代仿射变换，用 gyro 结构实现内禀统计量。
6. **核心idea**：点到超平面距离 → FC 层；gyro-centering + gyro-scaling → BatchNorm。

## 方法详解

### 整体框架
ILNN 完全在 Lorentz 双曲面 $\mathbb{L}_K^n$ 上操作（$K<0$）。所有中间计算不离开流形，$K \to 0$ 极限退化为标准欧几里得网络。

### 关键设计

1. **PLFC (Point-to-Hyperplane Lorentz FC)**:
   - 学习 $m$ 个 Lorentz 超平面，计算输入到每个超平面的有符号距离作为 logits
   - 用 $\sinh$ 映射恢复空间坐标，由双曲面约束计算时间坐标
   - $K \to 0$ 时退化为标准仿射变换 $Wx + b$

2. **GyroLBN (Gyro-Lorentz Batch Normalization)**:
   - Gyro-centering：闭式 Lorentzian centroid（非迭代 Frechet mean）
   - Gyro-scaling：Frechet 方差归一化
   - 比 GyroBN 更快（闭式 vs 迭代），比 LBN 更内禀（双曲均值 vs 欧几里得均值）

3. **辅助组件**：Log-radius 拼接、Lorentz dropout、Lorentz activation、Gyro-bias

## 实验关键数据

### 图像分类
| 方法 | CIFAR-10 | CIFAR-100 |
|------|----------|----------|
| Euclidean ResNet-18 | 95.14% | 77.72% |
| HCNN-Lorentz | 95.14% | 78.07% |
| **ILNN** | **95.36%** | **78.41%** |

### 基因组学 (TEB) — MCC 指标
| 任务 | Euclidean | **ILNN** | 提升 |
|------|-----------|---------|------|
| Processed Pseudogene | 60.66 | **70.26** | +9.6 |
| Unprocessed Pseudogene | 51.94 | **64.90** | +13.0 |

### 基因组学 (GUE)
| 任务 | Best Prior | **ILNN** |
|------|-----------|----------|
| Covid 分类 | 36.71 | **64.76** |
| Core Promoter (tata) | 79.87 | **83.90** |

### 图分类
Airport 96.03%, Cora 85.68%, PubMed 82.52%——均为 SOTA。

### 关键发现
- CIFAR 上提升较小（+0.2-0.7pp），但基因组学任务提升巨大（+10-28pp）
- HCNN-S 在 Covid 分类上崩溃（36.71），ILNN 稳健——内禀操作带来数值稳定性
- GyroLBN 速度和效果都优于 GyroBN 和 LBN

## 亮点与洞察
- **完全内禀的设计哲学**：始终在双曲面操作 > 映射到切空间再映射回来
- **闭式 centroid 替代迭代求解**：GyroLBN 既快又准
- **$K \to 0$ 退化**性质优雅——双曲网络在平坦极限不应比欧几里得差
- 基因组学大幅提升暗示双曲表示对生物序列特别有效

## 局限性 / 可改进方向
- 仅基于 ResNet-18，未验证 ViT 等现代架构
- CIFAR 上绝对提升很小
- 固定曲率 $K=-1$，未探索可学习曲率
- 仅验证分类任务

## 相关工作与启发
- **vs HCNN**: HCNN 某些操作回到 tangent space，ILNN 全程内禀
- **vs Poincare 网络**: Lorentz 更数值稳定，ILNN 强化此优势
- 可启发 LLM 双曲嵌入设计

## 补充技术细节

### PLFC 的几何直觉
在欧几里得空间中，全连接层计算 $y = Wx + b$，本质上是将输入投影到多个超平面上。PLFC 将这一操作内禀化：在 Lorentz 双曲面上定义超平面，计算点到超平面的双曲距离作为特征。这个距离在 $K \to 0$ 时自然退化为欧几里得距离，保证了兼容性。

### 为什么基因组学提升巨大？
基因序列具有天然的层次结构（基因家族 → 基因 → 外显子 → 序列基元），双曲空间能用有限维度更好地捕捉这种指数增长的层级关系，而欧几里得空间在低维时会产生严重的表示拥挤。HCNN-S 在 Covid 分类上崩溃可能正是因为其混合操作丢失了关键的层次信息。

### Lorentz vs Poincaré 模型
Lorentz 模型使用 $(n+1)$ 维环境空间，坐标 $(x_0, x_1, ..., x_n)$ 满足 $-x_0^2 + x_1^2 + ... + x_n^2 = 1/K$。相比 Poincaré 球模型在边界附近有数值不稳定性，Lorentz 模型通过将时间坐标 $x_0$ 解析计算来避免数值问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ 完全内禀操作设计有价值
- 实验充分度: ⭐⭐⭐⭐ 覆盖图像/基因组/图三类任务
- 写作质量: ⭐⭐⭐⭐ 数学清晰，退化分析优雅
- 价值: ⭐⭐⭐⭐ 双曲几何在深度学习中的重要推进
