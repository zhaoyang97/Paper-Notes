# A Unified Shape-Aware Foundation Model for Time Series Classification

**会议**: AAAI 2026  
**arXiv**: [2601.06429v1](https://arxiv.org/abs/2601.06429v1)  
**代码**: [https://github.com/qianlima-lab/UniShape](https://github.com/qianlima-lab/UniShape)  
**领域**: 时间序列分类 / 基础模型  
**关键词**: 时间序列分类, 基础模型, Shapelet, 原型学习, 可解释性  

## 一句话总结
提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

## 背景与动机
时间序列基础模型（FM）近年发展迅速，但绝大多数工作集中在**预测任务**（forecasting），如 Chronos、Moirai 等，而**分类任务**长期被忽视。预测关注的是趋势和周期性等时序动态，输出连续数值序列；分类则需要从固定长度的样本中提取判别性局部模式（如 ECG 中的 T 波异常），输出离散标签。因此，直接将预测型 FM 迁移到分类任务效果很差——GPT4TS、MOMENT、UniTS 在 UCR 分类上甚至不及非深度学习方法（Rocket 系列）。

此外，**可解释性**在时间序列分类中至关重要（尤其医疗领域），而 shapelet（判别性子序列）是经典的可解释工具。但现有 shapelet 方法依赖有标签监督，无法适配 FM 预训练场景；同时 shapelet 天然具有**多尺度**特性（不同长度的判别性模式），如何在 FM 中统一建模多尺度 shapelet 仍是未解决问题。

## 核心问题
1. 如何设计一个**专门面向分类**的时间序列基础模型，而非简单复用预测型 FM？
2. 如何在 FM 预训练框架中**无监督/弱监督地学习多尺度 shapelet 表示**，使其可迁移到不同领域？
3. 如何在保持分类性能的同时提供**可解释性**（哪些时间段对分类最关键）？

## 方法详解
UniShape 的核心思路是：用多尺度滑窗将时间序列切成不同粒度的子序列（shape），通过轻量 adapter 自适应地选出最有判别力的子序列尺度并聚合为 class token，再用原型对比学习在预训练阶段学习可迁移的 shapelet 模式。

### 整体框架
- **输入**: 单变量时间序列 x ∈ ℝ^T（统一 resize 到 T=512）
- **Shape-Aware Adapter**: 多尺度滑窗切子序列 → 归一化 + 线性映射为 shape token → 多分辨率 CNN 编码 → 注意力池化聚合为 class token → 由粗到细层级融合
- **Transformer Encoder**: 接收最终的 class token 和 shape token，输出精炼后的表示
- **Prototype-based Pretraining**: 实例级 + shape 级双层原型对比学习
- **输出**: class token 经分类头输出类别

### 关键设计

1. **Shape-Aware Adapter**:
   - 用 Q=5 个尺度的滑窗（窗长 W_q = 64, 32, 16, 8, 4）将时序切为不同粒度的子序列集合
   - 每个子序列经归一化（减去全局均值/标准差）后，拼接原始值编码、一阶差分编码、局部统计量嵌入，线性映射为 d 维 shape token
   - Adapter 内部由三个不同核大小的并行 1D CNN 提取多分辨率特征
   - 使用线性复杂度的**注意力池化**（Attention-based MIL pooling）将所有 shape token 加权聚合为一个 class token，注意力权重 α 直接反映每个子序列的判别重要性（可解释性来源）
   - 多尺度间采用**由粗到细的层级融合**：上一尺度的 class token 被拼接到下一尺度的 shape token 序列头部，逐级传递上下文信息
   - 所有尺度共享同一 adapter 参数，大幅减少参数量

2. **Instance-Prototype 对比学习**:
   - 维护一组可学习的类原型向量 {p_c}，用 EMA 动态更新
   - 有标签样本通过 ground-truth 对应原型更新；无标签样本通过与最近原型的余弦相似度分配伪标签
   - Instance 级对比损失 L_ins 拉近 class token 与对应类原型，推远与其他类原型的距离

3. **Shape-Prototype 对比学习**:
   - 从 shape token 中选出注意力分数最高的 top-ε（默认 60%）作为高置信度 shape token
   - Shape 级对比损失 L_shape 让这些高置信 shape token 也与对应类原型对齐
   - 这一设计让模型不仅在全局（实例级）学到类判别特征，还在局部（shape 级）学到 shapelet 模式

### 损失函数 / 训练策略
- **预训练损失**: L_pretrain = L_proto + L_self
  - L_proto = (1-λ)·L_ins + λ·L_shape，λ=0.01 平衡实例级和 shape 级
  - L_self: MoCo v3 自监督对比损失（两个随机裁剪视图的一致性），支持弱监督预训练
- **微调损失**: L_finetune = L_ce + μ·L_shape
  - 交叉熵 + 辅助 shape 对比损失（μ=0.01），微调时也保持 shapelet 的判别性学习
- 预训练 30 epochs, batch size 2048, 默认仅用 10% 标签数据（实验证明 10% 与 100% 标签差异不显著）
- 微调 300 epochs，选择训练 loss 最低的 checkpoint 做测试
- 动量系数 β=0.9，shape token 选择比例 ε=60%

## 实验关键数据
| 数据集 | 指标 | UniShape | 之前SOTA (MR-H/Mantis) | 提升 |
|--------|------|----------|------------------------|------|
| 128 UCR (全监督) | Avg. Acc | **0.8708** | 0.8621 / 0.8441 | +0.87% / +2.67% |
| 128 UCR (全监督) | Avg. Rank | **2.71** | 3.97 / 5.21 | - |
| 30 额外数据集 (零样本特征提取) | Avg. Acc | **0.7262** | 0.7052 (Mantis) | +2.1% |
| 30 额外数据集 (零样本) | Avg. Rank | **3.07** | 3.67 (Mantis) | - |

- UniShape 仅 **3.1M 参数**，远小于 GPT4TS (84.1M)、MOMENT (341.2M)
- 所有 P-value < 0.05（Wilcoxon 签名秩检验），显著优于所有基线

### 消融实验要点
- **去掉预训练**：准确率从 85.29% 降至 83.65%（-1.64%），预训练贡献显著
- **去掉 Adapter**：准确率降至 84.28%（-1.01%），多尺度 shapelet 建模有效
- Adapter 中 CNN 替换为 Transformer 或 MLP 均会掉点，CNN 更适合多尺度 shapelet 特征提取
- **去掉 Instance-Prototype (w/o Ins)**：掉点 0.85%；**去掉 Shape-Prototype (w/o Shape)**：掉点 0.59%；两者都去掉：掉点 1.18%——说明实例级原型更重要，两者互补
- Transformer encoder 在 FM 设定下优于 CNN encoder（与 domain-specific 训练中 CNN 更优的结论相反），MLP encoder 表现极差（-28.8%）
- 标签比例：10% 标签与 100% 标签性能差异不显著（P>0.05），说明模型对标签量不敏感

## 亮点
- **分类导向的 FM 设计**：明确指出预测型 FM 不适合分类任务，从 shapelet 角度重新出发，是少有的专门为 TSC 设计 FM 的工作
- **注意力池化提供天然可解释性**：shape token 的注意力权重直接指示判别性时间段，无需额外的后置解释方法。在 ECGFiveDays 上准确定位了已知的 T 波异常区间 [75,95]
- **参数高效**：3.1M 参数大幅优于同类 FM（MOMENT 341M, GPT4TS 84M），但性能更强
- **共享参数 adapter**：所有尺度共享同一 adapter，既实现了变长序列统一处理，又控制了参数量
- **弱监督预训练可行**：仅需 10% 标签即可达到接近全监督的效果，伪标签 + 原型学习机制有效

## 局限性 / 可改进方向
- **仅支持单变量**：多变量时间序列需拆分为独立通道处理（channel-independent），丢失通道间相互依赖关系。作者在 Conclusion 中明确提到这是主要局限
- **固定长度假设**：所有输入需 resize 到 T=512，插值可能丢失或引入伪特征，对原始长度远小于或远大于 512 的序列不友好
- **预训练数据域覆盖**：虽然 1.89M 样本来自多领域，但主要是 UCR/UEA 这类"干净"的基准数据，对工业级噪声时序的鲁棒性未验证
- **滑窗尺度固定**：5 个窗长 {64,32,16,8,4} 是手工设定的，未探索自适应尺度选择
- → 多变量扩展方向参见 [ideas/llm_nlp/20260317_multivariate_shape_aware_fm.md](../../../ideas/llm_nlp/20260317_multivariate_shape_aware_fm.md)

## 与相关工作的对比
- **vs. Mantis/NuTime**（分类导向 FM）：UniShape 通过 shapelet 显式建模获得更好的准确率和可解释性，Mantis 和 NuTime 关注多尺度归一化但缺乏 shapelet 概念。UniShape 3.1M vs Mantis 8.7M，参数更少性能更优
- **vs. MOMENT/GPT4TS/UniTS**（通用 FM）：这些 FM 主要面向预测任务，迁移到分类后表现远不及非深度方法，验证了分类需要专门设计的观点
- **vs. SoftShape**（Shapelet 方法）：SoftShape 是端到端 shapelet 学习方法但限于 domain-specific 训练，UniShape 将 shapelet 思想扩展到 FM 预训练范式，泛化能力更强

## 启发与关联
- 该工作验证了**任务专用 FM > 通用 FM**的观点——分类需要判别性局部特征而非时序动态建模，这一思路或可迁移到其他任务专用 FM 设计
- 已有相关 idea：[多变量时间序列的跨通道 Shape-Aware 基础模型](../../../ideas/llm_nlp/20260317_multivariate_shape_aware_fm.md)——在 UniShape 基础上增加跨通道交互模块处理多变量时间序列
- 注意力池化 + 原型学习的组合是一种通用的弱监督表示学习框架，可扩展到其他需要可解释性的序列分类场景（如文本分类、生物序列分类）

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 shapelet 思想融入 FM 预训练是新颖的角度，但各个组件（MIL pooling、原型学习、MoCo）均为成熟技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 128 UCR + 30 额外数据集 + 16 个基线 + 多维度消融 + 可解释性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机论证有力，但部分公式符号密集可读性一般
- 价值: ⭐⭐⭐⭐ 填补了时间序列分类 FM 的空白，但限于单变量场景，实际应用场景偏受限
