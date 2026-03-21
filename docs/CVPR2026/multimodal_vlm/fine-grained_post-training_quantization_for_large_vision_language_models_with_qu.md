# Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients

**会议**: CVPR 2026
**arXiv**: [2603.17809](https://arxiv.org/abs/2603.17809)
**代码**: https://github.com/ucas-xiang/QIG
**领域**: 多模态VLM
**关键词**: 后训练量化, LVLM压缩, token级敏感度, 积分梯度, 模型加速

## 一句话总结
提出量化感知积分梯度（QIG），将 LVLM 量化的灵敏度分析从模态级推进到 token 级，利用公理化归因原理精确量化每个 token 对量化误差的贡献，在 W4A8 和 W3A16 设置下显著提升量化模型精度，且几乎无额外计算开销。

## 研究背景与动机
1. **领域现状**：LVLM（如 LLaVA、InternVL、Qwen-VL）在多模态任务中表现出色，但模型体积大、推理慢，后训练量化（PTQ）是常用的加速手段。
2. **现有痛点**：现有 LVLM 量化方法（如 MBQ）仅在模态级别衡量 token 敏感度（视觉 vs 文本），忽略了跨 token 的复杂交互以及 token 间的量化敏感度差异。
3. **核心矛盾**：随着 token 在模型中逐层交互，模态边界逐渐模糊，同一模态内不同 token 的量化敏感度也存在巨大差异（massive activations、layer heterogeneity、sub-layer divergence、token variability 四个现象）。
4. **本文要解决什么？** 如何在 token 级别精确估计量化敏感度，并利用这些信息指导更精细的 channel-wise equalization。
5. **切入角度**：从机械可解释性中的公理化归因出发，利用积分梯度量化每个 token 从量化参考到实际输入的敏感度。
6. **核心idea**：用 Quantization-aware Integrated Gradients（QIG）替代模态级敏感度估计，在 token 级别指导量化校准。

## 方法详解

### 整体框架
输入多模态序列（视觉+文本+特殊token）→ 在校准阶段计算每个 token 的 QIG 分数 → IQR 裁剪和归一化 → 将 token 重要度系数 $\lambda_i$ 融入 channel-wise equalization 的优化目标 → 搜索最优量化缩放因子。

### 关键设计

1. **Quantization-aware Integrated Gradients (QIG)**:
   - 做什么：在 token 级别量化每个 token 对量化误差的贡献
   - 核心思路：不同于经典 IG 归因全精度预测，QIG 归因的是全精度模型和量化模型之间的输出差异。沿 $x^q$（量化输入）到 $x$（实际输入）的路径积分梯度：$QIG(x) = (x - x^q) \int_0^1 \frac{\partial(f(x_\alpha, w) - f(x_\alpha, w^q))}{\partial x_\alpha} d\alpha$
   - 设计动机：梯度和注意力等常用代理与量化误差的相关性弱，perturbation-based 方法虽准确但计算代价高。QIG 直接与 PTQ 误差关联，且满足完备性公理

2. **IQR 裁剪稳定化**:
   - 做什么：抑制 QIG 分数中的极端值
   - 核心思路：用四分位距裁剪 $C(QIG_i) = \text{clip}(QIG_i, Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR)$，然后归一化得到 $\lambda_i$
   - 设计动机：原始 QIG 分布重尾，少数极端 token 会主导优化

3. **Token 级加权 Channel-Wise Equalization**:
   - 做什么：将 token 重要度系数 $\lambda_i$ 融入 CWE 优化目标
   - 核心思路：$\mathbf{E}^* = \arg\min_{\mathbf{E}} \sum_{i=1}^T \lambda_i \|Q_W(\mathbf{W}*\mathbf{E}) Q_X(\mathbf{E}^{-1}*\mathbf{X}_i) - \mathbf{W}\mathbf{X}_i\|_2^2$
   - 设计动机：让缩放因子搜索偏向更敏感的 token，整体框架不变但精度更高

### 训练策略
- 完全无训练（PTQ），仅在校准阶段使用 128 对 ShareGPT4V 图文对
- 支持 weight-only (W3A16) 和 weight-activation (W4A8) 两种设置

## 实验关键数据

### 主实验（LLaVA-onevision-7B）
| 设置 | 方法 | VizWiz | MMMU | ChartQA | AI2D | ScienceQA | 平均 |
|------|------|--------|------|---------|------|-----------|------|
| FP16 | - | 60.41 | 49.22 | 80.04 | 81.31 | 95.88 | 73.37 |
| W3A16 | MBQ | 57.99 | 44.00 | 76.84 | 78.47 | 94.89 | 70.44 |
| W3A16 | **QIG** | **62.82** | **45.78** | **77.20** | **79.11** | **95.29** | **72.04** |
| W4A8 | MBQ | 58.13 | 44.78 | 74.92 | 78.27 | 94.70 | 70.16 |
| W4A8 | **QIG** | **59.10** | **45.00** | **74.52** | **78.30** | **94.25** | **70.23** |

### 消融实验
| 敏感度类型 | 粒度 | VizWiz 精度 |
|-----------|------|------------|
| 梯度 (SFT loss) | 模态级 | 57.36 |
| 梯度 | token级 | 55.78 (↓) |
| 注意力 | token级+special | 57.52 |
| 扰动 | token级+special | 57.72 |
| **QIG** | **token级** | **最优** |

### 关键发现
- W3A16 下 QIG 在 LLaVA-onevision-7B 上比 MBQ 平均提升 1.60%，与全精度差距仅 1.33%
- SFT 梯度做 token 级敏感度反而比模态级更差，说明 SFT 梯度与量化敏感度不对应
- 注意力 score 因 attention-sink 现象给出不稳定结果
- QIG 的 token 级敏感度与实际量化误差有强相关性

## 亮点与洞察
- **用可解释性工具解决工程问题**：巧妙地将积分梯度从"解释模型预测"迁移到"量化量化误差"，公理化归因给敏感度估计提供了理论保障
- **零额外推理开销**：QIG 仅在校准阶段计算，量化后的推理与基线完全相同
- 对 SFT 梯度和注意力这两种直觉上应该有效的代理进行了系统性否定，增强了 QIG 的说服力

## 局限性 / 可改进方向
- 校准集固定为 128 样本，未探索校准集选择对 QIG 的影响
- QIG 的积分步数是超参数，论文未充分讨论其敏感性
- 仅在 7B-26B 规模验证，更大模型（70B+）的效果未知
- IQR 裁剪的 1.5 倍为经典统计默认值，是否是量化场景下的最优选择值得探讨

## 相关工作与启发
- **vs MBQ**: MBQ 用模态级梯度加权，QIG 用 token 级量化感知积分梯度，粒度更细且与量化误差直接关联
- **vs AWQ/GPTQ**: 这些方法不考虑多模态结构，QIG 专门针对 LVLM 的异构 token 序列设计
- token 级敏感度分析的思路可以迁移到 LVLM 的剪枝和知识蒸馏中

## 评分
- 新颖性: ⭐⭐⭐⭐ 从可解释性到量化的跨领域迁移有新意
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准多设置，消融实验系统
- 写作质量: ⭐⭐⭐⭐ 动机分析和可视化做得好
- 价值: ⭐⭐⭐⭐ 即插即用的 PTQ 改进，实用价值高

