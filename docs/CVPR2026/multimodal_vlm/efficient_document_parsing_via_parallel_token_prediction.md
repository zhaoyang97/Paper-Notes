# Efficient Document Parsing via Parallel Token Prediction

**会议**: CVPR 2026
**arXiv**: [2603.15206](https://arxiv.org/abs/2603.15206)
**代码**: [GitHub](https://github.com/flow3rdown/PTP-OCR)
**领域**: 多模态VLM
**关键词**: 文档解析, 并行token预测, Register Token, VLM加速, OCR

## 一句话总结

提出 PTP（Parallel Token Prediction），一种模型无关的即插即用加速方法，通过在训练序列中插入可学习 register token 实现并行多 token 预测，在 OmniDocBench 上实现 1.6×-2.2× 吞吐提升且不损失精度。

## 研究背景与动机

1. **文档解析的实用需求**：文档解析需将非结构化文档转为机器可读输出，是 RAG、文档分析等应用的基石，对速度和精度均有高要求。
2. **VLM 彻底改变了文档解析**：VLM 端到端或管线式方法显著提高了解析质量，但自回归（AR）解码成为速度瓶颈。
3. **AR 解码的本质矛盾**：文档解析本质是高确定性转录任务而非开放式生成，输出由输入图像唯一确定，天然具有可并行性。
4. **现有加速方法的不足**：输出压缩、视觉 token 裁减、参数剪枝均未根本解决 AR 瓶颈。
5. **非自回归方法受限**：基于 CTC 的 NAR 模型性能有限且仅限 span 级 OCR。
6. **关键洞察**：图像可分解为多个 patch 独立识别，这种并行能力可内嵌到模型中。

## 方法详解

### 整体框架

PTP 在 VLM 的标准 NTP 训练基础上，插入可学习 register token 并设计对应的训练目标和注意力掩码，使模型获得并行解码能力。配合高质量数据生成管线。

### 关键设计

#### Register Token 设计

在训练序列的每个 token 后插入 $n$ 个 register token，所有 register 共享同一 token ID 和可学习嵌入，通过不同位置编码区分。第 $i$ 个 register token 学习预测后续第 $i+1$ 个位置的 token：

$$\hat{X}_a = (x_1, [r_2, r_3], x_2, [r_3, r_4], \ldots, x_l)$$

#### 注意力掩码设计

三条约束：(1) 常规 token 只关注前面的常规 token，与 register 隔离；(2) Register 关注所有前面的常规 token 和同组 register；(3) 不同组的 register 互相隔离。确保常规 NTP 训练完全不受 register 影响。

#### 位置编码调整

Register $r_i$ 的位置 ID = 前一个常规 token $x_{i-1}$ 的位置 + 1，依次递增。

### 损失函数

$$\mathcal{L}_{\text{PTP}} = \alpha \cdot \mathcal{L}_{\text{NTP}} + (1-\alpha) \cdot \mathcal{L}_{\text{reg}}$$

$\mathcal{L}_{\text{reg}} = -\sum_i \sum_j \log P_\theta(x_{i+j+1} | X_{a,\leq i}, r_{i+j})$

### 数据生成管线

200k 页多样化文档 → 布局分析分割子区域 → 多模型协作标注（强 VLM + 开源 VLM + 专用模型）→ 多数投票 + LLM 后处理 → CLIP 去重 + pHash 去重 → 最终 1.8M 高质量样本。

## 实验关键数据

### 主实验：OmniDocBench

| 模型类型 | 代表模型 | Overall Edit Distance↓ |
|---------|---------|------------------------|
| Pipeline | PP-StructureV3 | 0.0695 |
| 通用VLM | Gemini-2.5 Pro | 0.0734 |
| 通用VLM | GPT-4o | 0.2297 |
| PTP方法 | PTP-1 | 1.6× 加速 |
| PTP方法 | PTP-2 | 2.2× 加速 |

### 消融实验

| 配置 | 吞吐提升 | 精度影响 |
|------|---------|----------|
| PTP-0 (NTP baseline) | 1.0× | baseline |
| PTP-1 (1 register) | 1.6× | 无损/减少幻觉 |
| PTP-2 (2 registers) | 2.2× | 无损 |
| 与投机解码结合 | 82% 接受率 | - |

### 关键发现

- PTP 不仅加速还**减少**了模型幻觉，因为 register 提供了额外的预测约束
- 方法可泛化到通用视觉语言理解（VLU）任务
- 与投机解码正交且可协同，组合后达到 82% 接受率
- 估算加速比：$\text{SR} \approx ((1+n) \times L_\theta) / L'_\theta$

## 亮点与洞察

- **极致的即插即用性**：模型无关、不改架构、仅需添加 register token 和修改注意力掩码
- 训练时 register 不影响常规 token（通过掩码隔离），保证了 NTP 性能的下限
- 减少幻觉的附加效果令人惊喜——多 token 预测提供了隐式约束
- 数据管线设计全面：多源收集 + 多模型标注 + 多阶段过滤

## 局限性

- 推理时需在每步移除 register 的 KV cache，增加了实现复杂度
- Register 预测远期 token 的准确率会随距离下降
- 训练序列长度增加 $(1+n)$ 倍，训练成本上升
- 目前主要在文档解析场景验证，开放域生成效果待探索

## 相关工作与启发

- 与 DeepSeek-V3 的 MTP head 思路类似但实现不同：PTP 用 register token 而非额外预测头
- Register token 的灵感来自 ViT 中吸收高范数异常值的设计（DINOv2），但用途完全不同
- 方法与输出压缩、视觉 token 裁减等加速方法正交，可叠加使用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
