# VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service

**会议**: ACL 2025  
**arXiv**: [2506.15755](https://arxiv.org/abs/2506.15755)  
**代码**: https://github.com/wangdaha1/VLMInferSlow  
**领域**: 多模态VLM  
**关键词**: efficiency robustness, adversarial attack, VLM, black-box, inference slowdown

## 一句话总结
首次在黑盒设置下研究 VLM 的效率鲁棒性，提出 VLMInferSlow 方法，通过零阶优化搜索对抗性图像扰动，迫使 VLM 生成更长序列，将计算成本最高增加 128.47%，揭示了 VLM 在 MLaaS 部署场景下的效率安全隐患。

## 研究背景与动机

1. **领域现状**：VLM 已广泛部署为 API 服务（如 Microsoft Seeing AI、Be My Eyes），需要实时响应。NVIDIA 和 AWS 报告推理阶段占 ML 总能耗的 90% 以上。现有对抗攻击研究主要关注准确率鲁棒性。
2. **现有痛点**：少数针对 VLM 效率攻击的研究（如 NICGSlowdown、Verbose Images）都假设白盒访问（完全知道模型架构和参数），而现实中 VLM 多以 API 形式部署，白盒假设不切实际。
3. **核心矛盾**：黑盒设置下无法使用梯度信息进行优化，且零阶优化方法在目标函数存在剧烈变化时容易失效。
4. **本文要解决什么？** 在黑盒设置（仅通过 API 交互）下评估 VLM 的效率鲁棒性——能否通过不可感知的图像扰动显著增加 VLM 的推理开销？
5. **切入角度**：VLM 的自回归解码特性天然使推理效率与生成序列长度正相关。通过延长生成序列长度，可以有效增加推理开销。结合零阶优化估计梯度替代白盒梯度。
6. **核心 idea 一句话**：设计三个效率导向的对抗目标（延长序列 + 延迟 EOS + 增加 token 多样性），用零阶优化在黑盒下搜索不可感知的对抗图像扰动。

## 方法详解

### 整体框架
迭代式优化：每次迭代 (1) 计算效率导向目标函数 → (2) 零阶优化估计梯度 → (3) 梯度上升更新扰动并裁剪到 $\|\delta\| \leq \epsilon$。输入是正常图像 $\mathcal{I}$，输出是对抗图像 $\mathcal{I} + \Delta$，使 VLM 在处理对抗图像时消耗更多计算资源。

### 关键设计

1. **三合一对抗目标**:
   - **$\mathcal{L}_{len}$（延长序列）**: 直接最大化输出序列长度 $\text{Length}(\mathcal{F}(\mathcal{I}+\delta))$。虽然非可微，但适用于无导数优化。
   - **$\mathcal{L}_{eos}$（延迟终止）**: 降低每个位置输出 EOS token 的概率，并引入动态权重衰减策略，越靠后的位置权重越大：$\mathcal{L}_{eos} = -\sum_{i=1}^{N} \omega^{N-i} \text{Pr}^{\text{EOS}}(y_i | \mathcal{I}+\delta)$，$\omega=0.1$。
   - **$\mathcal{L}_{var}$（增加多样性）**: 将每个 token 位置的 top-k 概率分布向均匀分布对齐，使用 KL 散度：$\mathcal{L}_{var} = -\frac{1}{N}\sum_{i=1}^{N} D_{KL}(\tilde{\text{Pr}}(y_i | \mathcal{I}+\delta) \| \mathcal{U})$，$k=100$。
   - **最终目标**: $\mathcal{L} = \mathcal{L}_{len} + \alpha \mathcal{L}_{eos} + \beta \mathcal{L}_{var}$
   - 设计动机：单一目标不够，三个目标分别从序列长度、终止信号、token 分布三个角度综合施压。

2. **零阶梯度估计**:
   - 做什么：在无法获取梯度的黑盒设置下估计目标函数的梯度方向。
   - 核心思路：采用 Natural Evolution Strategies，在搜索分布 $\pi(z|\delta) = \mathcal{N}(\delta, \eta^2 I)$ 下采样 $2q$ 个高斯噪声扰动，估计梯度：$\hat{\nabla}_\delta J(\delta) = \frac{1}{2\eta q}\sum_{i=1}^{2q} \mu_i \mathcal{L}(\delta + \eta\mu_i)$。使用反镜像技巧（$\mu_{q+j} = -\mu_j$）降低方差。
   - 设计动机：标准零阶优化在损失面剧烈变化时不稳定；结合精细化的多目标使损失面更平滑，改善了零阶优化的搜索效果。

3. **扰动约束更新**:
   - 做什么：梯度上升更新扰动后裁剪到 $L_2$ 范数约束内。
   - 核心思路：$\delta \leftarrow \delta + \gamma \hat{\nabla}_\delta J(\delta)$，然后 $\text{Clip}(\delta, \epsilon)$，并确保 $(\mathcal{I}+\delta) \in [0,1]^n$。

### 损失函数 / 训练策略
本文不涉及模型训练，而是在推理时通过迭代优化生成对抗扰动。

## 实验关键数据

### 主实验

| 模型 | 方法 | MS-COCO I-length (%) | MS-COCO I-latency (%) | ImageNet I-length (%) | ImageNet I-latency (%) |
|------|------|:---:|:---:|:---:|:---:|
| Flamingo | Gaussian | -4.15 | -0.16 | -4.27 | -1.12 |
| Flamingo | NICGSlowdown-B | -3.54 | 0.19 | -1.14 | -0.12 |
| Flamingo | Verbose-B | -2.93 | 5.56 | -0.63 | 5.26 |
| **Flamingo** | **VLMInferSlow** | **128.47** | **105.56** | **103.44** | **78.42** |
| BLIP | Gaussian | 18.92 | 18.19 | 20.50 | 20.42 |
| **BLIP** | **VLMInferSlow** | **显著提升** | **显著提升** | **显著提升** | **显著提升** |

**关键结果**: VLMInferSlow 在黑盒设置下将 Flamingo 的计算成本最高增加 128.47%（长度）和 115.19%（能耗），远超所有黑盒基线。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 $\mathcal{L}_{len}$ | 部分有效 | 但缺少其他约束时优化方向不稳定 |
| $\mathcal{L}_{len} + \mathcal{L}_{eos}$ | 更好 | EOS 延迟显著延长序列 |
| $\mathcal{L}_{len} + \mathcal{L}_{eos} + \mathcal{L}_{var}$ | 最优 | 三目标协同提升效率攻击效果 |
| 无动态权重衰减 | 下降 | 均匀权重不如对靠后位置加权 |
| 黑盒 vs 白盒方法对比 | 接近白盒水平 | VLMInferSlow 黑盒效果可媲美白盒 |

### 关键发现
- **黑盒接近白盒效果**：VLMInferSlow 在黑盒下的攻击效果与白盒方法（需要完整模型参数）相当，说明 API 暴露的 logits 信息足以构造有效攻击。
- **扰动不可感知**：生成的对抗图像在视觉上与原图无异，通过了人类感知测试。
- **现有防御效果有限**：测试了多种防御策略，VLMInferSlow 仍能有效增加计算开销。
- **不同采样策略下都有效**：对 temperature、top-k、top-p 等采样策略都保持攻击效果。

## 亮点与洞察
- **首个黑盒 VLM 效率攻击**：填补了黑盒设置下 VLM 效率鲁棒性评估的空白。在 MLaaS 场景（OpenAI、Google Gemini 等部署模式）下更贴近真实威胁。
- **精细化多目标设计的巧妙之处**：三个目标从不同维度攻击效率瓶颈，且精细化目标使零阶优化的损失面更平滑，解决了零阶方法的固有缺陷。
- **安全启示大于攻击本身**：这项工作更重要的价值在于提醒社区重视 VLM 的效率安全问题，特别是在移动设备和 API 服务场景下的资源耗尽攻击风险。

## 局限性 / 可改进方向
- 需要 API 返回 logits 信息（$\mathcal{L}_{eos}$ 和 $\mathcal{L}_{var}$ 依赖 token 概率），部分 API 可能不提供。
- 对于已有长度截断机制的 API（如 max_tokens 限制），攻击效果可能受限。
- 零阶优化需要大量 API 查询（$2q$ 次/迭代），成本较高。
- 仅关注图像扰动，未探索文本 prompt 侧的效率攻击。

## 相关工作与启发
- **vs NICGSlowdown**: NICGSlowdown 通过延迟 EOS 攻击图像描述模型，但需要白盒访问；VLMInferSlow 在黑盒下实现了更强效果。
- **vs Verbose Images**: Verbose Images 设计了多种白盒损失函数增加 VLM 计算量；VLMInferSlow 以零阶优化替代梯度计算，在黑盒下实现了接近的效果。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个黑盒 VLM 效率攻击，问题定义有价值
- 实验充分度: ⭐⭐⭐⭐ 4 个 VLM + 2 个数据集 + 4 个基线 + 消融 + 防御评估
- 写作质量: ⭐⭐⭐⭐ 问题定位清晰，图表对比直观
- 价值: ⭐⭐⭐⭐ 对 VLM 部署安全有重要警示作用
