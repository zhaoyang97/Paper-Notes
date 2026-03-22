<!-- 由 src/gen_stubs.py 自动生成 -->
# Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking

**会议**: CVPR2026  
**arXiv**: [2602.20330](https://arxiv.org/abs/2602.20330)  
**代码**: [github.com/UIUC-MONET/vlm-circuit-tracing](https://github.com/UIUC-MONET/vlm-circuit-tracing)  
**领域**: 多模态VLM  
**关键词**: 可解释性, 电路追踪, VLM内部机制, transcoder, 归因图, 特征操纵

## 一句话总结
提出首个面向 VLM 的电路追踪框架，通过在 Gemma-3-4B 中训练 transcoder、构建归因图、发现多模态电路，揭示了视觉-语义概念的层次化整合、视觉数学推理电路、六指幻觉的内部机制等关键洞察。

## 研究背景与动机
1. VLM（如 CLIP、LLaVA、GPT-4o）在多模态任务上表现卓越，但其内部工作机制仍是黑箱
2. 理解 VLM 内部机制对高风险应用（医学影像、自动驾驶、内容审核）至关重要
3. 现有可解释性工作主要聚焦纯文本 LLM（如电路发现、注意力分析、探测），VLM 几乎未被触及
4. VLM 引入更深层挑战：需要整合不同统计特性和语义的两种模态，发现有意义的视觉-语言对应
5. Sparse autoencoders 和 transcoders 已在 LLM 中用于分解多义表示，但从未应用于 VLM
6. VLM 的多模态推理如何在内部实现——视觉特征如何绑定到 token、跨模态推理如何协调——仍是未知

## 方法详解

### 整体框架（三大组件）

**1. VLM 中的 Transcoders**：为 Gemma-3-4B 的每层 MLP 训练独立 transcoder，将多义表示分解为可解释的单义特征。

编码器：$z(x) = \text{ReLU}(W_{enc}x + b_{enc})$，使用 TopK 稀疏化（k=48）
解码器：$\text{TC}(x) = W_{dec}z(x) + b_{dec}$

训练数据：SmoLIM2 文本（144K）+ ImageNet 图像（144K）+ Cauldron QA（72K），在 8×H100 上训练 30K 步约 60 小时。

**2. 归因图（Attribution Graph）**：追踪特征间的因果关系。归因定义为：
$$A_{s \to t} = a_s \cdot w_{s \to t}$$
其中虚拟权重 $w_{s \to t} = f_{dec}^{(s)\top} J^\blacktriangledown_{(s)\to(t)} f_{enc}^{(t)}$ 包含解码器向量、冻结 Jacobian 和编码器向量。单次 QA 任务的归因图计算约需 H100 20 分钟。

**3. 视觉/文本 token 的电路发现**：通过注意力分析解释无名多模态特征，用 SigLIP 视觉编码器的 attention rollout 可视化图像区域激活。最终由人类专家标注和简化电路。

### 关键设计

**Feature Steering**：通过修改特征激活值观察输出变化
$$h_{\ell,t} \leftarrow h_{\ell,t} + \Delta z_{\ell,t,i} \cdot d_{\ell,i}$$

**Circuit Patching**：将一个电路的特征移植到另一个结构相似的电路中验证因果性

### 训练评估指标
方差未解释分数（FVU）评估重建质量：$\text{FVU} = \frac{\text{MSE}}{\text{Var}(y)}$

## 实验关键数据

### 训练配置与重建质量

| 扩展因子 $N_{latents}$ | 死特征比例 | FVU |
|------------------------|----------|-----|
| 32 | 最高 | 较高 |
| **64** | **适中** | **最优** |
| 128 | 最低 | 略升 |

### 多模态 vs 纯文本训练

| 训练数据 | 中间层 FVU | 高层 FVU |
|---------|-----------|---------|
| 纯文本 | 较高 | 相近 |
| **文本+图像** | **显著更低** | **略低** |

### 关键发现
1. **层次化整合**：视觉和语义概念的联合特征仅在约 Layer 20 以上出现，早期层保持模态独立
2. **视觉数学电路**：对图像化算术（如渲染的 $1+2$），模型部分在视觉空间内计算——中间层出现数字 "3" 的视觉特征
3. **六指幻觉机制**：源于视觉编码器过度强调"手"语义 + 模型内部电路放大手相关特征，压制了本可正确计数的视觉电路
4. **火星-航天飞机关联**：输入火星图像时，内部视觉关联特征（"航天飞机"）被激活，反映了独立于语义的视觉联想
5. **circuit patching 验证**：抑制火星视觉特征并激活地球特征，后续所有特征和输出均变为地球相关概念

## 亮点与洞察
- 首次在 VLM 中成功进行电路追踪，填补了多模态可解释性的重要空白
- 揭示了 VLM 保持不同的视觉和语义流一直到网络深层才合并的架构特性
- 六指幻觉的机制分析提供了超越"one failure mode"的深层理解——编码器偏差+电路竞争
- Intervention 实验证明电路是因果性的且可控的

## 局限性
- 仅分析 Gemma-3-4B 一个模型，结论的普适性未经验证
- 视觉编码器注意力图有时难以解读，限制了特征标注质量
- 电路发现依赖人类专家标注，大规模自动化困难
- per-layer transcoder 无法捕获跨层超位（cross-layer superposition）
- 计算成本高——单个归因图 20 分钟，特征激活分析约 20 H100 GPU-hours

## 相关工作与启发
- 与 LLM 电路追踪（Lindsey et al., Anthropic）的扩展：从纯文本到多模态，首次处理图像 token
- 与 attention visualization / probing 的区别：电路追踪是因果性的，而非相关性的
- 启发：VLM 中离散的视觉表征空间的存在暗示了视觉和语言可能在更深层次上是分离的

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (VLM 电路追踪的开创性工作)
- 实验充分度: ⭐⭐⭐⭐ (多维度分析+因果验证，但仅 1 个模型)
- 写作质量: ⭐⭐⭐⭐⭐ (洞察深刻，案例分析引人入胜)
- 价值: ⭐⭐⭐⭐⭐ (为 VLM 可解释性奠定基础框架)
