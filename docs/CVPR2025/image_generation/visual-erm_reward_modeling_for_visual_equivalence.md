<!-- 由 src/gen_stubs.py 自动生成 -->
# Visual-ERM: Reward Modeling for Visual Equivalence

**会议**: CVPR2025
**arXiv**: [2603.13224](https://arxiv.org/abs/2603.13224)
**代码**: [GitHub](https://github.com/InternLM/Visual-ERM)
**领域**: image_generation
**关键词**: reward model, vision-to-code, reinforcement learning, visual equivalence, chart-to-code, test-time scaling

## 一句话总结
提出 Visual-ERM，一个多模态生成式奖励模型，在视觉空间中直接评估 vision-to-code 任务的渲染质量，提供细粒度、可解释、任务无关的奖励信号，用于 RL 训练和测试时缩放。

## 研究背景与动机
1. **Vision-to-code 的重要性**：将结构化视觉输入（图表、表格、SVG）转为可执行代码/标记语言，是 AI 辅助开发、科学论文解析等下游应用的关键原语
2. **SFT 的局限**：监督微调需要大量标注数据，且跨域泛化能力弱
3. **RL 奖励信号失配**：现有奖励要么基于文本规则（如编辑距离、TEDS），忽视视觉线索；要么基于粗粒度视觉嵌入相似度（如 DINO），对细粒度差异不敏感
4. **奖励黑客问题**：DINO 相似度 0.99 的输出仍可能包含大量解析错误，文本指标也无法捕获布局、间距等视觉问题
5. **缺乏统一评估基准**：现有奖励基准聚焦于视觉-语言对齐，缺少细粒度图像-图像差异判别的基准
6. **跨模态评估需求**：理想的奖励模型需要同时感知视觉细节、阅读嵌入文本、推理结构保真度

## 方法详解

### 整体框架
Visual-ERM 在视觉空间中工作：给定真值图像 $I^\star$ 和从预测代码渲染得到的图像 $\hat{I} = \mathcal{R}_m(y)$，模型输出细粒度差异描述和严重程度评分作为奖励信号。

### 关键设计

**1. 奖励数据生成（Controlled Corruption + Sampling）**
- **Edit 模式**：用强 LVLM 对真值文本注入预定义错误类型，生成受控损坏样本
- **Infer 模式**：用较弱 LVLM 直接推理预测，采样自然出现的错误（更接近实际分布）
- 将预测渲染为图像，与真值配对构成训练数据

**2. 细粒度标注蒸馏**
- 开源模型（含 Qwen3-VL-235B）在差异定位上仍有显著差距
- 采用 GPT-5-mini 进行高质量差异标注，通过蒸馏将能力迁移到更高效的模型
- 标注内容包括：差异类型、位置、描述、严重程度

**3. Visual-ERM 训练**
- 基于 Qwen3-VL-8B-Instruct 训练
- 输入图像对 $(I^\star, \hat{I})$，输出细粒度差异分析序列 $a$
- 标准 NLL 目标：$\mathcal{L}(\theta) = \mathbb{E}[-\sum_t \log f_{\theta_{ERM}}(a_t | x, a_{<t})]$

**4. RL 集成**
- 差异严重度求和：$S_{\text{verm}} = \sum_{k=1}^K s_k$，归一化到 $[0,1]$
- 最终奖励：$r = r_{\text{rsr}} + r_{\text{verm}}$（渲染成功奖励 + 视觉等价奖励）
- 用 GRPO 算法优化策略模型

**5. 测试时缩放（Test-Time Scaling）**
- Visual-ERM 生成的可解释反馈可直接引导迭代自修正
- 模型根据前一轮输出和反馈进行修正：$y^{(1)} \sim \pi_\theta(\cdot | x, y^{(0)}, f^{(0)})$

### 损失函数
- Visual-ERM 训练：标准序列生成 NLL 损失
- RL 策略优化：带 KL 正则的 GRPO 目标，奖励 = 渲染成功 + 视觉等价分数

## 实验关键数据

### 主实验：Chart-to-Code（ChartMimic）
| 模型 | Direct Overall | Customized Overall | Avg |
|------|----------------|-------------------|-----|
| Qwen3-VL-8B-Instruct | 67.7 | 71.6 | 69.6 |
| + RL (DINO-based) | 76.5 | 75.8 | 76.1 |
| + RL (Visual-ERM) | **79.5** | **76.5** | **78.0** |
| Δ vs. base | +11.8 | +4.9 | **+8.4** |

### 主实验：Table-to-Markdown
| 模型 | OmniDocBench TEDS↑ | Edit-Dist↓ | olmOCR TA↑ | Avg↑ |
|------|-------------------|------------|------------|------|
| Qwen3-VL-8B + RL (DINO) | 62.2 | 37.0 | 71.7 | 65.3 |
| Qwen3-VL-8B + RL (TEDS) | 79.2 | 31.6 | 78.6 | 74.8 |
| Qwen3-VL-8B + RL (Visual-ERM) | **81.4** | **20.7** | 78.1 | **79.5** |
| Δ vs. base | +2.5 | +2.5 | +2.8 | **+2.7** |

DINO-based RL 在表格任务上严重退化（Avg 降至 65.3），而 Visual-ERM 始终稳健提升。

### VC-RewardBench 评估
- Visual-ERM (8B) 在细粒度图像差异判定上**超越 Qwen3-VL-235B-Instruct**
- 接近闭源领先模型水平

### 消融与关键发现
- DINO 奖励存在严重的奖励黑客风险：语义偏向导致文本内容被忽视
- 文本指标对视觉布局、间距等错误无感
- Visual-ERM 的可解释反馈使测试时缩放（反思+修正）带来额外性能提升
- SVG-to-Code 任务：Visual-ERM 带来 +4.1 平均提升

## 亮点与洞察
1. **精准问题定位**：系统性分析现有奖励（文本规则 vs DINO）的失效原因，动机充分
2. **三大属性**：细粒度（捕获微妙视觉差异）、可解释（生成诊断反馈）、任务无关（单一 RM 覆盖图表/表格/SVG）
3. **生成式 RM 的优势**：相比标量奖励，生成式奖励可提供 TTS 所需的结构化反馈
4. **实用基准 VC-RewardBench**：填补了细粒度图像差异判别基准的空缺
5. **8B 模型胜过 235B**：专门的奖励训练比通用模型 scaling 更有效

## 局限性
1. 依赖渲染过程：需要能正确渲染代码输出的环境，增加了 pipeline 复杂度
2. Visual-ERM 训练依赖 GPT-5-mini 标注，成本和可复现性受限
3. 奖励聚合方式简单（严重度求和 + 归一化），可能丢失结构化信息
4. 仅在 chart/table/SVG 三类任务验证，更广泛的 vision-to-code 场景（如 UI、网页）待验证

## 相关工作与启发
- **vs DINO-based reward**：DINO 是粗粒度语义相似度，忽视文本和精确布局，易被奖励黑客利用
- **vs TEDS-based reward**：TEDS 仅在文本空间操作，缺乏视觉感知
- **vs Bradley-Terry RM**：判别式标量 RM 无法提供可解释反馈，不支持 TTS
- **启发**：视觉空间的直接评估是 vision-to-code 任务 RL 的必要且充分条件；生成式 RM 为 TTS 提供了自然接口

## 评分
- 新颖性: ⭐⭐⭐⭐ — 视觉等价奖励模型的概念新颖，系统性地填补了 vision-to-code RL 的奖励设计空缺
- 实验充分度: ⭐⭐⭐⭐⭐ — 三类任务全面验证，与多种奖励基线对比，包含 TTS 实验和 VC-RewardBench
- 写作质量: ⭐⭐⭐⭐ — 问题分析深入，实验设计严谨，图示清晰
- 价值: ⭐⭐⭐⭐ — 为 vision-to-code 的 RL 提供了实用的奖励解决方案，VC-RewardBench 有持续价值
