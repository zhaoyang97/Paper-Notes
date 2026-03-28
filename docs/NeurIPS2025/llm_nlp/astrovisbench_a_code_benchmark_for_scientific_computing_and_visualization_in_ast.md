# AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy

**会议**: NeurIPS 2025  
**arXiv**: [2505.20538](https://arxiv.org/abs/2505.20538)  
**代码**: https://astrovisbench.github.io  
**领域**: LLM 代码生成 / 天文学  
**关键词**: 科学可视化, 天文代码生成, 领域特定基准, VLM评估, Jupyter Notebook

## 一句话总结
AstroVisBench 构建了首个评估 LLM 天文科学计算和可视化能力的代码基准——从 110 个 Jupyter Notebook 提取 864 个任务（处理+可视化），设计双重评估管线（执行式变量检查 + VLM-as-Judge 可视化评分，与专家 Spearman ρ=0.822），评测 8 个 SOTA 模型后发现 Gemini 2.5 Pro 最佳但无错误率仅 15.7%，FileNotFoundError 占 43% 错误。

## 研究背景与动机

1. **领域现状**：LLM 代码生成基准（SWE-bench, BigCodeBench）聚焦通用编程，不评估科学可视化的正确性。天文学研究工作流涉及专业 API（astropy, specutils, photutils）和复杂可视化（色-光图、光变曲线、全天投影）。
2. **现有痛点**：(a) 天文 API 高度专业化（38 个库，26 个天文专用）——LLM 可能从未在训练中充分接触；(b) 可视化评估难以自动化——不仅要"能跑"还要"图正确+好看"；(c) 查询的欠指定性——天文学家的需求常常是模糊的（"画一个色-光图"但不说具体参数）。
3. **核心矛盾**：天文可视化需要深层领域知识（如星等方向应逆序、轴标签需特定格式），但 LLM 的天文知识深度未知。
4. **本文要解决什么？** 构建评估 LLM 端到端天文研究工作流（数据处理→可视化）的标准化基准。
5. **切入角度**：从真实天文学家的 Jupyter Notebook 中提取任务（保证真实性），设计处理任务的执行式评估（变量检查）和可视化任务的 VLM 评估（Claude 3.5 Sonnet）。
6. **核心 idea 一句话**：110 个真实天文 Notebook → 864 个处理+可视化任务 + 执行式+VLM 双重评估 = 首个天文科学可视化代码基准。

## 方法详解

### 整体框架
110 个 Notebook（Astro Data Lab + STScI）→ 依赖追踪提取可视化 cell → 拆分为处理 $t_{process}$ 和可视化 $t_{visualize}$ → GPT-4o 生成自然语言查询 → **双重评估**: 处理任务比较执行变量值（VI score），可视化任务用 Claude 3.5 评判（No Error/Minor/Major）

覆盖 38 个库（26 个天文专用），6 个天文子领域（光谱学/测光学/图像处理/时间序列/宇宙学/仿真建模）。

### 关键设计

1. **任务提取与查询生成**:
   - 做什么：从 Notebook 系统地提取可评估的代码任务
   - 核心思路：追踪可视化 cell 的上游依赖，拆分为处理和可视化两阶段。GPT-4o 生成反映典型天文学家工作流的自然语言查询（不泄露专家知识）。欠指定处理：模糊查询中的常数映射到 ground truth
   - 设计动机：直接用 Notebook cell 作评估不公平——需要给模型合理的自然语言输入

2. **双重评估管线**:
   - 做什么：分别评估数据处理准确性和可视化质量
   - 核心思路：**处理**: 变量检查分数 $VI = |\mathbf{V}_M \cap \mathcal{V}_G| / |\mathcal{V}_G|$——直接比较执行后的变量值。**可视化**: Claude 3.5 Sonnet 根据参考图像评为 No Error(1)/Minor(2)/Major(3)，与 5 位天文专家的 Spearman ρ=0.822
   - 设计动机：处理可以用精确匹配，但可视化的"正确性"涉及美学/可读性/领域惯例——需要 VLM 判断

3. **错误分类学**:
   - 做什么：细粒度分析 LLM 代码生成的失败模式
   - 核心思路：执行错误（FileNotFoundError 43%, QueryClientError, ValueError/TypeError）+ 可视化错误（领域惯例违反、轴缩放不当、可读性差）
   - 设计动机：了解失败模式才能改进——FileNotFoundError 说明 LLM 幻觉路径名

### 损失函数 / 训练策略
- 评估基准，无训练
- 8 个模型：Gemini 2.5 Pro, Claude 3.7/Opus 4, GPT-o3-mini/4o, QwQ, Qwen-2.5, Llama-4

## 实验关键数据

### 主实验

| 指标 | 最佳模型 | 值 | 次佳 |
|------|---------|-----|------|
| 处理崩溃率 | Gemini 2.5 Pro | **30.8%** | Claude Sonnet 50.9% |
| VI Score | o3-mini | **0.694** | Claude Opus 0.644 |
| 可视化崩溃率 | Gemini 2.5 Pro | **20.1%** | o3-mini 30.3% |
| 无错误率 | Gemini 2.5 Pro | **15.7%** | Claude Sonnet 9.5% |
| 严重错误率 | o3-mini | 29.6% | Gemini 28.5% |

### 错误分析

| 错误类型 | 占比 | 说明 |
|----------|------|------|
| FileNotFoundError | **43%** | 幻觉文件路径 |
| QueryClientError | — | ADQL 查询失败 |
| ValueError/TypeError | — | 专业 API 误用 |
| 领域惯例违反 | 常见 | 如星等方向错误、轴标签不规范 |

### 关键发现
- 最佳模型也仅 15.7% 无错误率——天文科学可视化对 LLM 仍极具挑战
- FileNotFoundError 占 43%——LLM 大量幻觉文件路径和数据位置
- 处理 vs 可视化：处理任务更依赖 API 知识（VI 差异大），可视化更依赖领域惯例
- VLM-as-Judge（Claude 3.5）与人类专家相关性高（ρ=0.822）——可自动化评估
- 所有模型在专业天文库（如 wfc3tools, lightkurve）上比通用库差得多——训练覆盖不足

## 亮点与洞察
- **首个科学可视化代码基准**：填补了 LLM 评估在科学计算领域的重大空白
- **VLM-as-Judge 可行**：ρ=0.822 的高相关性使大规模自动评估成为可能
- **FileNotFoundError 43%**暴露了 LLM 代码生成的核心弱点——环境感知（文件系统、数据路径）严重不足

## 局限性 / 可改进方向
- 使用 LLM 自动预处理 Notebook 可能引入幻觉噪声——虽然专家验证了子集但全量未检查
- VLM 判断仍可能与专家不一致——特别在领域惯例（如星等方向）的细微点上
- 处理评估无法 pickle 生成器/lambda/OS 资源——限制了某些复杂任务的自动评估
- 8 个模型的快照评估——模型快速更新可能使结果过时
- 仅涵盖天文学——扩展到化学/物理/生物等其他科学领域需要类似工作
- 查询的欠指定性处理（映射常数到 ground truth）可能过于宽容

## 相关工作与启发
- **vs SWE-bench**: 通用代码修复，不评估科学可视化
- **vs BigCodeBench**: 通用代码生成，不涉及领域专业性
- **vs ML-Bench**: 评估 ML 工作流，不涉及天文科学
- **启发**: 每个科学领域都需要类似的领域特定代码基准——AstroVisBench 提供了建设此类基准的方法论模板

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个科学可视化代码基准，填补重大空白
- 实验充分度: ⭐⭐⭐⭐ 8 个模型 + 864 任务 + VLM 验证 + 错误分类
- 写作质量: ⭐⭐⭐⭐ 任务设计和评估流程描述清晰，错误分类学有价值
- 价值: ⭐⭐⭐⭐⭐ 推动 LLM 在科学计算领域的评测标准化，为其他科学领域提供模板