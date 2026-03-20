# ProBench: Benchmarking GUI Agents with Accurate Process Information

**会议**: AAAI 2026  
**arXiv**: [2511.09157](https://arxiv.org/abs/2511.09157)  
**代码**: 无（基于 adbutils）  
**领域**: LLM Agent / GUI Agent / Benchmark  
**关键词**: GUI Agent评估, 过程信息, 移动端Benchmark, Process Provider, 中英双语应用  

## 一句话总结
提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

## 研究背景与动机

1. **领域现状**：GUI Agent benchmark（AndroidWorld、AndroidLab 等）已能在真实设备上评估 Agent 执行 GUI 操作任务，但几乎所有评估都只检查"最终屏幕状态"来判断任务是否完成。
2. **现有痛点**：仅看最终状态会产生"虚假成功"——例如"买最便宜的无线鼠标"任务，如果 Agent 没有排序就随便选了一个，虽然最终屏幕显示购买确认也会判为成功。中间关键步骤（如"按价格排序"）的信息在最终页面上根本不可见。
3. **核心矛盾**：GUI 任务本质是多步链式操作，不是所有关键信息都展示在最后几个页面上。少数尝试引入过程评估的工作（SPA-BENCH、A3）要么需要手工标注中间状态（不可扩展），要么依赖 LLM 分解（不够准确）。
4. **本文要解决什么**：如何自动、准确地捕获操作过程信息，使 GUI Agent 评估既考虑最终结果又考虑关键中间步骤？
5. **切入角度**：设计 Process Provider 自动提供过程信息，两种可选方案——(a) 解析页面层级结构 获取操作描述；(b) 用 MLLM 比较操作前后截图识别操作变化。
6. **核心 idea 一句话**：区分 State-related Task（仅看最终状态）和 Process-related Task（需要检查关键中间操作），用 Process Provider 自动提供精确过程信息。

## 方法详解

### 整体框架
ProBench 包含三个模块：(1) Task Curation（任务构建）；(2) Dynamic Environment（动态执行环境）；(3) Evaluation Pipeline（评估管线含 Process Provider）。

### 关键设计

1. **两类任务划分**:
   - **State-related Task**：所有必要信息在最终截图上可见（如"查看支付宝余额"），仅检查最终状态即可
   - **Process-related Task**：需要特定中间操作但最终状态无法完全反映（如"找评分最高的寿司店并查看完整菜单"——需要排序+筛选+选择，这些操作在最终页面不可见）
   - 设计动机：真实世界大量任务需要正确的操作过程，仅看结果不够

2. **Process Provider — 两个可选组件**:
   - **Structure Description Converter**：每次 click 后解析 a11y tree（无障碍树），定位最小可点击节点，提取其 text/content_desc/resource_id 属性，生成人可读的操作描述
   - **MLLM-based Summarizer**：将操作前后两张截图拼接+标注点击坐标，让 MLLM 比较差异并生成操作摘要（如"点击了 Airbnb 首页的搜索框"）
   - 设计动机：Structure Description Converter 快速准确但依赖 a11y tree 质量；MLLM Summarizer 更灵活但需要 MLLM 推理。两者互补，用户可选

3. **评估准确性验证**:
   - State-related Task：Evaluator 准确率 96.0%
   - Process-related Task + Structure Description Converter：89.7%
   - Process-related Task + MLLM Summarizer：94.1%

### Benchmark 规模
34 个主流应用（14 英文 + 20 中文），200+ 任务，覆盖媒体/新闻/社交/购物/生活等场景。每个任务最多 15 步交互。

## 实验关键数据

### 主实验

| 模型 | State-related | Process-related | 总平均 |
|------|-------------|----------------|-------|
| **Gemini 2.5 Pro** | **45.6** | **27.9** | **40.1** |
| Qwen2.5-VL-72B | 40.9 | 27.9 | 36.9 |
| Qwen2.5-VL-32B | 18.8 | 11.8 | 16.6 |
| Qwen2.5-VL-7B | 6.7 | 1.5 | 5.1 |
| UI-TARS-1.5-7B | 11.4 | 2.9 | 8.8 |
| GPT-4o | 0.0 | 0.0 | 0.0 |
| Claude 4 Sonnet | 0.0 | 0.0 | 0.0 |
| UI-R1-E-3B | 0.0 | 0.0 | 0.0 |

### 错误分析

| 模型 | 未完成任务比例 | 其中早停比例 |
|------|-------------|----------|
| Gemini 2.5 Pro | 90.0% | 49.6% |
| Qwen2.5-VL-72B | 71.5% | 50.0% |
| Qwen2.5-VL-7B | 93.7% | 63.7% |

### 关键发现
- **最强模型也不到 50%**：Gemini 2.5 Pro 仅 40.1%，说明真实在线环境的 GUI 操作对模型仍极具挑战
- **Process-related 比 State-related 难很多**：所有模型在 Process 任务上都显著降低（Gemini: 45.6% → 27.9%），证明过程评估的必要性
- **GPT-4o 和 Claude 4 惨败**：因为 grounding 能力极差——无法准确定位 GUI 元素坐标，连第一步都过不去
- **社交和生活类 App 最难**：信息刷新频繁、布局复杂、多广告弹窗，与用户日常需求恰恰相反
- **GUI 专用模型泛化能力有限**：UI-TARS-1.5-7B 在英文任务上超过同尺寸通用模型，但在中文任务上远落后，说明特定数据微调的局限性
- **重复操作死循环**是普遍问题：Agent 无法识别操作已完成（如已经点击了搜索框但不知道），导致反复点击同一位置
- **任务规划过于简单**：Agent 倾向于将整个复杂指令直接输入搜索框，而非分步执行

## 亮点与洞察
- **State vs Process 的任务划分**是核心贡献——首次系统地区分了两类 GUI 任务，并设计了自动化过程信息采集方案
- **三大错误模式总结**非常有价值：(a) Grounding 不足（定位 GUI 元素失败）；(b) 历史操作感知差（重复操作+死循环）；(c) 任务规划过于简化（把复杂任务等同于搜索）——这些问题指明了 GUI Agent 研究的未来方向
- **中文 App 的包含**填补了现有 benchmark 的重要空白——大部分 benchmark 只有英文 App
- **真实在线环境**（而非模拟/离线 App）更能反映实际使用场景中的挑战（网络延迟、内容动态变化、反自动化机制）

## 局限性 / 可改进方向
- 评估是二值的（成功/失败），缺少对"部分完成"的度量
- 200+ 任务规模相对有限，且高度依赖中国市场 App（微信、支付宝等）
- Process Provider 的准确性（89.7%-94.1%）仍有提升空间
- 未提供模型改进建议的实验验证（如加入操作历史记忆是否有效）
- 缺少与 SPA-BENCH、A3 等同类 benchmark 的直接结果对比

## 相关工作与启发
- **vs AndroidWorld**：AndroidWorld 只用终态评估+F-Droid 开源 App，ProBench 增加了过程评估+真实在线主流 App
- **vs SPA-BENCH**：SPA-BENCH 手动分解任务步骤限制了可扩展性，ProBench 用 Process Provider 自动化
- **vs A3**：A3 用 LLM 分解任务判断完成度但准确性受限，ProBench 用 a11y tree 解析+MLLM 截图对比更可靠
- **对 GUI Agent 开发的启示**：(a) 模型需要更好的 grounding 训练数据；(b) 操作历史的注意力机制亟待改进（防止死循环）；(c) 任务规划能力比 grounding 能力更难提升

## 评分
- 新颖性: ⭐⭐⭐⭐ State/Process 划分和 Process Provider 设计新颖，但整体是 benchmark 工作
- 实验充分度: ⭐⭐⭐⭐ 9个模型 + 3类错误分析 + 应用类别分析 + 评估管线验证，但缺少直接改进实验
- 写作质量: ⭐⭐⭐⭐⭐ 错误案例分析生动直观，问题定义清晰，附录包含所有任务列表和 prompt
- 价值: ⭐⭐⭐⭐⭐ 揭示了"最强模型<50%成功率"的现实，三大错误模式总结对社区有重要指导意义
