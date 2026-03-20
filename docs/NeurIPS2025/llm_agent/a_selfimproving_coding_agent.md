# A Self-Improving Coding Agent

**会议**: NeurIPS 2025  
**arXiv**: [2504.15228](https://arxiv.org/abs/2504.15228)  
**代码**: [https://github.com/MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent) (有)  
**领域**: LLM Agent / 自动化Agent设计  
**关键词**: self-improving agent, meta-agent, coding agent, automated agent design, self-referential  

## 一句话总结
提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

## 背景与动机
LLM Agent系统（prompting策略、工具调用、多Agent协作等）的设计目前高度依赖人工，研究者需要手动尝试各种prompting方案（CoT、ToT、GoT、Self-Refine等），这种trial-and-error的方式只能探索方案空间的一小部分。ADAS（Automated Design of Agentic Systems）尝试用meta-agent来优化target-agent，但这两个Agent是分离的——meta-agent本身不会变好。一个自然的追问是：能不能让Agent自己改进自己？特别是coding agent，因为Agent本身就是代码写的，一个能编程的Agent天然具有修改自身代码的能力。

## 核心问题
如何构建一个真正的**自引用自改进系统**（self-referential self-improving system），让Agent在不需要人工干预的情况下，通过修改自身代码（而非仅仅调整prompt或单个函数），持续提升coding benchmark上的表现？核心挑战包括：(1) 如何让改进机制本身也受益于之前的改进（复合效应）；(2) 如何保证安全性和可观测性；(3) 如何平衡性能、成本和时间。

## 方法详解

### 整体框架
SICA的核心循环类似ADAS但有本质区别：
1. 维护一个**Agent档案库**（archive），记录每一轮迭代的Agent代码和benchmark结果
2. 每轮迭代：选择档案库中表现最好的Agent作为**meta-agent**（注意：这个meta-agent就是之前某轮的target-agent，没有分离）
3. meta-agent查阅档案，分析历史表现，提出改进方案并实现代码修改
4. 新Agent在benchmark集上评测，结果存入档案库
5. 循环往复

关键区别于ADAS：ADAS的meta-agent固定不变，只改target-agent的单个forward函数；SICA是完全自引用的，meta-agent本身来自档案库中的最优Agent，且可以修改整个Python代码库。

### 关键设计

1. **Utility函数设计**: 不仅考虑benchmark准确率，还综合成本和时间：$U = w_{score} \cdot p_{score} + w_{cost} \cdot (1 - \min(1, p_{cost}/\$10)) + w_{time} \cdot (1 - \min(1, p_{time}/300s))$，权重分别为0.5、0.25、0.25。超时还有额外惩罚（$\tau=0.5$）。这驱动Agent不仅提升准确率，也要降低成本和延迟。

2. **异步Overseer机制**: 一个独立LLM在并行线程中每30秒运行一次，检查Agent是否陷入循环、偏离任务或出现病态行为。Overseer可以发通知、强制取消Agent执行。这是保证长时间自主运行（单轮改进约1小时）安全性的关键。

3. **Sub-agent架构与上下文管理**: 初始Agent包含coding agent、problem-solver agent和reasoning agent三个子Agent。子Agent调用类似tool call，返回字符串值。这一设计既实现任务分解（benchmark任务完全自主运行），又管理了上下文长度——子Agent返回后其上下文自然被丢弃，仅保留返回值摘要。

4. **KV Cache友好的上下文布局**: LLM context依次为system prompt（Agent定义+工具文档）→ core prompt（问题+打开的文件+目录树）→ assistant messages（CoT推理+tool调用）。内容只追加不重写，保留KV cache以降低延迟和成本。文件编辑以diff形式追加，周期性合并。

5. **XML格式Tool Calling**: 使用无约束的XML结构化生成（而非JSON），因为XML不需要转义字符串字面量——编辑代码时避免了大量转义问题。

6. **合成Benchmark**: 除SWE-Bench和LiveCodeBench外，Agent还使用了两个自动构建的合成benchmark——文件编辑benchmark（从git commit对中构建）和符号定位benchmark（在代码库中定位符号定义），展示了系统为自身生成新评测的能力。

## 实验关键数据

| 数据集 | 指标 | 初始(Iter 0) | 最终(Iter 15) | 提升 |
|--------|------|------|----------|------|
| SWE-Bench Verified（50题子集） | 准确率 | 17% | 53% | +36pp |
| LiveCodeBench（50题） | 准确率 | 有微小提升 | — | 轻微 |
| 文件编辑（合成） | 准确率 | — | — | 显著↑ |
| 符号定位（合成） | 准确率 | — | — | 显著↑（Iter 9 AST工具后） |
| AIME 2024 + GPQA Diamond | 平均准确率 | ~76% | ~80% | 边际 |

- 主模型：Sonnet 3.5 v2，推理子Agent用o3-mini
- 15轮迭代总API成本约$7,000
- Agent发现的关键改进（按迭代顺序）：Smart Edit工具 → Code Context Summarizer → File Edit Verification → AST Symbol Locator → Hybrid Symbol Locator
- 推理任务（AIME/GPQA）改进有限——因为o3-mini单模型已很强（AIME 87%、GPQA 79%），Agent scaffolding反而可能干扰reasoning model的思维链

### 消融实验要点
- **Smart Edit工具**（基于diff和range的编辑替代全文覆写）是早期最大增益点，直接减少了token消耗和编辑时间
- **AST Symbol Locator**（Iter 9）带来跨benchmark的显著提升，利用了benchmark以Python为主的特点
- **Hybrid Symbol Locator**（ripgrep候选+AST验证）进一步提升了符号定位精度
- 推理任务中，包含粗糙推理组件（如iteration 4和6）反而**降低**了reasoning model的性能——scaffolding干扰了内建思维链
- 缓存命中率从32.5%逐步提升到~40%，说明Agent学会了优化上下文复用

## 亮点
- **真正的自引用**：消除meta/target分离，改进的Agent本身就是下一轮的改进者，改进能力可以复合
- **修改完整代码库**而非仅一个函数——Agent可以创建新工具、新子Agent、修改prompt、甚至修改上下文管理逻辑
- Agent自主发现的改进非常实用（Smart Edit、AST定位等），与人类工程师的直觉一致
- Utility函数平衡accuracy/cost/time的设计很务实，避免了只追求准确率而忽略效率
- 异步Overseer是保证长时安全自主运行的优雅方案
- 展示了"scaffolding-only"改进（不改模型权重）的上限和局限

## 局限性 / 可改进方向
- **路径依赖严重**：早期差的改进idea会影响后续所有迭代的idea质量，variance很高
- **成本高**：15轮迭代花费$7,000，且单次meta-improvement约需1小时
- **推理任务改进有限**：对已经很强的reasoning model（o3-mini），Agent scaffolding的边际收益很小甚至为负
- **5分钟超时限制**：导致初始benchmark分数偏低（不反映模型真实能力），许多早期"改进"只是加速而非提质
- **未做权重更新**：纯scaffolding改进有天花板，论文本身也承认联合更新模型权重+scaffolding是重要未来方向
- **Benchmark固定**：静态benchmark集可能很快饱和，需要self-curatable的benchmark机制
- **Novelty生成难题**：Agent难以产生真正新颖、可行的改进idea——这是open-ended learning的核心挑战

## 与相关工作的对比
- **vs ADAS**：ADAS有固定meta-agent和受限的DSL（只改forward函数）；SICA是完全自引用的，操作完整Python代码库。但SICA需要更强的底座模型（Sonnet 3.5 + o3-mini），成本更高
- **vs Gödel Agent**：Gödel Agent提供特定tool来修改Agent逻辑的小部分，不是通用coding agent，也没在coding benchmark上评测
- **vs STOP (Zelikman et al.)**：STOP是递归自改进的代码生成器但仅限算法任务（parity、3-SAT等），不是可做任意SE任务的通用Agent
- **vs AlphaEvolve**：AlphaEvolve更依赖结构化进化搜索，且聚焦科学发现和计算基础设施优化；SICA更自由形式，专注Agent系统本身的改进

## 启发与关联
- 自引用自改进的范式对Agent系统设计有深远影响：如果Agent能自主创建新工具和优化prompt，人类工程师的角色将转向定义utility函数和安全约束
- scaffolding改进有天花板这一发现很重要——暗示未来的突破需要将代码层改进与模型权重更新结合
- Utility函数设计（accuracy + cost + time）可以广泛应用于其他Agent优化场景
- 异步Overseer机制可直接复用到任何需要长时自主运行的Agent系统中

## 评分
- 新颖性: ⭐⭐⭐⭐ 真正自引用的coding agent自改进，首次在coding benchmark上验证，但核心循环与ADAS相似
- 实验充分度: ⭐⭐⭐⭐ 多benchmark、15轮迭代、推理任务对比、详细的改进日志，但SWE-Bench只用了50题子集
- 写作质量: ⭐⭐⭐⭐⭐ 论文写得非常清晰，动机、方法、实验、安全讨论层次分明，附录的agent trace极具参考价值
- 价值: ⭐⭐⭐⭐ 验证了自改进Agent的可行性，开源代码，但实际可复现性受限于高成本($7k/run)
