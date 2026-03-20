# EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths

**会议**: NeurIPS 2025  
**arXiv**: [2512.03571](https://arxiv.org/abs/2512.03571)  
**代码**: 待确认  
**领域**: LLM Agent  
**关键词**: Agent框架, 推理时搜索, 非确定性编程, Beam Search, 程序执行路径

## 一句话总结
提出 Probabilistic Angelic Nondeterminism (PAN) 编程模型及 EnCompass Python 框架，将 agent 的核心工作流逻辑与推理时搜索策略解耦，程序员只需在 LLM 调用处加 `branchpoint()` 标记，即可用几行参数切换 best-of-N、beam search、tree search 等策略，代码修改量减少 3-6x。

## 研究背景与动机

1. **领域现状**：推理时计算扩展（inference-time compute scaling）已成为提升 LLM agent 性能的关键手段，包括 best-of-N 采样、refinement (Reflexion)、tree search (ToT, LATS) 等策略。

2. **现有痛点**：
   - 程序员通常将推理时策略**硬编码到 agent 工作流中**，搜索逻辑和业务逻辑深度耦合
   - 切换搜索策略（如从 best-of-N 切换到 beam search）需要大量结构性代码重构
   - 不同 agent 对相同搜索策略需要各自实现，无法复用
   - 复杂的搜索策略因实现难度大而被放弃，导致错过更好的 scaling law

3. **核心矛盾**：agent 工作流（做什么）与推理时策略（怎么搜索）的耦合既降低了可读性，又限制了策略探索空间。

4. **切入角度**：从概率编程的"模型与推理分离"范式获得灵感——概率编程将模型定义和推理算法分离，类似地，agent 编程也应将工作流和搜索策略分离。

5. **核心idea一句话**：推理时策略本质是对非确定性程序执行路径的搜索，用 `branchpoint()` 标记不确定点，让框架自动构建搜索树。

## 方法详解

### 整体框架
EnCompass 是一个 Python 框架，实现 PAN 编程模型。程序员用 `@encompass.compile` 装饰器标记 agent 函数，在 LLM 调用等不可靠操作前插入 `branchpoint()`，在验证点调用 `record_score()`。装饰器在运行时将函数编译为搜索空间对象，然后通过 `.search(algo, **config)` 调用不同搜索算法。

### 关键设计

1. **PAN 编程模型**:
   - 做什么：将程序的不确定执行建模为马尔可夫链，`branchpoint()` 是标记位置，程序状态 = (位置, 变量值)
   - 核心思路：Angelic nondeterminism（天使非确定性）——程序员写代码时假设不可靠操作总是产生好输出，运行时由搜索负责找到真正好的执行路径
   - 与经典搜索的区别：不能枚举所有子节点，只能随机采样——通过指定 branching factor 来适配经典图搜索算法

2. **统一推理时策略**:
   - **Best-of-N → 深度1搜索树**：开头加一个 `branchpoint()`，采样 N 次取最高分
   - **Beam Search → 多层搜索树**：每步前加 `branchpoint()`，beam width 和 branching factor 控制搜索广度
   - **Refinement → 带记忆的回溯**：用 `NoCopy` 共享反馈列表，不同执行路径共享历史尝试信息
   - **Self-Consistency → 组评估**：`record_score()` 支持组评估函数（如 majority vote）
   - 关键insight：Global BoN（beam width=N, branching=1）和 Local BoN（beam width=1, branching=N）是 beam search 的两个极端，beam search 在两者间插值

3. **EnCompass 编译器**:
   - 做什么：将 Python 函数编译为可被搜索算法操控的搜索空间对象
   - 核心思路：`@encompass.compile` 装饰器将函数转为状态机，`branchpoint()` 成为状态转移点
   - vs 手写状态机：EnCompass 代码修改量比手写 plain Python 少 3-6x，且不破坏代码可读性

## 实验关键数据

### 主实验：代码仓库翻译 (Java→Python)

| 搜索策略 | ps0 性能 | 说明 |
|---------|---------|------|
| 无搜索 (baseline) | ~50% | 温度0.0，单次执行 |
| Global BoN | ~65% | N=16 |
| Local BoN (coarse) | ~68% | N=16, 文件级 |
| Beam (coarse) + Beam (fine) | **~78%** | 文件级+方法级 beam search |

### 代码修改量对比

| Case Study | 无 EnCompass (新增行/词) | 有 EnCompass (新增行/词) | 节省 |
|-----------|----------------------|---------------------|------|
| 代码翻译 | +423行/+2735词 | +75行/+514词 | 5.3x |
| Hypothesis Search | +21行/+120词 | +8行/+27词 | 4.4x |
| Reflexion | +27行/+181词 | +9行/+32词 | 5.7x |

### 关键发现
- Beam search 的 scaling 明显优于简单 BoN——性能与 log(cost) 线性关系中，beam search 斜率最大
- 最佳策略恰恰是实现最复杂的那个（细粒度 beam search），说明框架降低实现门槛的价值
- 策略切换只需改 2 行参数，而手写需要重构整个状态机
- 在 ps1-ps4（更大仓库）上，beam search 持续优于 BoN

## 亮点与洞察
- **"关注点分离"降低搜索策略探索成本**：最核心的贡献不是新算法，而是让程序员以极低成本尝试各种搜索策略。这类似于 PyTorch 对深度学习的意义——降低实验门槛释放创新空间
- **统一视角**：将 BoN、beam search、refinement、self-consistency 统一为"搜索执行路径树"的不同策略，概念上非常优雅
- **概率编程的 agent 版本**：PAN 之于 agent 编程，如同 Stan/PyMC 之于概率推理——分离模型和推理

## 局限性 / 可改进方向
- 仅适用于 "program-in-control" 风格 agent，不适用于 LLM 自主决策工具调用的场景（如 SWE-Bench、WebArena）
- 案例研究规模较小（MIT OCW 作业仓库），未在大规模工业级 agent 上验证
- 搜索空间随 branchpoint 数指数增长，需要好的 pruning/scoring 策略
- 评估指标（self-validation %）可能不完全可靠
- 缺少与 LangGraph 等现有 agent 框架的集成示例

## 相关工作与启发
- **vs LangGraph**：LangGraph 将工作流建模为状态机，但搜索仍需手动实现；EnCompass 让搜索策略成为一等公民
- **vs DSPy**：DSPy 自动化 prompt 工程，EnCompass 自动化搜索策略——两者正交可互补
- **vs ToT/LATS**：它们是具体的搜索算法，EnCompass 是让这些算法可以轻松应用到任意 agent 的框架

## 评分
- 新颖性: ⭐⭐⭐⭐ PAN模型和"关注点分离"的思路很有价值
- 实验充分度: ⭐⭐⭐ 案例研究有说服力但规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 概念清晰，代码示例丰富，易于理解
- 价值: ⭐⭐⭐⭐ 对 agent 开发实践有直接影响
