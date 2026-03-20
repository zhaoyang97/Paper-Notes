# Sharing State Between Prompts and Programs

**会议**: ICLR 2026  
**arXiv**: [2512.14805](https://arxiv.org/abs/2512.14805)  
**代码**: [https://github.com/psg-mit/nightjarpy](https://github.com/psg-mit/nightjarpy)  
**领域**: 编程语言 / LLM编程  
**关键词**: 共享程序状态, 自然语言编程, prompt-program互操作, Nightjar, 编程抽象  

## 一句话总结
提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

## 研究背景与动机

1. **领域现状**：LLM 催生了自然语言编程——用 prompt 指示模型执行任务。现有系统（LangChain、DSPy、SGLang 等）支持 prompt 与程序的互操作，但采用隔离状态设计（isolated program state）：prompt 在独立环境中执行，开发者需要手动序列化/反序列化数据以传递程序状态。

2. **现有痛点**：隔离状态设计导致大量样板代码——需要定义 schema 类、序列化函数、反序列化函数来在 prompt 和程序之间传递数据。这增加了开发复杂度，也容易引入错误。

3. **核心矛盾**：prompt 本质上需要访问程序上下文才能做出合理决策（读取变量值、修改对象状态、控制分支/循环），但现有系统将 prompt 执行与程序状态严格隔离，开发者必须手写桥接代码。

4. **本文要解决什么？**：(a) 定义共享程序状态的编程抽象；(b) 设计 natural function interface 的形式化 schema；(c) 实现 Nightjar 系统验证其可行性和效益。

5. **切入角度**：借鉴编程语言中的 effects & handlers 范式，将 prompt 对程序状态的操作形式化为效应（effects），由宿主语言的处理器（handlers）实现。

6. **核心idea一句话**：让 prompt 像函数一样直接访问程序变量作用域、堆和控制流，消除手动状态传递的开发负担。

## 方法详解

### 整体框架
Nightjar 将 prompt 作为 Python 程序中的一等代码。开发者用 `@nightjar.fn` 装饰函数，在函数体内用三引号字符串写 prompt。Prompt 中可用 `<variable>` 读取局部变量，用 `<:variable>` 写入变量，直接操作 Python 对象，并实现 break/continue 等控制流。

### 关键设计

1. **共享作用域（Shared Scopes）**:
   - 做什么：prompt 可以读取和写入 Python 变量
   - 核心思路：prompt 中的 `<graph>` 引用当前作用域中的 `graph` 变量，LLM 输出中的 `<:response>` 将值绑定到 `response` 变量。系统在 prompt 执行前捕获作用域快照，执行后更新变量
   - 设计动机：消除了手动传入/传出数据的 schema 定义和序列化代码，使 prompt 真正成为程序的一部分

2. **共享堆（Shared Heap）**:
   - 做什么：prompt 可以操作 Python 对象（修改属性、调用方法、就地更新可变对象）
   - 核心思路：LLM 不直接操作堆，而是通过 reference/dereference 效应间接操作。系统维护对象引用表，将 LLM 的操作指令转化为对 Python 对象的实际操作
   - 设计动机：让 prompt 能够修改复杂的程序数据结构（如图、列表），而不是只返回序列化的新版本

3. **共享控制流（Shared Control State）**:
   - 做什么：prompt 可以触发 break、continue 等控制流操作
   - 核心思路：prompt 通过标签（labels）引用程序中的控制流结构。LLM 输出 break 效应时，Nightjar 的 handler 在宿主 Python 程序中执行对应的 break
   - 设计动机：使 prompt 能够根据对话语义决定何时终止循环或跳过迭代，避免额外的条件判断代码

4. **Natural Function Interface Schema**:
   - 做什么：形式化 prompt 与程序的交互接口
   - 核心思路：基于 effects & handlers 范式。Effects 定义 prompt 可以执行的操作类型（读变量、写变量、引用对象、break等），handlers 定义这些操作在宿主语言中的实现
   - 设计动机：提供语言无关的规范，使共享程序状态可以在任何编程语言上实现

### 损失函数 / 训练策略
Nightjar 不涉及模型训练，是编程系统层面的贡献。核心技术挑战在于如何将 LLM 的自然语言输出映射到正确的程序操作。

## 实验关键数据

### 主实验（Nightjar vs 手动实现）

| 任务 | Nightjar 准确率 | 手动实现准确率 | 代码行减少 | 运行时开销 |
|------|---------------|-------------|----------|-----------|
| 图操作 | +4-19% | 基线 | ~40% | 0.4-4.3x |
| 数据处理 | 持平或更高 | 基线 | ~40% | 中等 |
| 控制流任务 | 更高 | 基线 | 显著 | 略高 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 完整共享状态 | 最优 | 作用域+堆+控制流 |
| 仅共享作用域 | 可用但受限 | 不能修改可变对象 |
| 隔离状态（基线） | 需要大量样板代码 | 传统方式 |

### 关键发现
- 代码量平均减少 **39.6%**，主要来自消除 schema 定义和序列化/反序列化代码
- 准确率提升 **+4-19%**：因为共享状态避免了手动序列化引入的信息丢失和格式错误
- 运行时开销 0.4-4.3 倍：主要来自引用解析和效应处理的额外通信

## 亮点与洞察
- **编程抽象层面的贡献**比具体系统更重要：shared program state 是一个新的编程范式，不仅限于 Python。natural function interface schema 是语言无关的，任何编程系统都可以实现。
- **Effects & Handlers 在 LLM 编程中的应用**非常巧妙：将 prompt 对程序状态的操作抽象为效应，由宿主语言的处理器实现，是 PL 理论与实际 LLM 系统的优雅结合。
- 揭示了一个趋势：计算越来越多地被动态地、自适应地规划和执行，LLM 使得"运行时编程"成为现实。

## 局限性 / 可改进方向
- 运行时开销（0.4-4.3x）在延迟敏感场景下可能不可接受
- LLM 对复杂程序对象的操作可能出错（幻觉写入错误值）
- 目前仅实现了 Python 版本，需要验证在其他语言上的可移植性
- 安全性问题：prompt 直接操作程序状态可能带来意外的副作用

## 相关工作与启发
- **vs LangChain/DSPy**: 这些系统采用隔离状态，需要手动定义 schema 和序列化。Nightjar 消除了这一负担。
- **vs AskIt/ANPL**: 这些系统用 LLM 生成函数来替代 prompt，部分共享状态但不支持写变量和控制流。
- **vs tool use**: 工具使用要求开发者定义自定义函数，Nightjar 的共享状态不需要开发者定义任何额外函数。

## 评分
- 新颖性: ⭐⭐⭐⭐ 共享程序状态是新的编程抽象，effects & handlers 的应用创新
- 实验充分度: ⭐⭐⭐ 任务数量有限，缺少大规模应用验证
- 写作质量: ⭐⭐⭐⭐ PL 形式化规范与实际系统结合良好
- 价值: ⭐⭐⭐⭐ 对 LLM 编程系统的设计有启发意义
