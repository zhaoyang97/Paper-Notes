# An Epistemic Perspective on Agent Awareness

**会议**: AAAI 2026  
**arXiv**: [2511.05977v1](https://arxiv.org/abs/2511.05977v1)  
**代码**: 无  
**领域**: AI Safety / 形式化验证  
**关键词**: 认知逻辑, agent awareness, de re/de dicto, 2D语义学, 完备性证明  

## 一句话总结

本文首次将 agent awareness（智能体感知/意识）视为一种知识形式，区分了 de re（关于物理对象的）和 de dicto（关于概念/描述的）两种感知模态，并基于 2D 语义学提出了一个可靠且完备的逻辑系统来刻画这两种模态与标准"事实知识"模态之间的相互作用。

## 背景与动机

随着人工智能体越来越多地参与影响人类生活的重要决策，正确的决策往往取决于对其他智能体存在的感知（awareness）。例如：
- **战争机器人**必须在感知到平民存在时最小化伤亡
- **自动驾驶汽车**必须在感知到让行标志处有来车时停车
- **医疗AI**在感知到有人生病时必须提供帮助
- **价值对齐的机器人**在感知到有人被冒犯时必须道歉

然而，"awareness"是一个模糊的术语，现有文献将其视为一个独立概念。剑桥词典将其定义为"knowledge that something exists"（知道某物存在的知识），这暗示了一种认知论（epistemic）解释。本文正是基于这一认知论视角，给出了 awareness 的形式化账户。

## 核心问题

1. 如何形式化地区分两种不同形式的 agent awareness？
2. 如何建立一个逻辑系统来推理这两种 awareness 及其与标准知识的关系？
3. 该逻辑系统是否可靠且完备？

## 方法详解

### 整体框架

论文构建了一个基于**自我中心逻辑**（egocentric logic）和 **2D 语义学**的认知逻辑系统，包含三个模态算子：

- **K φ**（"knows φ about herself"）：标准的自我知识模态，表示智能体知道关于自身的属性 φ
- **R φ**（"de re aware"）：de re 感知模态，表示智能体作为物理对象感知到某个具有属性 φ 的智能体
- **D φ**（"de dicto aware"）：de dicto 感知模态，表示智能体在概念层面感知到某个具有属性 φ 的智能体

### 关键设计

**1. De Re vs De Dicto 的区分（核心创新）**

通过一个运行示例清晰说明：Ann 在新加坡博物馆外看到一辆车（实际是便衣警车），同时收到 WeRide 自动驾驶车已到的短信但看不到车。

- **De re awareness**（R φ）：Ann 看到了警车（作为物理对象），但不知道它是警车。智能体 b 在当前世界具有属性 φ，且 b 存在于所有 a 无法区分的世界中。
- **De dicto awareness**（D φ）：Ann 知道附近有一辆 WeRide（作为概念），但不知道具体是哪辆车。在每个 a 无法区分的世界中，至少存在一个具有属性 φ 的智能体。

**2. 认知模型（Epistemic Model）**

形式化定义为五元组 (W, A, P, ∼, π)：
- W：可能世界集合
- A：智能体集合  
- P ⊆ A × W：存在关系（哪个智能体出现在哪个世界）
- ∼ₐ：智能体 a 的不可区分等价关系
- π(p)：命题变元的赋值

**3. 满足关系的三元定义**

采用三元满足关系 w, a ⊩ φ（世界 w、智能体 a、公式 φ），这是 2D 语义学的核心，允许同时捕捉"世界的属性"和"智能体的属性"。

**4. 公理系统（8条公理 + 4条推理规则）**

公理：
1. **Truth**：K φ → φ（知道的必为真）
2. **Negative Introspection**：¬K φ → K ¬K φ（不知道则知道自己不知道）
3. **Distributivity**：K(φ→ψ) → (Kφ→Kψ)
4. **Self-Awareness**：φ → R φ 和 Kφ → Dφ（自知）
5. **Introspection of Awareness**：Dφ → KDφ（de dicto 感知可内省）
6. **Unawareness of Falsehood**：¬R⊥ 和 ¬D⊥（不可能感知到假命题）
7. **Disjunctivity**：R(φ∨ψ) → Rφ ∨ Rψ（de re 的析取性）
8. **General Awareness**：D(Rφ ∨ Dφ) → Dφ（一般感知公理）

推理规则：Modus Ponens、Necessitation（φ ⊢ Kφ）、两条 Monotonicity 规则（针对 D 和 R）。

### 损失函数 / 训练策略

本文是纯理论/形式化验证工作，不涉及损失函数或训练策略。核心技术是**完备性证明**，使用了改进的 "matrix" 技术：

- **Frame 构造**：定义了 frame 作为部分构建的模型，包含显式的 awareness 关系 ↝
- **λ-assured 集合**：引入 λ-assured 的概念处理模型构建中的"幽灵间谍"现象——只有绝对不可检测的智能体才与数据集一致
- **完备 frame**：通过逐步扩展有限 frame（添加新世界/新智能体）满足五类完备性要求
- **典范模型**：基于完备 frame 构建典范模型，证明 Truth Lemma（φ ∈ X^a_w ⟺ w, a ⊩ φ）

## 实验关键数据

本文为纯理论工作，主要结果为两个定理：

| 定理 | 内容 | 意义 |
|------|------|------|
| Theorem 1（Soundness） | 若 ⊢φ，则对所有认知模型的世界 w 和智能体 a，w,a ⊩ φ | 公理系统不会推出错误结论 |
| Theorem 2（Strong Completeness） | 若 X ⊬ φ，则存在认知模型使 X 中所有公式为真但 φ 为假 | 公理系统足以推出所有语义有效公式 |

### 消融实验要点

作为理论工作，论文通过以下方式验证了各公理的必要性：

- **Self-Awareness 公理**的合理性来源于模型设计：每个智能体必然存在于其出现的所有世界中
- **Introspection of Awareness** 仅对 D（de dicto）成立，对 R（de re）一般不成立——这是一个重要的不对称性
- **Disjunctivity** 仅对 R 成立（由语义的存在量词结构保证），对 D 不成立
- **General Awareness 公理**连接了 R 和 D，名称源自 A φ = R φ ∨ D φ 的"一般感知"缩写

## 亮点

1. **概念创新**：首次将 awareness 从独立概念转变为知识的子类型，这一视角在哲学上更优雅且在实际应用中更有操作性
2. **De re/de dicto 的精确形式化**：用无量词的模态逻辑（而非一阶认知逻辑）捕捉了本质上需要量词的概念区分
3. **运行示例设计精妙**：通过 Ann、WeRide 和警车的场景，将抽象的逻辑概念变得直观
4. **λ-assured 集合的引入**：巧妙处理了 frame 构建中"添加新世界导致 awareness 丢失"的技术困难
5. **完备性证明的 matrix 技术创新**：在已有技术基础上增加了 awareness 关系和行标签，解决了 2D 语义学中世界与智能体的"解耦"问题

## 局限性 / 可改进方向

1. **缺乏计算复杂度分析**：论文未讨论模态满足问题或模型检测的复杂度
2. **静态逻辑**：未考虑动态更新（如信息获取/遗忘导致 awareness 变化的动态逻辑扩展）
3. **跨世界身份假设**：假设了跨世界身份（transworld identity）的存在，但这在语言哲学中本身是有争议的话题
4. **单一智能体视角**：虽然模型中有多个智能体，但模态 K、R、D 都是关于"当前智能体"的属性，未直接刻画多智能体交互推理
5. **缺乏应用验证**：未展示该逻辑系统在实际 AI 系统（如自动驾驶决策）中的应用或 model checking 实现
6. **与概率/不确定性的结合**：现实中 awareness 往往是渐进式的，而非二值逻辑能完整捕捉

## 与相关工作的对比

| 工作 | 关注点 | 与本文区别 |
|------|--------|-----------|
| Fagin & Halpern (1987) | 概念性 awareness（awareness of concepts） | 本文关注 agent awareness（对其他智能体的感知） |
| Board & Chung (2021, 2022) | 基于对象的 unawareness | 不区分 de re/de dicto |
| Epstein, Naumov & Tao (2023) | De re/de dicto "know who" | 使用量词，无法表达 awareness |
| Epistemic Logic with Assignments (Wang & Seligman 2018) | 通用的带赋值认知逻辑 | 更通用但不针对 awareness |
| Jiang & Naumov (2025) | 数据匿名化中的 de re/de dicto | 关注数据集属性推断，非 awareness |
| Naumov & Tao (2023) | "Telling apart" 模态的完备性 | 无 awareness 模态，本文基于其 matrix 技术但做了重要改进 |

本文的独特贡献在于：(1) 首次提出无量词的 awareness 模态 R 和 D；(2) 在完备性证明中引入 awareness 关系和 λ-assured 条件。

## 启发与关联

1. **对 AI Safety 的启示**：为自动驾驶等系统的"感知-决策"链提供了形式化验证框架——可以精确定义"系统应该在感知到什么时做什么"
2. **与 multi-agent systems 结合**：可扩展为多智能体协作/博弈中的 awareness 推理，例如"我知道你知道我在这里"的高阶 awareness
3. **与 LLM agent 的关联**：当前 LLM-based agent 的 awareness 机制（如 tool use、环境感知）缺乏形式化保证，本文的逻辑框架可为此提供理论基础
4. **model checking 工具开发**：基于该公理系统开发自动验证工具，检验 AI 系统的 awareness 属性是否满足安全规范

## 评分

- **新颖性**: ★★★★☆ — 将 awareness 作为知识处理的视角新颖，de re/de dicto 在 awareness 中的形式化是原创贡献
- **理论深度**: ★★★★★ — 完整的可靠性和强完备性证明，技术含量高
- **实用性**: ★★☆☆☆ — 纯理论工作，距离实际应用尚有距离
- **表达清晰度**: ★★★★☆ — 运行示例有效地辅助理解，但证明部分技术性较强
- **综合评分**: ★★★★☆
