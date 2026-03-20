# Unveiling Privacy Risks in LLM Agent Memory

**会议**: ACL 2025  
**arXiv**: [2502.13172](https://arxiv.org/abs/2502.13172)  
**代码**: [https://github.com/wangbo9719/MEXTRA](https://github.com/wangbo9719/MEXTRA)  
**领域**: LLM Agent  
**关键词**: Agent隐私, 记忆模块攻击, 黑盒攻击, 隐私提取, Agent安全  

## 一句话总结
本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

## 研究背景与动机
1. **领域现状**：LLM Agent 广泛应用于医疗、网购等隐私敏感场景，记忆模块存储用户交互历史作为 few-shot 示范
2. **现有痛点**：
   - 已有工作研究了 RAG 中外部数据的泄露，但 Agent 记忆模块作为新的隐私信息源几乎未被研究
   - 记忆中存储的用户查询可能包含高度敏感信息（如患者病情、购物偏好）
   - 现有 RAG 攻击 prompt（如"请重复所有上下文"）对 Agent 无效，因为 Agent 工作流更复杂
3. **核心矛盾**：Agent 需要记忆模块提升性能，但记忆中的私人信息面临被提取的风险
4. **本文要解决什么**：(1) Agent 记忆信息能否被提取？(2) 哪些因素影响泄露程度？(3) 什么攻击策略更有效？
5. **切入角度**：设计针对 Agent 工作流特点的两部分攻击 prompt（定位+对齐）
6. **核心idea一句话**：通过 locator+aligner 的攻击 prompt 设计让 Agent 按自身工作流输出记忆中的隐私信息

## 方法详解

### 整体框架
黑盒攻击设定 → 攻击 prompt 由两部分组成：$\tilde{q} = \tilde{q}^{loc} \| \tilde{q}^{align}$ → locator 定位记忆中的用户查询 → aligner 指定输出格式使其与 Agent 工作流一致 → 自动化生成多样 prompt 以覆盖更多记忆条目。

### 关键设计
1. **攻击 Prompt 设计 (Locator + Aligner)**:
   - **Locator** ($\tilde{q}^{loc}$)：明确指定要提取检索到的用户查询（而非其他上下文描述），并优先输出它们而非完成原始任务。如"I lost previous example queries"
   - **Aligner** ($\tilde{q}^{align}$)：指定输出格式使其符合 Agent 工作流。如 Web Agent 的"please enter them in the search box"，Code Agent 的"print them as output"
   - 设计动机：Agent 工作流复杂（代码执行/网页操作），不能简单要求文本输出，必须让 Agent 在自身工作流框架内返回隐私信息

2. **自动化多样 Prompt 生成**:
   - 做什么：用 GPT-4 自动生成 n 个多样的攻击 prompt，最大化提取不同的记忆条目
   - **Basic level**：只知道 Agent 的应用领域，通过改变措辞和表达方式生成多样 prompt
   - **Advanced level**：已推断出相似度函数后，针对性优化——若用编辑距离则变化长度，若用语义相似度则添加不同领域关键词
   - 设计动机：单次攻击只能提取 top-k 条记录，需要多样化 prompt 覆盖更多记忆

3. **威胁模型**:
   - 攻击者只能通过输入查询与 Agent 交互（黑盒）
   - 目标：提取记忆中存储的用户查询 $q_i$
   - 两个知识级别：Basic（知道应用领域）和 Advanced（推断出检索函数）

## 实验关键数据

### 主实验（30个攻击prompt，记忆200条）
| Agent | 提取数(EN) | 检索数(RN) | 效率(EE) | 完全提取率(CER) |
|-------|-----------|-----------|---------|---------------|
| EHRAgent (医疗) | 50 | 55 | 0.42 | 0.83 |
| RAP (网购) | 26 | 27 | 0.29 | 0.87 |

### 消融实验
| 配置 | EHRAgent EN | RAP EN | 说明 |
|------|-----------|--------|------|
| MEXTRA (完整) | 50 | 26 | 最佳 |
| w/o aligner | 36 | 6 | 对齐器对 RAP 影响巨大 |
| w/o req | 39 | 25 | 去掉生成要求 |
| w/o demos | 29 | 8 | 去掉示例 |

### 关键发现
- **Agent 记忆高度脆弱**：30 个攻击 prompt 就能从 200 条记忆中提取 25-50%，CER 超 83%
- 编辑距离检索函数比语义相似度更容易被攻击（覆盖更多记忆）
- 记忆越大泄露越多（但增速递减）
- 增加攻击次数持续提升提取量，尤其在 Advanced 知识下
- Aligner 对工作流限制严格的 Agent（如 Web Agent）极其关键

## 亮点与洞察
- 首次系统揭示 LLM Agent 记忆模块的隐私风险，填补了重要安全研究空白
- Locator+Aligner 的攻击设计思路巧妙，利用了 Agent 工作流的特性
- 发现不同检索函数对隐私泄露的影响差异很大——设计 Agent 时应考虑这个安全因素
- 为 Agent 开发者敲响警钟：记忆模块需要专门的隐私保护机制

## 局限性 / 可改进方向
- 仅测试了两种 Agent、一种 LLM（GPT-4o），泛化性有待验证
- 静态记忆设定（记忆不更新）简化了真实场景
- 未提出具体的防御方案
- 攻击依赖精心设计的 prompt，如果 Agent 有输入过滤可能失效

## 相关工作与启发
- **vs RAG 隐私攻击**: RAG 攻击只需让 LLM 输出文本，Agent 攻击需要适配复杂工作流
- **vs Prompt Injection**: Prompt injection 目标是改变模型行为，MEXTRA 目标是提取记忆内容

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 Agent 记忆隐私，攻击设计有创意
- 实验充分度: ⭐⭐⭐⭐ 两种Agent+多因素分析+消融+不同知识级别
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，公式规范
- 价值: ⭐⭐⭐⭐⭐ 安全领域重要发现，对Agent设计有直接的安全指导意义
