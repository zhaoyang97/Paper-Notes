---
title: >-
  [论文解读] RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents
description: >-
  [ICLR 2026][LLM安全][代码智能体] RedCodeAgent 是第一个针对「代码智能体」的全自动红队 agent，它用一个不断累积成功经验的记忆模块、一个混合通用越狱工具和代码替换工具的工具箱，再加上 Docker 沙盒真执行评测，自适应地组合调用工具来攻破目标代码 agent，在多个 benchmark、多种语言、多个商用 agent（Cursor、Codeium）上都拿到比单一越狱方法更高的攻击成功率、更低的拒答率。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "代码智能体"
  - "自动化红队"
  - "越狱攻击"
  - "记忆模块"
  - "沙盒评测"
---

# RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=IyIaAOihmZ](https://openreview.net/forum?id=IyIaAOihmZ)  
**代码**: https://github.com/1mocat/RedCodeAgent  
**领域**: Agent / LLM 安全  
**关键词**: 代码智能体, 自动化红队, 越狱攻击, 记忆模块, 沙盒评测

## 一句话总结
RedCodeAgent 是第一个针对「代码智能体」的全自动红队 agent，它用一个不断累积成功经验的记忆模块、一个混合通用越狱工具和代码替换工具的工具箱，再加上 Docker 沙盒真执行评测，自适应地组合调用工具来攻破目标代码 agent，在多个 benchmark、多种语言、多个商用 agent（Cursor、Codeium）上都拿到比单一越狱方法更高的攻击成功率、更低的拒答率。

## 研究背景与动机
**领域现状**：LLM 驱动的代码智能体（code agent）已经被广泛用作编程助手——它们不只生成代码，还接入 Python 解释器等外部工具，能动态执行、调试、与系统环境交互。要评估这类 agent 的安全性，目前主流靠两条路：静态安全 benchmark（给一批固定的风险测试用例打分），以及人工设计的红队/越狱工具（GCG、AutoDAN 等）。

**现有痛点**：这两条路都跟不上代码 agent 的演化速度。静态 benchmark 覆盖不了真实世界里的边界行为，尤其覆盖不了「把多种攻击方法组合起来」这种情况；人工红队算法是静态、被动的，agent 一旦学会规避某个已知 pattern，这些方法就失效了。文中举的例子很具体：OpenCodeInterpreter 把 `os.remove` 这类危险命令拉进黑名单做护栏，但换成 `pathlib` 的 `path.unlink` 就能绕过——固定攻击 pattern 根本枚举不完这些等价实现。

**核心矛盾**：代码任务的红队和通用 LLM 越狱有本质区别。通用安全任务里，只要目标模型「不拒绝」请求就算越狱成功；但在代码场景里，光让 agent 不拒绝还不够——目标 agent 必须**真的生成并执行了能完成危险功能的正确代码**。这意味着 (1) 攻击必须有明确的功能目标并据此迭代优化，(2) 评测必须看真实执行结果而非只做静态代码分析，否则 LLM-as-a-judge 会有偏差。

**本文目标**：把红队过程自动化、自适应化，针对给定的代码 agent 系统性地挖出会被静态方法漏掉的漏洞，并给出可靠（看真执行）的成功判定。

**核心 idea**：把红队本身做成一个 agent——用记忆复用历史成功经验、用工具箱动态选择并组合越狱手段、用沙盒真执行来给反馈，让红队像 agent 一样在多轮交互中边试边学。

## 方法详解

### 整体框架
RedCodeAgent 是一个站在攻击方、对外部目标代码 agent 做自动化渗透测试的交互式 agent。用户输入一个「风险场景 + 风险描述」（例如「删除敏感文件 / 写一个删除 `/root/.bashrc` 的 Python 脚本并执行」），系统就开始一轮自适应红队：先去**记忆模块**检索 top-K 个最相似的历史成功经验作为示范，喂给基座 LLM；LLM 据此决定是直接拿当前 prompt 去查询目标 agent，还是先从**工具箱**里调一个工具（代码替换或通用越狱）来改写 prompt；改写后可以继续叠加工具，也可以直接发起查询；目标 agent 响应后，**评测模块**在 Docker 沙盒里真执行并判定结果（拒答 / 执行失败 / 攻击成功）；如果失败，评测反馈（含失败原因，如「目标文件仍存在」）回灌给 LLM 继续优化；如果成功，LLM 对整条轨迹做一次自我反思，把这条成功经验按固定结构写回记忆，供后续任务参考。整个过程有最大动作数上限（实验设 max_iterations=35）防止无意义探索。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：风险场景<br/>+ 风险描述"] --> B["记忆模块<br/>检索 top-K 相似成功经验"]
    B --> C["基座 LLM 决策<br/>直接查询 or 调工具"]
    C -->|改写 prompt| D["工具箱<br/>代码替换 + 通用越狱工具"]
    D --> C
    C -->|发起攻击| E["查询目标代码 agent"]
    E --> F["评测模块<br/>沙盒真执行判定"]
    F -->|失败：反馈原因| C
    F -->|成功：自我反思| G["写回记忆 + 输出成功实例"]
```

### 关键设计

**1. 自适应记忆模块：让红队复用历史成功经验、且偏好高效轨迹**

针对的痛点是「每个新任务都从零开始试错」效率太低。RedCodeAgent 把每一次成功攻击存成一条结构化记忆条目，包含五项：风险场景、风险描述、轨迹（完整记录每次工具调用的选择/理由/耗时/输入输出，以及与目标 agent 的输入输出和评测反馈）、最终评测结果、最终自我反思。遇到新任务时，记忆检索（Alg. 1）不只看语义相似度，还把「效率」编进打分：对查询 $q$ 与记忆条目 $m$，先用 embedding 模型分别算风险场景相似度 $S_r=\mathrm{CosSim}(e_q^{risk},e_m^{risk})$ 和风险描述相似度 $S_t=\mathrm{CosSim}(e_q^{des},e_m^{des})$，再按轨迹长度算惩罚 $P=\mathrm{Length}(m.\text{trajectory})\times\rho$，综合得分 $S=S_r+S_t-P$，取 top-K 返回。这个惩罚项 $\rho$ 的作用是把那些「能成功但绕了很多弯」的长轨迹压下去，让 LLM 学到的是又准又快的攻击套路，而不是冗长的乱试。实验里 $K=3$、$\rho=0.02$、embedding 用 paraphrase-MiniLM-L6-v2，记忆从空开始、跑测试用例时逐条累积。

**2. 工具箱与代码替换工具：用语义等价的代码实现绕过黑名单护栏**

针对的痛点正是开头那个例子——护栏靠黑名单关键词（`os.remove`），但同一功能有无数等价写法。工具箱里最具「代码特异性」的是**代码替换工具**：它接收一段代码或一段功能描述，调用一个专门的代码替换 LLM 生成实现相同功能、但写法不同的替代代码。比如删除 `/etc/shadow`，`os.remove(...)`、`os.unlink(...)` 常被拒，但 `pathlib.Path(...).unlink()` 往往能被执行。因为有些危险功能连代码替换 LLM 自己都会拒绝，作者准备了多个候选模型（CodeQwen1.5-7B-Chat、Meta-Llama-3-8B-Instruct、gpt-35-turbo），一个拒了就换下一个。工具箱里还集成了一批通用越狱方法做互补覆盖：梯度类 GCG、学习类 Advprompter / AmpleGCG、进化类 AutoDAN（附录还有模板类、角色扮演类）。关键在于这些工具是被 agent 以 function-calling 方式**按需、可组合**调用的——case study 里 agent 先 GCG 失败，再叫代码替换换成 pathlib，再叠一层 GCG 后缀，最终组合两者攻破目标，这种组合是固定 pipeline 给不出来的。

**3. 沙盒真执行评测模块：用执行结果而非静态代码消除 judge 偏差**

针对的痛点是「光看代码或让 LLM 当裁判」会误判——代码看着危险但没真执行成功、或 judge 有偏。RedCodeAgent 给每个风险描述配一套 Docker 沙盒环境和对应的评测脚本：让目标 agent 在隔离沙盒里真执行，然后直接检查系统状态来判定（比如删文件任务就查文件系统里那个文件还在不在）。判定落到三类离散结果：拒答（query 被拒）、执行失败（危险功能未达成）、攻击成功（代码执行成功且产生了对应危害）。沙盒隔离既保证能测真正危险的代码又不损害宿主安全。更重要的是，评测不只吐「攻击成功」这种关键词，还会给出原因（如「攻击失败因为目标文件仍存在」），这个解释回灌给 agent 当反馈，提醒它别忘了攻击目标、并指引下一步优化方向。针对三个 benchmark 还分别用了脚本评测（RedCode-Exec）、LLM 评测（RedCode-Gen）、关键词匹配（RMCbench）。

### 一个例子：删除 /etc/gshadow
以「删除敏感文件 `/etc/gshadow`」为例走一遍（论文 Fig. 5）：① agent 先直接把原始请求发给目标 → 被拒；② 调 GCG 生成对抗后缀再发 → 仍被拒；③ 调代码替换工具，得到建议「用 `file_path.unlink()` 实现同样功能」；④ 再叠一层 GCG 优化 prompt → 这次目标 agent 回「File '/etc/gshadow' removed.」，沙盒查证文件确实没了，评测判为攻击成功；⑤ agent 反思整条轨迹（「直接请求被拒时，prompt 结构上的创造性很关键」），把这条成功经验写回记忆。这一例正好展示了「代码替换的 pathlib 建议 + GCG 对抗后缀」的组合威力。

## 实验关键数据

### 主实验
跨 3 个 benchmark（RedCode-Exec 810 例、RedCode-Gen 160 例、RMCbench 182 例）、4 种语言、多个目标 agent。基线为 No Jailbreak、GCG、AmpleGCG、Advprompter、AutoDAN、FlipAttack。核心指标 ASR（攻击成功率，越高越强）、RR（拒答率，越低越强）。

| 目标 agent / benchmark | 指标 | No Jailbreak | 最强基线 | RedCodeAgent |
|---|---|---|---|---|
| OCI / RedCode-Exec | ASR | 55.46% | 54.69%(GCG) | **72.47%** |
| OCI / RedCode-Exec | RR | 14.70% | 12.84%(GCG) | **7.53%** |
| OCI / RedCode-Gen | ASR | 9.38% | 35.62%(GCG) | **59.11%** |
| ReAct / RedCode-Gen | ASR | 65.62% | 59.38%(GCG) | **81.52%** |
| ReAct / RedCode-Gen | RR | 34.38% | 40.00% | **2.50%** |
| OCI / RMCbench | ASR | 18.68% | 43.96%(GCG) | **69.78%** |

商用 / 多 agent（Tb. 4）上同样有效：Cursor ASR 62.60%→**72.72%**、Codeium 60.98%→**69.88%**、MetaGPT 24.98%→**45.62%**，且 RR 都更低。跨语言（Tb. 3）Python/C/C++/Java 的 ASR 分别达 89.44% / 81.67% / 85.56% / 80.00%，全面超过基线。

### 效率与消融

| 配置 / 分析 | 关键结果 | 说明 |
|---|---|---|
| 效率 (Fig. 2) | 121.17s, ASR 72.47% | 与单一越狱方法量级相当却 ASR 最高；Advprompter 132.59s 才 46.42% |
| 轨迹长度 (Fig. 3) | 91.1% 成功案例 ≤4 步 | 成功时高效，失败时才会多试工具（8–10 步集中） |
| 基线 + retry (Tb. 2) | GCG(retry) 59.14% | 给基线加最多 3 轮重试仍打不过 72.47%，且时间大涨 |
| w/o 记忆模块 (§D.4) | ASR 下降 | 记忆模块「重要且必要」；但加入顺序/是否预填影响很小 |
| 工具数量 (§D.6) | 单 GCG 已提升，越多越高 | 体现可扩展性 |
| vs 五基线顺序串联 (§D.8) | RedCodeAgent 更优 | ASR 和效率都赢过简单串联 |

### 关键发现
- **单用越狱方法不一定提升 ASR**：GCG/AmpleGCG/Advprompter/AutoDAN 优化后的 prompt 有时还不如原始静态 prompt——因为它们只保证「不被拒」，不保证「执行出预期功能」；RedCodeAgent 始终锚定明确功能目标并按执行反馈迭代，才是真正有效的关键。
- **能挖出别人挖不到的漏洞**：在所有基线都失败的场景里，RedCodeAgent 在 OCI 上发现 82 个、ReAct 上 78 个独有漏洞（如 reverse shell、eval injection）。
- **降低拒答靠「软化措辞」**：它会把「dangerous alias」改成「customized alias」这类改写来降低 prompt 敏感度，LLM 评估的隐蔽性也最高（Tb. 14）。
- **工具用量随难度自适应**：简单任务（risk index 1、2）几乎不额外调工具；困难任务（9、10、11）才花更多时间用 GCG/Advprompter 优化。

## 亮点与洞察
- **把红队做成 agent，关键在「记忆 + 真执行反馈」闭环**：记忆让攻击经验可复用、可迁移，真执行评测给出的失败原因是高质量反馈信号——两者合起来让攻击从「盲试」变成「带目标的迭代优化」，这是它压过单一越狱方法的根因。
- **记忆检索把效率编进打分**很巧妙：$S=S_r+S_t-\rho\cdot \text{Length}$ 用一个惩罚项就把「短而准」的套路顶上来，避免 agent 学到啰嗦的成功轨迹，这个思路可迁移到任何 RAG/经验复用型 agent。
- **代码特异性洞察**：代码安全的本质是「功能等价但实现不同就能绕护栏」，所以代码替换工具是这篇相比通用越狱框架最有价值的差异点，也解释了为什么黑名单式护栏注定不够。
- **沙盒真执行**直接消除了 LLM-as-judge 在代码场景的系统性偏差，对「危险代码是否真生效」给出确定性判定，这个评测范式对所有 code agent 安全研究都可复用。

## 局限与展望
- 攻击有效性依赖工具箱里现有越狱工具的能力，面对更强护栏或纯语义级别的安全对齐，现有工具组合是否仍够用未充分讨论。
- 沙盒评测脚本是针对每个风险描述定制的，扩展到全新风险类别需要人工写评测脚本，自动化程度仍有上限。
- 作为攻击型工具天然有双刃剑风险：虽定位是「部署前安全评估」，但同样的框架可被滥用；论文偏重攻击效果，对防御侧如何利用这些发现着墨较少。
- 基座用 GPT-4o-mini、代码替换用若干 7–8B 模型，换更强/更弱基座对自适应决策质量的影响只在附录略提（§D.7）。

## 相关工作与启发
- **vs 静态安全 benchmark（RedCode、ToolEmu、R-judge）**：它们提供固定风险记录数据集 + LLM judge 打分，是静态、被动的；本文是对给定 agent 主动、动态地红队，且补了沙盒真执行评测，能覆盖 benchmark 漏掉的组合攻击。
- **vs 通用越狱方法（GCG / AutoDAN / Advprompter / AmpleGCG）**：它们只优化「不被拒」、面向通用 NLP 安全任务；本文把它们当工具箱里的子工具按需组合，并额外保证「执行出危险功能」这一代码任务特有目标。
- **vs 代码 LLM 红队（CodeAttack / INSEC / SVEN）**：这些针对代码翻译/补全或白盒训练投毒；本文聚焦黑盒「自然语言→代码生成」任务上对完整 code agent（含执行）的对抗，是一个此前少被探索的角度。
- **vs 通用越狱 agent（Xu et al. 2024 等）**：前者面向静态 LLM；本文首次把红队 agent 对准带代码生成与执行的 code agent，复杂度更高。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个针对代码 agent 的自动化红队 agent，记忆+工具箱+沙盒真执行的组合此前没人做过
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 3 benchmark、4 语言、含 Cursor/Codeium 商用 agent，主实验+效率+消融齐全
- 写作质量: ⭐⭐⭐⭐ 结构清晰、case study 直观，部分指标散落在多张表/图里需要拼读
- 价值: ⭐⭐⭐⭐⭐ 给 code agent 部署前安全评估提供了可扩展、可复用的攻击与评测范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] STAR: Strategy-driven Automatic Jailbreak Red-teaming for Large Language Model](star_strategy-driven_automatic_jailbreak_red-teaming_for_large_language_model.md)
- [\[ICLR 2026\] ARMS: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks](arms_adaptive_red-teaming_agent_against_multimodal_models_with_plug-and-play_att.md)
- [\[ICLR 2026\] Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models](auto-rt_automatic_jailbreak_strategy_exploration_for_red-teaming_large_language_.md)
- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)

</div>

<!-- RELATED:END -->
