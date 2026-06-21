---
title: >-
  [论文解读] When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery
description: >-
  [ICML 2026][AI安全][自主科学发现] 本文提出 **Cartograph**——一个挂在自主"AI 科学家"循环里的验证层,它用同一套"未解子空间(unresolved subspace)"对象同时回答三件事:选哪个实验最能消歧(select)、什么时候算问题解决了(resolve)、以及——这是最关键的——当模型库本身结构性错误时该**拒绝**继续给出任何结论(refuse),并能在后续残差暴露失配时**撤回**早期已下的判定。
tags:
  - "ICML 2026"
  - "AI安全"
  - "自主科学发现"
  - "实验设计(BOED)"
  - "模型判别"
  - "拒绝(refuse)"
  - "治理"
---

# When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery

**会议**: ICML 2026  
**arXiv**: [2606.07576](https://arxiv.org/abs/2606.07576)  
**代码**: 待确认  
**领域**: AI安全 / 自主科学发现 / 实验设计  
**关键词**: 自主科学发现, 实验设计(BOED), 模型判别, 拒绝(refuse), 治理

## 一句话总结
本文提出 **Cartograph**——一个挂在自主"AI 科学家"循环里的验证层,它用同一套"未解子空间(unresolved subspace)"对象同时回答三件事:选哪个实验最能消歧(select)、什么时候算问题解决了(resolve)、以及——这是最关键的——当模型库本身结构性错误时该**拒绝**继续给出任何结论(refuse),并能在后续残差暴露失配时**撤回**早期已下的判定。

## 研究背景与动机

**领域现状**:如今 AI 系统已经进入闭环科学发现:LLM 充当规划器提出候选实验,自动化实验室执行,统计/神经模块解读数据。配套已有蛋白结构预测、规模化材料发现、符号方程发现等端到端能力。

**现有痛点**:这些栈里没有一个会在"模型库或假设空间本身结构性不充分"时发出一个**可验证的拒绝信号**。瓶颈早已不是"提不出实验"——LLM 能生成的候选远多于实验室能跑的——而是要决定哪个实验真正有信息量、当前机制问题何时算答完、以及什么时候系统应该**彻底停止下结论**,因为它正在搜索的模型库根本就是错的。

**核心矛盾**:现代贝叶斯实验设计(BOED)把 select 做得很透,经典模型判别准则(Box–Hill 等)做了受限版的 select 和 resolve,但**两条线都没把 refuse 当成一等输出**:BOED 假设先验支撑里就包含真相,模型判别假设至少有一个对手是对的。对临床药代、材料合成、毒理学这类高风险自主发现,这个缺口是致命的。

**本文目标**:把 AI 科学家的"验证与转向层"形式化为三个相互关联的决策——Select(哪个候选实验最能减少当前未解的科学歧义)、Resolve(歧义何时小到可以宣布机制问题已答)、Refuse(何时该停止在当前库里识别任何模型,因为库本身不充分)。

**切入角度**:作者区分了对科学库的两种"访问模型"——**符号访问**(直接读系数向量,recovery 是一个 coverage 覆盖性质)和**行为访问**(只能跑实验看数值输出,recovery 是一个 rank 秩性质)。当前 AI 科学家几乎都处于后者:查模拟器、实验机器人、工具端点,只看到数值而非符号方程。

**核心 idea**:用累积"分歧矩阵"的奇异值分解界定一个"未解子空间" $U_\tau$,select/resolve 都从这同一个对象导出;而 refuse 则是另挂一个基于残差的守卫,使系统能在序列循环内部既选下一个实验、又能审计式地"停下并上报"。

## 方法详解

### 整体框架
Cartograph 是一个跑在自主发现循环里的验证层。给定一个模型库 $\mathcal{M}=\{M_1,\dots,M_L\}$(它们共享一组机制基 $\Phi$、但保留不同子集)和一份候选实验菜单 $\mathcal{E}$,系统在行为访问下把每个候选实验线性化成一个"分歧设计矩阵" $H_e$,逐轮做三件事:用未解子空间打分选实验(select)→检查歧义是否消尽(resolve)→用残差守卫判断库是否结构失配(refuse/撤回)。整个循环只有三个**物理可解释**的超参数 $(\tau,\delta,\mu_{\min})$,每轮主要成本是一次截断 SVD,CPU 毫秒级,无需蒙特卡洛积分。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["模型库 + 候选实验菜单<br/>(行为访问)"] --> B["未解子空间 U_τ<br/>分歧矩阵 SVD 取小奇异值方向"]
    B -->|dim(U_τ)=0| R["Resolve：歧义消尽<br/>宣布机制问题已答"]
    B -->|仍有未解方向| C["Select：A-最优打分<br/>选最能消歧的实验 e*"]
    C --> D["执行 e*，把 H_e* 拼进 H_cur"]
    D --> E["Refuse 残差守卫<br/>归一化残差 ρ + BIC 间隔 μ"]
    E -->|ρ>δ| F["拒绝/撤回：标记库结构失配"]
    E -->|ρ≤δ 且 μ≥μ_min| G["暂定识别最佳模型"]
    F --> B
    G --> B
```

### 关键设计

**1. 未解子空间 $U_\tau$:把"还没搞清楚什么"变成一个可计算的几何对象**

整个框架的痛点是:在行为访问下,系统看不到系数,只能从实验数值里推断到底哪些"有争议的机制坐标" $a_C^\star$ 还没被现有实验区分开。作者把至今累积的分歧矩阵 $H_{\mathrm{cur}}$ 做 SVD,取右奇异向量里**奇异值小于阈值 $\tau$** 的那些,张成未解子空间 $U_\tau=\operatorname{span}\{v_j:\sigma_j\le\tau\}$(精确情形 $\tau=0$ 时就等于 $\ker(H_{\mathrm{cur}})$)。直观上 $U_\tau$ 就是"争议系数空间里,目前实验几乎没带来信息的那块"。对候选实验 $e$,先把各库成员的预测线性化得到雅可比 $J_{\ell,e}$,对每个模型对 $(i,j)$ 算分歧块 $D_{ij,e}=J_{i,e}-J_{j,e}$,堆叠成 $H_e$(式 3),于是 $H_e U_\tau$ 直接度量"这个实验对尚未消歧的方向作用有多强"。这一步把抽象的"科学不确定性"落成了一个秩/子空间问题,理论上(Thm 4.2)证明了行为访问下 $a_C^\star$ 可唯一恢复 **当且仅当** $H$ 列满秩,误差界为 $\|\hat a_\tau-a_C^\star\|_2\le\eta/\tau+\|P_{U_\tau}a_C^\star\|_2$。

**2. A-最优 Select 打分:从"挑分歧最大"升级到"挑最能缩小后验方差"**

最朴素的打分是各向同性的未解投影 $\operatorname{score}_{\mathrm{cart}}(e)=\|H_e U_\tau\|_F^2$(式 4)。但论文证明(Prop 4.8)这一项在各向同性噪声下恰好等于未解子空间上的 Fisher 信息迹——它只是一阶 A-最优。真正的默认采集规则是**精确 A-最优**:先构造带噪信息矩阵 $G_e=U_\tau^\top H_e^\top\Sigma_e^{-1}H_e U_\tau$(式 5),再用当前未解后验协方差 $\Lambda_{\mathrm{cur}}$ 算

$$\operatorname{score}_{\mathrm{A}}(e)=\operatorname{tr}(\Lambda_{\mathrm{cur}})-\operatorname{tr}\big((\Lambda_{\mathrm{cur}}^{-1}+G_e)^{-1}\big),$$

即"做完这个实验后未解后验方差能降多少"(式 6)。当 $\dim(U_\tau)>1$、后验协方差各向异性时,投影打分(raw)和 A-最优会显著拉开——这正是后面级联实验里性能全靠 A-最优升级、而非投影本身的根因。作者还给出了它与闭式 EIG(Prop 4.10)和 Box–Hill(Prop 4.11)的局部线性-高斯桥:弱信息极限下 EIG $\approx\frac12\operatorname{tr}(\Lambda_{\mathrm{cur}}G_e)$ 与投影一阶对齐,Box–Hill 在共享各向同性噪声下退化为"分歧幅度"目标——说明 Cartograph 不是随手拼的启发式。

**3. 基于残差的 Refuse 守卫:让 resolve 和"真的对吗"解耦,并允许撤回**

resolve 只能证明"库内相对歧义已闭合",**不能**证明最佳拟合的库成员是对的。所以 refuse 是另挂在同一序列循环上的残差守卫,而不是从 $U_\tau$ 单独导出的量。它用两个物理可解释的诊断:归一化残差

$$\rho=\frac{\min_{\ell}\|y_{\mathrm{obs}}-f_{m_\ell}(\hat\theta_\ell)\|_2}{\|\phi(y_{\mathrm{obs}})\|_2},\qquad \mu=\mathrm{BIC}(m_{(2)})-\mathrm{BIC}(m_{(1)}),$$

其中 $\phi(\cdot)$ 是物理有意义的汇总特征(如 $C_{\max}$、终端斜率、对数线性 RMSE),$\mu$ 是排名前二库成员的 BIC 间隔(式 7、8)。**只有当 $\rho\le\delta$ 且 $\mu\ge\mu_{\min}$ 才宣布识别**。关键在于 $\rho$ 每一步都被监控,所以一个**暂定识别可以被撤回**:系统可以在前几轮宣布某模型,等后续轮次残差暴露结构失配再收回这个判断。作者实证这正是面对库外机制时的主导行为——这也是全文最有特色的发现("revocation 撤回",而非普适的 select 增益)。

### 损失函数 / 训练策略
本方法**无任何神经网络训练**、CPU-only。三个超参数按固定 warm-start 协议标定后在该 benchmark 族内冻结:$\tau$ 取 $H_{\mathrm{cur}}$ 奇异谱的"肘点"(PK 中固定 $\tau=10^{-3}\sigma_{\max}$);$\delta$ 标到 warm-start 内库残差的 95 分位(Refuse 窗口落在 $[0.20,0.25]$);$\mu_{\min}$ 按标准 BIC 尺度的"positive 证据"阈值(PK 中得 $\mu_{\min}=2.0$)。后验协方差 $\Lambda_{\mathrm{cur}}$ 用经验贝叶斯从各向同性先验序列更新,噪声 $\Sigma_e=\sigma_e^2 I$ 由 warm-start 残差方差估。每轮复杂度独立于后验样本数,这是它比蒙特卡洛 EIG 便宜数量级的原因。

## 实验关键数据

实验围绕三大职责(select / resolve / refuse)展开,跨 5 个测试台:符号 Duffing 振子、可变维结构非线性级联、药代(PK)模型库、公开 EPA 真实时间序列、以及对已发表 A-Lab 自主材料系统的回溯审计。

### 主实验:结构级联(Select 头条)
在可把未解维度从 $d=2$ 调到 $d=16$ 的级联 ODE 上,每个维度跑 144 试验(6 真相 × 24 噪声种子)。Cartograph-A(精确 A-最优)对 raw 投影的胜负如下:

| 维度 $d$ | Raw 命中率 | Cartograph-A 命中率 | EIG 命中 | Raw regret | Cartograph-A regret | A vs Raw(W/T/L) | $p$ 值 |
|---|---|---|---|---|---|---|---|
| 2 | 0.44 | 0.44 | 0.44 | 0.052 | 0.052 | 0 / 144 / 0 | n.s. |
| 4 | 0.00 | 0.09 | 0.10 | 0.312 | 0.418 | 73 / 0 / 71 | 0.46 |
| 8 | 0.02 | **0.65** | 0.63 | 19.94 | **0.010** | 129 / 0 / 15 | $<10^{-21}$ |
| 16 | 0.07 | **0.72** | 0.70 | 2.832 | **0.014** | 120 / 0 / 24 | $<10^{-16}$ |

$d=8$ 时 A-最优选中 oracle 隐藏最优实验的比例 65% vs raw/分歧的 2%,regret 从 19.94 暴跌到 0.01。Cartograph-A 在每个维度都把闭式 EIG 追到 2 个百分点以内,却只需每轮一次 SVD、无需蒙特卡洛——这是"闭式 EIG 廉价代理"的实证。

### 诚实的边界与撤回结果
论文刻意不只报喜:

| 设置 | $n$ | 结果 |
|---|---|---|
| PK 药代(低维 $k/d\approx1/2$) | 7 | raw vs 分歧 1W/6T/0L,$p=0.5$(不显著) |
| EPA 真实 PK 时间序列 | 8 | raw vs 分歧 1W/7T/0L(可行性检查) |
| A-Lab 已确认正例 | 36 | 32 通过 / 4 标记(flag) |
| A-Lab 后修正为 inconclusive 的正例 | 4 | **4 全部 flag,0 通过** |

PK 的近似平局**被理论精确预测**(Thm 4.5:$k/d=1/2$ 时分歧启发式已捞走一半有用信号)。最有特色的是 Refuse:对 3 个**库外**药代机制,框架先早期自信、再随残差暴露失配而**撤回**识别,而 1 个扰动过的库内对照始终保持识别。在 A-Lab 审计中,残差守卫把全部 4 个事后被人工复核标为 inconclusive 的正例都 flag 出来,而仅用 $R_{wp}$ 的残差一个都没 flag 到。

### 关键发现
- **性能来自 A-最优升级而非投影本身**:raw Cartograph 和分歧启发式在级联上几乎无法区分,差距全在 A-最优;这被 Prop 4.7/4.8 共同预测($\dim(U_\tau)>1$ 时投影只是一阶 A-最优)。
- **理论诚实地解释了负结果**:维度越高分歧越大(Thm 4.6:$d\to\infty$ 时投影选择器与分歧选择器不一致概率 $\to 1-1/n$),低维则必然近似平局——作者没把 PK 包装成 select 胜利。
- **撤回(revocation)才是治理 AI 科学家真正需要的**:在同一个挑下一个实验的循环里给出可审计的"停下并上报"信号。

## 亮点与洞察
- **把"AI 科学家何时该停"形式化为 select/resolve/refuse 三个解耦决策**,且 select 与 resolve 共用同一个 $U_\tau$ 对象——这种"同一几何对象既驱动行动又证明完成"的设计很优雅,$\dim(U_\tau)=0$ 直接当 resolve 的 drop-in 信号。
- **refuse 被提为一等输出**,补上了 BOED(假设真相在先验里)和模型判别(假设至少一个对手对)都不敢碰的缺口;残差守卫可撤回早期判定,这对高风险自主发现的治理价值远大于"再多挑准几个实验"。
- **符号访问 vs 行为访问的 coverage/rank 二分**很有迁移价值:任何"agent 只能查工具数值、读不到内部结构"的场景(工具调用 agent、黑盒模型审计)都能借这套"什么时候可恢复"的刻画。
- **全程 CPU、毫秒级、三个物理可解释超参**,落地门槛极低,可直接 drop-in 到 LLM 规划的 AI 科学家作为验证层(附录给了 LLM-in-the-loop 范例)。

## 局限与展望
- 作者承认:BOED 桥是**局部线性-高斯**的,不主张全局最优、非线性等价或完整非线性 BOED 归约;一旦 $\|\Lambda_{\mathrm{cur}}^{1/2}G_e\Lambda_{\mathrm{cur}}^{1/2}\|$ 不再小,EIG 一阶近似就失效。
- select 的普适增益其实有限:低维场景(如 PK)对分歧启发式只是近似平局,真正的卖点是 refuse/撤回而非选择增益,读者不应误以为它在所有维度都碾压基线。
- A-Lab 审计样本极小(4 个 inconclusive + 36 confirmed),EPA 只有 8/96 条序列拟合稳定;这些是"可审计 pass/flag 日志"的可行性证据,不是对 A-Lab 的重新裁决,统计功效有限。
- 残差守卫依赖人工选定的物理特征 $\phi(\cdot)$ 和 BIC,$\delta/\mu_{\min}$ 的标定迁移到新领域是否稳健仍待验证。

## 相关工作与启发
- **vs 现代 BOED(EIG / Foster 等)**: 他们把 select 做得很透但假设先验支撑含真相,不输出 refuse;本文证明自己在局部线性-高斯下是闭式 EIG 的廉价一阶代理(每轮一次 SVD vs 蒙特卡洛采样),并补上 refuse。
- **vs 经典模型判别(Box–Hill / T-optimality)**: 它们假设至少一个库成员是对的、只做受限 select+resolve;本文证明 Box–Hill 在各向同性噪声下退化为"分歧幅度",并把 refuse 提为一等输出。
- **vs 端到端自动科学(AlphaFold / GNoME / 符号回归)**: 这些栈产出能力强但**不发可验证拒绝信号**;本文专门补这块治理层,定位为挂在任意自主发现循环上的验证模块。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 refuse/撤回提为自主发现的一等决策,填了 BOED 与模型判别都不碰的治理缺口。
- 实验充分度: ⭐⭐⭐⭐ 5 个测试台 + 真实 A-Lab 审计且诚实报负结果,但 refuse 的关键样本量偏小。
- 写作质量: ⭐⭐⭐⭐⭐ "Where the paper lands honestly" 这种自我设限的诚实写法少见且有说服力,理论-实验对照清晰。
- 价值: ⭐⭐⭐⭐⭐ 给"AI 科学家何时该停"提供了可审计、可落地、几乎零成本的验证层,治理意义突出。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models](position_stop_chasing_the_c-index_when_evaluating_survival_analysis_models.md)
- [\[ICML 2026\] Regret-Based Federated Causal Discovery with Unknown Interventions](regret-based_federated_causal_discovery_with_unknown_interventions.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[CVPR 2026\] DSO: Direct Steering Optimization for Bias Mitigation](../../CVPR2026/ai_safety/dso_direct_steering_optimization_for_bias_mitigation.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)

</div>

<!-- RELATED:END -->
