---
title: >-
  [论文解读] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test
description: >-
  [ICLR 2026][LLM安全][黑盒审计] 针对 API 厂商可能偷偷把宣称的模型换成量化版/微调版/越狱版的问题，本文提出**基于秩的均匀性检验（Rank-based Uniformity Test, RUT）**：每个 prompt 只查一次目标 API、在本地参考模型上多采样，把 API 输出的 log-rank 打分映射成"在参考分布里的百分位排名"，若两模型一致则这些排名应服从均匀分布，再用 Cramér–von Mises 检验偏差——做到了高检测力、每个 prompt 仅 1 次 API 调用、且查询长得像普通用户流量从而难被对抗性厂商识别绕过。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "黑盒审计"
  - "模型等价检验"
  - "基于秩的均匀性"
  - "Cramér–von Mises"
  - "量化检测"
---

# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PSIe9mmF7a](https://openreview.net/forum?id=PSIe9mmF7a)  
**代码**: https://github.com/xyzhu123/RUT (有)  
**领域**: LLM 安全 / 模型审计  
**关键词**: 黑盒审计, 模型等价检验, 基于秩的均匀性, Cramér–von Mises, 量化检测

## 一句话总结
针对 API 厂商可能偷偷把宣称的模型换成量化版/微调版/越狱版的问题，本文提出**基于秩的均匀性检验（Rank-based Uniformity Test, RUT）**：每个 prompt 只查一次目标 API、在本地参考模型上多采样，把 API 输出的 log-rank 打分映射成"在参考分布里的百分位排名"，若两模型一致则这些排名应服从均匀分布，再用 Cramér–von Mises 检验偏差——做到了高检测力、每个 prompt 仅 1 次 API 调用、且查询长得像普通用户流量从而难被对抗性厂商识别绕过。

## 研究背景与动机

**领域现状**：API 已成为访问大模型的主要入口，但用户面对的是黑盒——拿不到权重，多数情况下连输出 logits 都没有，只能看到文本（最多 top-5 token 的 log 概率）。出于降本和压低首 token 延迟（TTFT）的压力，厂商有动机偷偷部署更小或量化后的变体。

**现有痛点**：这类替换对用户完全不透明，却会损害性能、引入安全风险；更严重的情况是厂商可能挂上越狱系统提示、做了有害微调，甚至无意中接入了被错误配置的组件。于是产生了 **LLM API 审计** 这一需求：核查实际部署的模型是否如其所宣称。已有方法各有短板（论文 Table 1）：训练文本分类器（Sun et al.）需要海量 API 查询、身份提问（Huang et al.）抓不到尺寸/版本/量化这类细微差异、MMD（Gao et al.）和基准对比都依赖**特制的查询分布**——而特制分布会被狡猾厂商识别出"这是来审计的"，从而把这些可疑 prompt 重新路由回真模型，或借助 prompt caching 绕过。

**核心矛盾**：一个好的审计检验要同时满足三条——**准确**（detection power 高）、**查询高效**（少花 API 钱才有人愿意常审计）、**对抗鲁棒**（既要避免被厂商识别出审计意图、又要在概率性替换攻击下保持检测力）。已有方法没有一个能三者兼得。

**对抗威胁的形式化**：狡猾厂商用**概率性替换**逃避检测，目标模型可建模为参考模型与替代模型的混合：

$$\pi_{\text{tgt}}(\cdot\mid x;\varphi) = \bigl(1-q(x)\bigr)\,\pi_{\text{ref}}(\cdot\mid x;\varphi) + q(x)\,\pi_{\text{alt}}(\cdot\mid x;\varphi)$$

其中 $\pi_{\text{alt}}$ 是任意替代模型（量化版、有害微调版等），路由函数 $q:\mathcal{X}\to[0,1]$ 未知、可依赖 prompt 甚至历史交互（如基于 prompt caching 路由）。当 $q(x)$ 很小却作用在不可忽略比例的常规 prompt 上时，混合分布极难区分，但合格的检测仍必须保持统计功效。

**核心 idea**：把审计形式化为**模型等价检验**——给定对目标 API 的查询权限和一个完全可访问的参考模型，判断二者在共享 prompt 上是否产生统计上不可区分的输出。RUT 的关键观察是：用本地参考模型给 API 输出打一个标量分，再看这个分在参考分布里排第几（百分位），**如果两模型相同，这个百分位必然均匀分布**——于是把"两模型是否相等"转化为"一列排名是否服从 Uniform[0,1]"这个干净的一维均匀性检验。

## 方法详解

### 整体框架

RUT 是一个**非对称的双样本假设检验**。所谓非对称，是指它对两侧模型的使用方式完全不同：目标 API 贵且只能看到一次输出，所以**每个 prompt 只查目标一次**；参考模型在本地、可无限采样，所以**每个 prompt 在参考模型上采 $m$ 个样本**来近似参考分布。流程是：取一批长得像真实用户流量的 prompt（实验用 WildChat），对每个 prompt 拿到目标的 1 个回复和参考的 $m$ 个回复，用一个标量打分函数（最终选 log-rank）把每个回复打成分数，计算目标分数在参考分数集合里的"随机化秩"（一个 $[0,1]$ 的百分位），最后对这批秩做 Cramér–von Mises 均匀性检验，$p<0.05$ 就判定 API 模型与宣称模型不同。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["真实用户风格 prompt<br/>(WildChat)"] --> B["非对称采样<br/>目标查1次 + 参考采m次"]
    B --> C["打分函数 f<br/>log-rank 经验选优"]
    C --> D["随机化秩统计量<br/>目标分在参考分布的百分位"]
    D -->|两模型相同则秩~Uniform[0,1]| E["Cramér–von Mises<br/>均匀性检验"]
    E -->|p<0.05| F["判定模型被替换"]
```

### 关键设计

**1. 秩 = 均匀性：把"模型是否相等"转成"一列排名是否均匀"**

这是全文的理论支点，针对的是"黑盒下没有 logits、难以直接比两个分布"的痛点。对固定 prompt $x$，定义参考模型分数的累积分布 $F_{\pi_{\text{ref}}}(s\mid x) := \Pr_{y\sim\pi_{\text{ref}}}[f(y,x)\le s]$。把目标输出的分数 $s_{\text{tgt}}$ 代进这个 CDF 得到的就是它的百分位排名。由概率积分变换，**若 $\pi_{\text{tgt}}=\pi_{\text{ref}}$，则这个排名服从 Uniform[0,1]**；反之只要排名均匀，就能反推目标分数与参考分数同分布，再借助打分函数的（近似）单射性反推两模型输出同分布。于是"两个高维文本分布是否相等"被压缩成一个干净的一维均匀性问题。

由于文本回复是离散的、$f$ 取值离散，CDF 是阶梯函数，直接代入会得到离散的秩、在原假设下并不连续。论文用 **randomized quantile residual**（Dunn & Smyth, 1996）把概率积分变换推广到离散分布，定义连续化的秩统计量：

$$r_{\text{tgt}} := F_{\pi_{\text{ref}}}(s^-_{\text{tgt}}) + U\cdot P\bigl(f(y,x)=s_{\text{tgt}}\bigr),\quad U\sim\text{Uniform}[0,1]$$

其中 $F_{\pi_{\text{ref}}}(s^-_{\text{tgt}})$ 是 CDF 在 $s_{\text{tgt}}$ 处的左极限（严格小于的概率质量），后一项用一个独立均匀随机变量在"打平"的概率质量内做随机插值。如此一来，原假设下 $r_{\text{tgt}}$ 恰好落在 $[0,1]$ 上均匀分布，离散打平问题被干净处理。

**2. 用 $m$ 个参考样本经验估计 CDF + 随机化破平**

真实 $F_{\pi_{\text{ref}}}$ 不可计算，实践中用每个 prompt 的 $m$ 个参考样本构造经验 CDF。给目标响应分数 $s_i=f(y_i,x_i)$ 和参考响应分数 $s_{ij}=f(y_{ij},x_i)$，秩统计量的经验版为：

$$r_i = \frac{1}{m}\left(\sum_{j=1}^{m}\mathbb{1}\{s_i>s_{ij}\} + U_i\cdot\sum_{j=1}^{m}\mathbb{1}\{s_i=s_{ij}\}\right),\quad U_i\sim\text{Uniform}[0,1]$$

第一项数"有多少参考样本严格低于目标分"，第二项在与目标分相等的样本里用 $U_i$ 随机插值破平。这个 $U_i$ 不是可有可无的工程细节——正是它保证了 $r_i$ 是 $r_{\text{tgt}}$ 的**无偏估计**、并让原假设下的秩在离散场景里也保持均匀，从而保住检验的 type-I error 控制（不会因为离散打平而虚高拒绝）。

**3. log-rank 打分函数：经验选出最具判别力的标量分**

理论上需要一个对 $y$ 单射的打分函数才能保证"任何差异都反映在分数上"，但能算出 CDF 的单射函数实际造不出来。论文退而求其次要求打分函数**足够有判别力**：只要 $\pi_{\text{ref}}\neq\pi_{\text{tgt}}$，两者诱导的分数分布就不同（$P_{S_{\pi_{\text{ref}}}}\neq P_{S_{\pi_{\text{tgt}}}}$），这样分布差异就会让秩偏离均匀。为找到最优打分函数，作者用 Monte Carlo（500 trials、每 trial 随机取 10 个 WildChat prompt、每 prompt 从两模型各采 50 个补全、temperature 0.5、长度上限 30 token）比较五个候选——log-likelihood、token rank、log-rank、entropy、log-likelihood log-rank ratio（LRR）——以平均 AUROC 偏离 0.5 的程度衡量判别力。结果 **log-rank（token 排名取对数后求平均）的 AUROC 分布最稳定地远离 0.5**（图 2 中 Gemma-2-9b-it vs 其 4-bit 量化版即为一例），因此选作 RUT 的打分函数。直觉上，量化/微调最先扰动的是"模型在每一步给真实输出 token 排第几"，log-rank 正好刻画这种 token 级典型性，而对数压缩让偶发的高排名不至于主导整体分数。

**4. Cramér–von Mises 均匀性检验得出 p 值**

拿到一批秩 $\{r_i\}_{i=1}^n$ 后，用 **Cramér–von Mises（CvM）检验**评估它们与 Uniform[0,1] 的偏离，检验 $H_0: r_i\sim\text{Uniform}[0,1]$。把秩排序为 $r_{(1)}\le\cdots\le r_{(n)}$，统计量为：

$$\omega^2 = \frac{1}{12n} + \sum_{i=1}^{n}\left(\frac{2i-1}{2n}-r_{(i)}\right)^2$$

它度量经验 CDF 与理论均匀 CDF 的积分平方距离。$p$ 值由观测统计量 $\omega^2_{\text{obs}}$ 在原假设分布下的尾概率 $\Pr_{H_0}(\omega^2_{\text{null}}\ge\omega^2_{\text{obs}})$ 给出，$p<0.05$ 则拒绝原假设、判定模型被替换。选 CvM 而非 KS 是因为它对整段分布的偏离更敏感（KS 只看最大间隙）——论文也实现了一个用同样 log-rank 信息的 KS baseline 作对照，实验中 CvM 路线一致更强。

### 损失函数 / 训练策略
本文是一个统计检验方法，不涉及模型训练，无损失函数。唯一的"调参"是离线的打分函数选择（§4.2 的 Monte Carlo AUROC 比较）。检验侧关键超参：每 trial 取 100 个 prompt，目标查 1 次、参考查 100 次（$m=100$），temperature 0.5、生成上限 30 token，显著性水平 $\alpha=0.05$，功效用 500 次 Monte Carlo trial 估计。

## 实验关键数据

评测以 **statistical power AUC** 为主指标：模拟概率性替换攻击，让比例 $q\in[0,1]$ 的 API 查询被路由到替代模型，对每个 $q$ 估计统计功效（正确拒绝原假设的概率），再对 power–substitution 曲线在 $q\in[0,1]$ 上求 AUC，越高代表在各替换比例下越能稳定检出。数据用 WildChat（真实用户对话），baseline 为 MMD（Gao et al., 2025）和一个用同样 log-rank 信息的 KS 检验。

### 主实验：四类威胁全覆盖

| 威胁场景 | 设置 | 结论 |
|--------|------|------|
| 量化 (§5.2) | 3 个模型 ×（4-bit/8-bit） | 4/6 设置中 RUT 在 3 个里最强；8-bit 的 Gemma/Mistral 所有方法都几乎失效（功效近 0） |
| 越狱 (§5.3) | Mistral/Gemma × Dan/Anti-Dan/Evil-Bot | **6/6 设置 RUT 功效最高**，全部 power AUC > 0.75 |
| 有害微调 SFT (§5.4) | Llama/Mistral × Alpaca/BeaverTails，LoRA 5 epoch | RUT 全配置功效最高，**第 1 个 epoch 内就能检出**早期分布漂移 |
| 整模型替换 (§5.5) | 5 个开源模型两两对比 | RUT 所有模型对上 AUC 均最高 |

代表性数字（量化检测 power AUC，Table 2a 节选）：

| 模型–量化 | RUT | MMD | KS |
|----------|-----|-----|----|
| Gemma–4bit | **0.392** | 0.214 | 0.017 |
| Llama–4bit | **0.642** | 0.625 | 0.474 |
| Mistral–4bit | **0.586** | 0.500 | 0.330 |
| Gemma/Mistral–8bit | ~0.05（均失效） | ~0.05 | ~0.00 |

### 跨查询域与真实 API（消融/泛化）

| 配置 | 关键发现 |
|------|---------|
| 跨域 (§5.6) BigCodeBench / MATH | 把量化检测搬到代码/数学域，8 个设置中 RUT 在 7 个里胜出，说明对查询分布泛化 |
| 真实 API (§5.7) | Llama/Mistral/Gemma 跨 Nebius、Novita、HF Inference 等厂商，以本地 A100 为基线 |
| KS 对照 | 趋势相同但灵敏度明显偏低，验证 CvM 比 KS 更优 |

### 关键发现
- **8-bit 量化是公认难题**：Gemma、Mistral 的 8-bit 变体所有方法功效都近 0——8-bit 对输出分布扰动太小，是当前审计的盲区。
- **对字符串格式扰动鲁棒**：真实 API 实验里 Mistral + HF Inference 一例，MMD 功效冲到 1.0、其余方法 < 0.2；溯源发现是 HF Inference 把参考输出开头的前导空格吞了，而 MMD 用字符级 Hamming 距离，被格式差异虚高。补回空格后 MMD 降到 0.211 与其他方法对齐。RUT 基于 log-rank 而非字符串，**天然不被这种微小解码不一致误导**。
- **越狱最好抓、量化最难抓**：越狱系统提示对输出分布扰动大（power AUC 普遍 > 0.85），高位量化扰动最小最难检。

## 亮点与洞察
- **"排名均匀性"这个转化非常优雅**：把"两个高维文本分布是否相等"压成"一列 $[0,1]$ 数是否均匀"，理论干净（概率积分变换）、检验工具成熟（CvM），还顺手解决了黑盒下没 logits 的难题——只要本地能采样参考模型即可。
- **非对称采样切中成本痛点**：贵的目标 API 每 prompt 只查 1 次，便宜的本地参考模型随便采，把审计成本压到极低，这正是"让审计变得人人愿意做"的关键。
- **by-design 的对抗鲁棒**：因为用的是真实用户风格 prompt（WildChat）、没有特制查询分布，狡猾厂商无法靠"这 prompt 看着像审计"来识别并重路由——鲁棒性是设计出来的而非补丁。
- **randomized quantile residual 破平的细节值得学**：离散分布做均匀性检验时，不处理打平会让秩不连续、虚高拒绝率，这个随机插值技巧可迁移到任何"离散打分 + 均匀性检验"的场景。

## 局限与展望
- **未验证全知对手**：作者承认没测过"完全了解 RUT 机制"的对手——这种对手可以专门把会产生异常 log-rank 的 prompt 重路由回真模型来规避，评估对更强对手的鲁棒性是重要的下一步。
- **必须有本地参考模型**：RUT 要求能本地部署一份可信的参考实现并采样，因此只适用于开源权重模型；闭源模型无法审计。放宽这一前提会显著扩大适用面。
- **8-bit 量化盲区**：高位量化的分布扰动太小，所有方法都抓不到，说明在最隐蔽的替换上审计仍力不从心。
- **个人观察**：log-rank 打分函数是经验选出的，对新型替换（如蒸馏、推测解码）是否仍最优没有理论保证；且固定 temperature 0.5 / 30 token 的设置下结论是否随解码参数变化也值得进一步考察。

## 相关工作与启发
- **vs MMD（Gao et al., 2025）**：同样把审计形式化为 model equality test，但 MMD 用特制查询分布 + 字符级 Hamming 距离的核 MMD，既能被对抗厂商识别绕过、又会被格式扰动误导；RUT 用自然 prompt + log-rank 秩检验，对抗鲁棒且对解码不一致稳健，每 prompt 还只需 1 次 API 调用。
- **vs 分类器指纹（Sun et al., 2025）**：训练文本分类器抓模型 idiosyncrasy，需海量 API 查询且被报告抓不到量化；RUT 查询高效且对量化（至少 4-bit）有效。指纹的目标是"无论怎么微调都认得出来源"，与审计"任何偏离都要报警"的目标恰好相反。
- **vs 身份提问（Huang et al., 2025）/ 基准对比（Chen et al., 2023a）**：前者抓不到尺寸/版本/量化差异，后者靠原始性能无法暴露隐蔽替换或部分路由；RUT 直接比较输出分布，对这些细微替换更敏感。
- **vs 并行工作 Cai et al. (2025)**：后者揭示厂商可用量化、随机替换、基准规避来逃检测；RUT 正是针对这些规避策略设计了更鲁棒的方法，并把威胁模型扩展到越狱、有害微调等更现实的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "秩→均匀性"的转化把黑盒模型等价检验做得既优雅又实用，对抗鲁棒性是设计内生的
- 实验充分度: ⭐⭐⭐⭐ 量化/越狱/SFT/整模型替换/跨域/真实 API 六大场景齐全，但缺全知对手的实证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰、Table 1 三准则定位精准、真实 API 的格式扰动案例很有说服力
- 价值: ⭐⭐⭐⭐⭐ API 时代模型审计是刚需，方法查询高效、易落地、代码开源，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[ICLR 2026\] Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs](bias_similarity_measurement_a_black-box_audit_of_fairness_across_llms.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](../../CVPR2026/llm_safety/omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)

</div>

<!-- RELATED:END -->
