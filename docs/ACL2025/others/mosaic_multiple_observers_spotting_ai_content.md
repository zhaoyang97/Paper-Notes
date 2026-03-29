# MOSAIC: Multiple Observers Spotting AI Content

**会议**: ACL 2025 (Findings)
**arXiv**: [2409.07615](https://arxiv.org/abs/2409.07615)
**代码**: [GitHub](https://github.com/BaggerOfWords/MOSAIC)
**领域**: AI安全/AIGC检测
**关键词**: AI-generated text detection, ensemble LLM, information theory, Binoculars, zero-shot detection

## 一句话总结
基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

## 研究背景与动机

1. **领域现状**：LLM 生成文本的检测已成为紧迫需求（虚假新闻/学术抄袭/有害内容）。零样本检测方法（如 GPTZero、Binoculars）基于 detector LLM 的 perplexity/surprisal 判别，但严重依赖单一或固定的 detector 模型对选择。
2. **现有痛点**：(a) 单一 detector 的检测效果因域/语言/生成器不同而波动大；(b) Binoculars 等双模型方法需搜索最优模型对，不同验证集给出不同最优对；(c) 随模型数量增加，枚举所有可能对组合指数爆炸。
3. **核心矛盾**：如何在不依赖特定 detector 选择的情况下，鲁棒地检测多种生成器的输出？
4. **本文要解决什么**：设计一个有理论基础的多 LLM 集成检测方法，自动为各 LLM 分配最优权重。
5. **切入角度**：将检测问题与通用压缩（universal compression）联系——最优的多模型混合就是能最好"压缩"（即最低困惑度描述）人类文本的混合分布。用 Blahut-Arimoto 算法求解这个信息论优化问题。
6. **核心 idea**：$q^*(y_t|\mathbf{y}_{<t}) = \sum_{m \in \mathcal{M}} \mu^*(m|\mathbf{y}_{<t}) p_m(y_t|\mathbf{y}_{<t})$，先用 BA 算法找最优权重 $\mu^*$，再用混合模型 $q^*$ 替代 Binoculars 中的单一 detector $q$。

## 方法详解

### 整体框架
输入：待检测文本 $\mathbf{y}$，一组 detector LLM $\mathcal{M} = \{m_1, ..., m_K\}$ + 一个参考模型 $m^*$。输出：MOSAIC 分数 $S_\mathcal{M}(\mathbf{y})$（低=AI 生成，高=人类写作）。

### 关键设计

1. **最优混合分布 (Blahut-Arimoto)**
   - 做什么：为 $K$ 个 detector LLM 计算位置自适应的最优权重 $\mu^*(m|\mathbf{y}_{<t})$
   - 核心思路：通过迭代 Blahut-Arimoto 算法最大化互信息 $\mathcal{I}(\mathbb{M}; Y_t|\mathbf{y}_{<t})$，使混合模型 $q^*$ 是对当前上下文最具判别力的模型组合
   - 设计动机：来自通用压缩理论——最优混合分布是所有模型中能最短编码（最低困惑度）描述观测文本的分布

2. **MOSAIC 评分**
   - 做什么：每个 token 计算 surprisal 与期望交叉熵的差
   - 公式：$s_t(\mathbf{y}) = \mathcal{L}_{q^*}(y_t|\mathbf{y}_{<t}) - \sum_{y \in \Omega} p_{m^*}(y|\mathbf{y}_{<t}) \mathcal{L}_{q^*}(y|\mathbf{y}_{<t})$
   - 直觉：AI 生成文本在混合模型下 surprisal 低（因为某个 LLM 很好地"预测"了它），但参考模型的期望交叉熵也低 → 差值小 → 检测为 AI
   - 最终分数：$S_\mathcal{M}(\mathbf{y}) = \frac{1}{T}\sum_t s_t(\mathbf{y})$

3. **参考模型选择**
   - 做什么：从集成中选择对人类文本困惑度最低的模型作为 $m^*$
   - 核心思路：$m^* = \arg\min_{m \in \mathcal{M}} -\sum_t \log p_m(y_t|\mathbf{y}_{<t})$ on human text
   - 实践：通常是集成中最大的 LLM（如 Llama-3-70B）

4. **与 Binoculars/FastDetectGPT 的统一视角**
   - Binoculars = MOSAIC 的特例（$|\mathcal{M}|=1$, 固定 $q$）
   - FastDetectGPT = 差值版 Binoculars（逐 token 归一化而非全句平均）
   - MOSAIC 推广为任意数量 detector 的集成，权重自适应

## 实验关键数据

### 多域多生成器检测 (AUROC)

| 方法 | ChatGPT | GPT-4 | Llama | Mistral | 平均 |
|------|---------|-------|-------|---------|------|
| Log-likelihood | 0.82 | 0.78 | 0.85 | 0.83 | 0.82 |
| Binoculars (best pair) | 0.996 | 0.969 | 1.000 | 0.999 | 0.99 |
| Binoculars (fixed pair) | 0.94 | 0.91 | 0.97 | 0.95 | 0.94 |
| **MOSAIC** | **0.995** | **0.972** | **0.999** | **0.998** | **0.99** |

### 消融：集成模型数量

| 集成大小 | 平均 AUROC |
|---------|-----------|
| 1 模型 | 0.91 |
| 2 模型 | 0.95 |
| 4 模型 | 0.98 |
| **8 模型 (MOSAIC)** | **0.99** |

### 关键发现
- **MOSAIC ≈ Binoculars-best-pair 性能，但无需搜索最优对**——鲁棒性大幅提升
- **Binoculars-fixed-pair 退化严重**：固定模型对在某些生成器上 AUROC 降到 0.91，MOSAIC 稳定在 0.99
- **增量集成持续改善**：每增加一个模型，检测性能单调提升——更多"观察者"=更鲁棒
- **跨域/跨语言鲁棒**：在新闻/学术/社交媒体域和英/法/德等语言上一致有效
- **BA 算法权重可解释**：权重自动收敛到对当前文本最相关的 detector——内置"模型选择"

## 亮点与洞察
- **信息论基础优雅**：从通用压缩到 BA 算法再到 token-level 评分，理论链条完整——不是"试了集成发现好"而是"理论推导出如何集成"
- **免验证集的自动模型选择**：MOSAIC 通过 BA 算法自动为每个文本片段找到最优 detector 组合，无需预先在验证集上搜索
- **从 Binoculars 到 MOSAIC 的自然推广**：清晰展示了单模型→双模型→多模型的理论统一视角
- **Blahut-Arimoto 的 NLP 新应用**：BA 算法在通信领域经典，在 NLP 中的应用新颖

## 局限性 / 可改进方向
- **计算成本高**：需要对所有 detector 计算每个 token 的 logits + BA 迭代
- **需要 open-weight 模型**：需要访问模型的 logit 分布，不适用于纯 API 模型
- **共享 tokenizer 假设**：当前要求所有 detector 共享 tokenizer，限制了跨族模型集成
- **对采样策略的鲁棒性**：高温度随机采样的文本可能让检测器过度谨慎
- **未考虑混合文本**：人类写+AI 修改的混合文本是更现实的场景

## 相关工作与启发
- **vs Binoculars (Hans et al., 2024)**：Binoculars 用固定双模型，MOSAIC 推广到最优多模型集成——消除模型对选择的脆弱性
- **vs FastDetectGPT (Bao et al., 2024)**：FastDetectGPT 也基于 surprisal-crossentropy 对比，但用单一模型；MOSAIC 的多模型版本更鲁棒
- **vs DetectGPT (Mitchell et al., 2023)**：基于扰动的方法，计算成本高且依赖扰动质量；MOSAIC 无需扰动

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从信息论通用压缩推导出多LLM集成检测，理论优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多生成器×多域×多语言+消融+鲁棒性分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，Algorithm 1 清晰，与 Binoculars 的统一视角精妙
- 价值: ⭐⭐⭐⭐⭐ 解决了生成器无关检测的鲁棒性核心痛点，实用性强
