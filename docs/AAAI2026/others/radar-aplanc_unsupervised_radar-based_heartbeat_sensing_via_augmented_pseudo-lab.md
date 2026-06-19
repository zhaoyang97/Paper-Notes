---
title: >-
  [论文解读] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast
description: >-
  [AAAI 2026][雷达心跳感知] 提出首个雷达心跳感知的无监督学习框架 Radar-APLANC，通过噪声对比三元组损失（NCT loss）和增强伪标签生成器实现两阶段无监督训练，无需昂贵的生理信号标注即可达到接近监督方法的性能。 雷达心跳感知的价值与挑战 FMCW 雷达可以通过检测亚毫米级（0.1-0.5mm）的胸壁…
tags:
  - "AAAI 2026"
  - "雷达心跳感知"
  - "无监督学习"
  - "伪标签"
  - "噪声对比学习"
  - "FMCW 雷达"
---

# Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast

**会议**: AAAI 2026  
**arXiv**: [2511.08071](https://arxiv.org/abs/2511.08071)  
**代码**: [https://github.com/RadarHRSensing/Radar-APLANC](https://github.com/RadarHRSensing/Radar-APLANC)  
**领域**: 其他  
**关键词**: 雷达心跳感知, 无监督学习, 伪标签, 噪声对比学习, FMCW 雷达

## 一句话总结

提出首个雷达心跳感知的无监督学习框架 Radar-APLANC，通过噪声对比三元组损失（NCT loss）和增强伪标签生成器实现两阶段无监督训练，无需昂贵的生理信号标注即可达到接近监督方法的性能。

## 研究背景与动机

### 雷达心跳感知的价值与挑战

FMCW 雷达可以通过检测亚毫米级（0.1-0.5mm）的胸壁位移来实现非接触式心跳感知，在隐私保护、环境鲁棒性和持续监测方面具有独特优势。然而：

**传统方法的噪声敏感性**：基于相位提取和解缠绕的传统方法在运动伪影、多径干扰和低信噪比条件下性能严重退化，相位缠绕歧义和噪声敏感性是根本限制。

**监督方法的标注瓶颈**：深度学习方法（如 Equipleth RF, VitaNet, CardiacMamba）虽然噪声鲁棒性更好，但需要大规模高质量的生理信号标注（如同步 PPG 信号），采集成本高昂，限制了训练数据的可扩展性。

**视频域方法不可直接迁移**：已有视频领域的无监督生理监测方法（如对比学习范式），但雷达数据的心跳信号信噪比更低、表征形式不同（胸壁运动 vs 面部颜色变化）、传统正负样本构造策略在强噪声下失效。

### 核心洞察

雷达距离矩阵中天然存在"心跳距离 bin"和"噪声距离 bin"的对比结构——可以利用这种内在的信号-噪声分离来构建正负样本，无需外部标注。

## 方法详解

### 整体框架

Radar-APLANC 是一个两阶段无监督框架：
- **Stage 1**：利用传统雷达方法生成伪标签，结合噪声对比三元组（NCT）损失进行预训练
- **Stage 2**：引入增强伪标签生成器，通过质量评估和自适应噪声感知标签选择改善伪标签质量，进一步微调

### 预备知识

**距离矩阵获取**：FMCW 雷达发送线性调频信号 $s(t)$，接收反射信号 $u(t)$，通过 IQ 解调得到中频信号 $m(t)$，对每个 chirp 的 IF 信号做 FFT 得到距离剖面 $M_n[f]$，拼接 $N$ 个 chirp 得到距离矩阵 $M \in \mathbb{R}^{N \times D}$。

**基础心跳感知**：(1) 选择最大功率占比的距离 bin $d^*$（人体位置）；(2) 计算该 bin 的相位信号；(3) 相位解缠绕；(4) 0.8-3.0 Hz 带通滤波得到心跳信号 $\Phi(\cdot) \in \mathbb{R}^N$。

### 关键设计

#### 1. **噪声对比三元组损失（NCT Loss）—— Stage 1**

**核心思路**：利用雷达距离矩阵中心跳 bin 和噪声 bin 的天然对比来构建自监督学习信号。

- **伪标签生成**：用传统雷达方法从中心距离 bin $d^*$ 提取心跳信号，经随机时间采样和功率谱密度（PSD）变换得到伪标签集 $S_{PL}$
- **正样本构造**：取中心 bin 周围窗口的心跳矩阵 $M(\cdot, d^* \pm \Delta d)$ 输入心跳提取器，输出预测心跳信号 $p(\cdot)$，同样做 PSD 变换得到正样本集 $S_P$
- **负样本构造**：从距离矩阵中随机选择非中心 bin $d'$ 的窗口作为噪声矩阵，输入噪声提取器得到噪声信号 $q(\cdot)$，PSD 变换得到负样本集 $S_N$

**NCT Loss**：
$$\mathcal{L}_{NCT} = \underbrace{\frac{1}{K^2}\sum_{i,j}\|S_{PL}[i] - S_P[j]\|^2}_{\text{正项：拉近心跳与伪标签}} + \underbrace{(-\frac{1}{K^2}\sum_{i,j}\|S_P[i] - S_N[j]\|^2)}_{\text{负项：推远心跳与噪声}}$$

**设计动机**：所有传统信号处理方法可以看作某种伪标签生成器，虽然有噪声，但仍包含心跳信息。而距离矩阵中非人体位置的 bin 主要包含背景噪声，天然构成负样本。PSD 变换使频域比较更稳定。

#### 2. **增强伪标签生成器（Augmented Pseudo-Label Generator）—— Stage 2**

**核心思路**：利用 Stage 1 的预训练模型来评估和选择更高质量的伪标签。

分为两个子模块：

**质量评估模块**：
- 从心跳窗口的 $2\Delta d + 1$ 个距离 bin 各自提取传统心跳信号 $\{\Phi_1, \ldots, \Phi_{2\Delta d+1}\}$
- 计算噪声距离 $X_i = D(\Phi_i, q)$：与预训练噪声信号的距离，越大表示信号质量越好
- 计算心跳距离 $Y_i = D(\Phi_i, p)$：与预训练心跳信号的距离，越小表示信号质量越好
- 距离度量使用两个信号心率的平均绝对误差

**决策模块**（自适应噪声感知标签选择）：
- 理想情况：$\arg\max_i X_i = \arg\min_i Y_i$（噪声距离最大 = 心跳距离最小），直接选该信号
- 冲突情况：$\arg\max_i X_i \neq \arg\min_i Y_i$，则检查心跳距离最小的信号 $\Phi_{\arg\min Y_i}$ 的噪声距离是否大于预训练心跳信号的噪声距离 $D(p, q)$
    - 若是：选 $\Phi_{\arg\min Y_i}$（传统方法更好）
    - 否则：选预训练心跳信号 $p$（深度学习方法更好）

### 损失函数 / 训练策略

- Stage 1 和 Stage 2 都使用 NCT Loss，唯一区别是伪标签来源
- 两阶段都训练心跳提取器和噪声提取器
- 优化器：AdamW，学习率 1e-4，训练 200 epochs
- 评估时使用 10 秒窗口

## 实验关键数据

### 主实验

两个数据集：Equipleth（公开，91 人 550 段录制）和 RHB（自采集，80 人 240 段录制）。

| 方法 | 类型 | Equipleth MAE↓ | Equipleth r↑ | RHB MAE↓ | RHB r↑ |
|------|------|---------------|-------------|---------|--------|
| FFT-based RF | 传统 | 13.51 | 0.24 | 12.25 | 0.26 |
| Equipleth RF | 监督 | **2.18** | **0.89** | 3.19 | 0.82 |
| VitaNet | 监督 | 3.14 | 0.77 | 5.28 | 0.66 |
| mmFormer | 监督 | 6.50 | 0.52 | 8.89 | 0.28 |
| **Radar-APLANC** | **无监督** | 3.95 | 0.64 | 3.92 | 0.77 |

跨数据集测试（泛化能力）：

| 方法 | RHB→Equipleth MAE | Equipleth→RHB MAE |
|------|-------------------|-------------------|
| Equipleth RF | 4.53 (+107.8%) | 2.68 |
| VitaNet | 7.43 (+136.6%) | 2.38 |
| **Radar-APLANC** | **4.10 (+3.8%)** | **3.52** |

### 消融实验

| Stage 1 配置 | Stage 2 配置 | MAE↓ | RMSE↓ | r↑ |
|-------------|-------------|------|-------|-----|
| 仅噪声矩阵 | - | 34.48 | 38.34 | 0.01 |
| 仅伪标签 | - | 8.94 | 15.88 | 0.30 |
| 噪声+伪标签 | - | 4.40 | 9.89 | 0.63 |
| 噪声+伪标签 | 仅增强伪标签 | 7.42 | 13.61 | 0.38 |
| 噪声+伪标签 | 增强伪标签+噪声 | **3.95** | **9.72** | **0.64** |

增强伪标签生成器消融：

| 预训练心跳 | 传统心跳 | 噪声信号 | MAE↓ |
|-----------|---------|---------|------|
| ✓ | - | - | 4.56 |
| ✓ | ✓ | - | 8.75 |
| - | ✓ | ✓ | 14.48 |
| ✓ | ✓ | ✓ | **3.95** |

### 关键发现

1. **噪声矩阵是关键**：单独使用伪标签 MAE=8.94，加入噪声对比后 MAE=4.40，降低超过一半。这证明了利用雷达距离矩阵内在噪声结构的有效性。
2. **两阶段互补**：Stage 1 从 8.94 降到 4.40，Stage 2 进一步降到 3.95。增强伪标签需要三种信号配合才能最优。
3. **跨数据集稳定性**：无监督方法 MAE 波动仅 0.4 bpm，而监督方法波动超过 100%。
4. **肤色公平性**：雷达方法在深浅肤色间的性能差异（fairness 指标）远小于 RGB 方法，无监督雷达方法的公平性与监督雷达方法相当。

## 亮点与洞察

1. **首创雷达无监督心跳感知**：填补了重要空白，既解决了标注瓶颈，又保留了雷达的隐私和鲁棒性优势
2. **噪声作为资源的范式转换**：传统上噪声是要消除的对象，本文将其作为对比学习的负样本来源，化害为利
3. **两阶段渐进式改进**：先用粗糙伪标签预训练，再用改进的伪标签微调，是一种通用的自举策略
4. **实用的肤色公平性**：在关注 AI 公平性的大背景下，雷达无监督方法展现了比 RGB 方法更公平的感知能力
5. **新数据集贡献**：RHB 数据集（80 人）将开源，有助于促进社区研究

## 局限与展望

1. 与最佳监督方法仍有约 1.8 bpm 的 MAE 差距，伪标签质量是瓶颈
2. 评估仅限于静坐场景（0.5-1m 距离），运动干扰和更远距离的场景未测试
3. 仅验证心率估计，未扩展到呼吸率或心率变异性等更复杂的生理指标
4. Stage 2 的增强伪标签生成器依赖启发式的决策规则，可用更端到端的方法替代
5. 未与视频域的无监督方法做跨模态比较实验

## 相关工作与启发

- **视频领域的无监督 rPPG**（Gideon 2021, Sun 2022）提供了对比学习用于生理信号的思路，但其空间-时间正负样本构造不适用于雷达
- **Equipleth**（Vilesov 2022）提供了雷达-视频多模态数据集和监督基线
- **自监督雷达方法**（Song 2022, Zhang 2025）探索了雷达的自监督学习，但仍需标注微调，不是真正的无监督

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首个雷达无监督心跳框架，噪声对比思路新颖）
- 实验充分度: ⭐⭐⭐⭐（两个数据集、跨数据集测试、公平性分析全面）
- 写作质量: ⭐⭐⭐⭐（结构清晰，动机论述充分）
- 价值: ⭐⭐⭐⭐⭐（解决实际痛点，新数据集+代码开源）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RaUF: Learning the Spatial Uncertainty Field of Radar](../../CVPR2026/others/rauf_learning_the_spatial_uncertainty_field_of_radar.md)
- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](../../NeurIPS2025/others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[AAAI 2026\] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise](predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)

</div>

<!-- RELATED:END -->
