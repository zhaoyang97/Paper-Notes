# Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**会议**: CVPR 2026  
**arXiv**: [2603.13182](https://arxiv.org/abs/2603.13182)  
**代码**: 无  
**领域**: 医学图像分类 / 对抗鲁棒性 / 特征空间防御  
**关键词**: brain tumor classification, NNMF, diffusion-based defense, adversarial robustness, AutoAttack  

## 一句话总结
将 MRI 脑肿瘤分类任务分解为 NNMF 特征提取 → 统计特征筛选 → 轻量 CNN 分类 → 特征空间扩散净化四阶段流水线，在 AutoAttack 下将鲁棒精度从基线 0.5% 提升到 59.5%。

## 背景与动机
脑肿瘤 MRI 分类中深度学习模型达到了高精度，但对对抗性扰动极为脆弱。在临床部署中，模型的可靠性和安全性至关重要。NNMF 能产生非负的 parts-based 可解释表示，天然适合 MRI 等非负数据；扩散模型的去噪能力已在图像空间防御中展现潜力，但在特征空间的应用尚未充分探索。

## 核心问题
如何在不牺牲分类精度的前提下，为脑肿瘤 MRI 分类模型提供对抗鲁棒性？特别是在 AutoAttack 这种强攻击基准下维持可靠性能。

## 方法详解

### 整体框架
四阶段流水线：(1) 预处理 + NNMF 特征提取 → (2) 统计特征排序与筛选 → (3) 轻量 CNN 分类 → (4) 特征空间扩散前向加噪 + 学习去噪器进行净化，最后在净化特征上执行分类。

### 关键设计
1. **NNMF 特征提取**：MRI 图像转为灰度、resize 到 128×128、归一化到 [0,1]，向量化构成非负矩阵 $V$。使用 KL 散度 + 乘法更新规则分解为 $V \approx WH$（rank=15），$W$ 为基分量矩阵，$H$ 为系数矩阵。验证/测试集特征通过非负最小二乘投影到固定 $W$ 上，最终 L2 归一化。
2. **统计特征筛选**：对 15 个 NNMF 分量逐一评估 AUC（区分能力）、Cohen's d（效应量）、Welch's t-test p 值（统计显著性），选取多指标综合排名靠前的 Top-M 特征子集。
3. **轻量 CNN 分类**：在选定的 NNMF 特征上训练小型 CNN（非图像输入），验证集精度 ~83%，测试集 ~85.1%。
4. **特征空间扩散防御**：定义线性噪声时间表在 NNMF 特征空间执行前向扩散（逐步加高斯噪声），训练回归去噪网络（输入为噪声特征 + 正弦位置嵌编码的 timestep，输出干净特征 $\hat{x}_0$）。推理时：对测试特征加噪（选定 timestep, 如 t=41）→ 去噪器还原 → 净化特征送入分类器。由于加噪具有随机性，采用 Expectation over Transformation（EOT, K=8 次均值）。

### 损失函数 / 训练策略
- NNMF：KL 散度目标函数 + 乘法更新
- CNN 分类：标准交叉熵损失
- 去噪器：MSE 损失（$\|\hat{x}_0 - x_0\|^2$）
- 鲁棒性评估：AutoAttack（$L_\infty$, $\epsilon=0.10$），包含 APGD-CE 和 Square Attack 两个组件

## 实验关键数据
- 数据集：Kaggle 脑肿瘤 MRI 数据集，约 2200 张图像，二分类（正常 vs 肿瘤），70/20/10  split
- Clean Baseline：Accuracy 86.05%, ROC-AUC 0.9105, MCC 0.7178
- Clean Defended（扩散净化后）：Accuracy 85.12%, ROC-AUC 0.8967, MCC 0.6988（仅轻微下降）
- Robust Baseline（无防御 + AutoAttack）：Accuracy 0.47%, ROC-AUC 0.0075, MCC −0.9906（完全崩溃）
- Robust Defended（扩散防御 + AutoAttack）：Accuracy 59.53%, ROC-AUC 0.7485, MCC 0.1703（大幅恢复）
- 计算效率：GPU 总运行时间 116.6s（vs CPU 201.5s），加速 1.73×

### 消融实验要点
- NNMF 基分量可视化显示其捕获了颅骨边界、组织分布、局部密度变化等解剖结构
- 类别均值热力图显示肿瘤类在特定分量上有系统性更高激活
- 去噪器有效降低重建误差：去噪后误差 $\|\hat{x}_0 - x_0\|$ 明显小于噪声误差 $\|x_t - x_0\|$
- Brier Score 从基线 Robust 的 0.4702 降至 Defended 的 0.2150，概率校准显著改善

## 亮点
- 在特征空间而非像素空间执行扩散防御，计算成本低且与任何下游分类器解耦
- NNMF 的 parts-based 表示提供了可解释性，可以直观看到每个分量对应的解剖结构
- 防御效果明显：在 AutoAttack 下从 0.5% 恢复到 59.5%，同时干净精度仅下降 ~1%

## 局限性 / 可改进方向
- 数据集仅 ~2200 张图像，二分类任务过于简单；未确认 patient-wise split，可能存在切片级数据泄漏
- NNMF rank=15 的选择缺乏系统消融，不确定是否最优
- 仅在单一 $\epsilon=0.10$ 下评估，未探索不同攻击强度的鲁棒性曲线
- 扩散 timestep 的选择（t=41）看似 ad-hoc，缺乏关于 timestep 与鲁棒性/精度权衡的分析
- MATLAB+Python 混合流水线影响实用性，端到端实现更可取

## 与相关工作的对比
- vs **端到端 CNN 分类**（如 Hossain et al. 的 5 层 CNN, 97.87%）：本文策略性牺牲少量 clean accuracy 来换取鲁棒性
- vs **NMF-CNN 混合方法**（Chan et al., 声学事件检测）：思路类似但本文首次将 NMF+CNN+扩散防御结合用于医学影像
- vs **Classification-Denoising Networks**（Thiry & Guth）：后者同时学习分类和去噪，本文采用分离式设计更模块化
- vs **AutoAttack 基准**（Croce & Hein）：严格使用 AA 标准评估，避免了仅用弱攻击导致的虚假鲁棒性

## 启发与关联
- 特征空间扩散防御的思路可迁移到其他医学影像分类任务的对抗鲁棒性研究
- NNMF 作为可解释中间表示层，有助于构建更透明的临床 AI 系统

## 评分
- 新颖性: ⭐⭐⭐ NNMF + 扩散防御的组合有新意，但各组件均非原创
- 实验充分度: ⭐⭐⭐ 评估指标丰富但数据集太小，缺乏消融和多攻击场景
- 写作质量: ⭐⭐ 语法和表达有较多问题，部分段落可读性差
- 价值: ⭐⭐⭐ 特征空间防御思路有参考价值，但实验规模和严谨性有待提升
