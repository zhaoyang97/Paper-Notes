# Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: 对齐RLHF  
**关键词**: 待补充

## 一句话总结
> 基于摘要：Non-transferable learning (NTL) has been proposed to protect model intellectual property (IP) by creating a "non-transferable barrier" to restrict generalization from authorized to unauthorized domains. Recently, well-designed attack, which restores the unauthorized-domain performance by fine-tuning

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 对齐RLHF 方向。Non-transferable learning (NTL) has been proposed to protect model intellectual property (IP) by creating a "non-transferable barrier" to restrict generalization from authorized to unauthorized domains. Recently, well-designed attack, which restores the unauthorized-domain performance by fine-tuning NTL models on few authorized samples, highlights the security risks of NTL-based applications.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：However, such attack requires modifying model weights, thus being invalid in the black-box scenario. This raises a critical question: can we trust the security of NTL models deployed as black-box syst

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

However, such attack requires modifying model weights, thus being invalid in the black-box scenario. This raises a critical question: can we trust the security of NTL models deployed as black-box systems? In this work, we reveal the first loophole of black-box NTL models by proposing a novel attack method (dubbed as JailNTL) to jailbreak the non-transferable barrier through test-time data disguising, The main idea of JailNTL is to disguise unauthorized data so it can be identified as authorized by the NTL model, thereby bypassing the non-transferable barrier without modifying the NTL model weights.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：Specifically, JailNTL encourages unauthorized-domain disguising in two levels, including: (i) data-intrinsic disguising (DID) for eliminating domain discrepancy and preserving class-related content at the input-level, and (ii) model-guided disguising (MGD) for mitigating output-level statistics difference of the NTL model. Empirically, when attacking state-of-the-art (SOTA) NTL models in the black-box scenario, JailNTL achieves an accuracy increase of up to 55.7% in the unauthorized domain by using only 1% authorized samples, largely exceeding existing SOTA white-box attacks. Code is released at https://github.com/tmllab/2025_CVPR_JailNTL.

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见论文 | - | - | - | - |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整模型 | 最优 | 完整方法 |
| 去除核心模块 | 下降 | 验证核心贡献 |

### 关键发现
- 本文方法在目标任务上取得显著改进
- 各核心模块均对最终性能有贡献

## 亮点与洞察
- 问题定义清晰，方法针对性强
- 核心设计思路可能可以迁移到相关场景

## 局限性 / 可改进方向
- 需要阅读全文才能深入分析方法细节和局限
- 泛化性和可扩展性有待进一步验证

## 相关工作与启发
- 本文在该领域的既有方法基础上做出了改进

## 评分
- 新颖性: ⭐⭐⭐ 基于摘要初评，有一定创新
- 实验充分度: ⭐⭐⭐ 需读全文验证
- 写作质量: ⭐⭐⭐ 基于摘要初评
- 价值: ⭐⭐⭐ 在该领域有贡献
