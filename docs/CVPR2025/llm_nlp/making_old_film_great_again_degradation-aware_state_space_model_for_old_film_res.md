# Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: NLP理解  
**关键词**: 待补充

## 一句话总结
> 基于摘要：Unlike modern native digital videos, the restoration of old films requires addressing specific degradations inherent to analog sources. However, existing specialized methods still fall short compared to general video restoration techniques. In this work, we propose a new baseline to re-examine the c

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 NLP理解 方向。Unlike modern native digital videos, the restoration of old films requires addressing specific degradations inherent to analog sources. However, existing specialized methods still fall short compared to general video restoration techniques.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：In this work, we propose a new baseline to re-examine the challenges in old film restoration. First, we develop an improved Mamba-based framework, dubbed MambaOFR, which can dynamically adjust the deg

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

In this work, we propose a new baseline to re-examine the challenges in old film restoration. First, we develop an improved Mamba-based framework, dubbed MambaOFR, which can dynamically adjust the degradation removal patterns by generating degradation-aware prompts to tackle the complex and composite degradations present in old films. Second, we introduce a flow-guided mask deformable alignment module to mitigate the propagation of structured defect features in the temporal domain.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：Third, we introduce the first benchmark dataset that includes both synthetic and real-world old film clips. Extensive experiments show that the proposed method achieves state-of-the-art performance, outperforming existing advanced approaches in old film restoration. The implementation and model is available at https://github.com/MaoAYD/MambaOFR.

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
