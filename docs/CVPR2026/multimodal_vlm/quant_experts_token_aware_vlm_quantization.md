# Quant Experts: Token-aware Adaptive Error Reconstruction for Large VLM Quantization

**会议**: CVPR 2026  
**arXiv**: [2602.24059](https://arxiv.org/abs/2602.24059)  
**代码**: 待确认  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: PTQ量化, MoE, VLM, token感知, 自适应补偿, 低秩适配器  

## 一句话总结
揭示VLM中重要通道的分布和出现频率在跨模态和token间差异显著，提出基于MoE的token感知PTQ框架：共享专家补偿全局token无关误差，路由专家自适应补偿局部token依赖误差，72B模型W4A6恢复5.09%精度。

## 背景与动机
VLM PTQ中，现有方法依赖静态识别和全局补偿敏感通道，但重要通道的位置不固定——不同模态和token的重要通道分布差异巨大。少数通道在多数token中出现（token无关），多数通道仅在特定token中激活（token依赖）。

## 核心问题
需区分token无关和token依赖两类重要通道，采用不同补偿策略。全局补偿无法应对token级动态性。

## 方法详解

### 整体框架
校准数据统计通道频率 → 划分token无关/依赖通道 → SE用whitened SVD重建全局误差 → REs根据NPMI共现聚类+谱聚类分组，每组配低秩适配器 → 路由器动态选最优专家

### 关键设计
1. **通道划分**: 按频率$f_c$排序，前$k$个为token无关，其余为token依赖
2. **共享专家(SE)**: Whitened SVD低秩重建 + 通道缩放降低激活量化误差
3. **路由专家(REs)**: NPMI共现矩阵+谱聚类分组→每组加权SVD低秩适配器→路由器预测最小误差专家

### 损失函数 / 训练策略
误差重建目标$\min\|(E-\tilde{E})x\|_F$；可选细化：逐层16epochs, AdamW lr=1e-4；SVD总秩=64

## 实验关键数据
| 模型/设置 | QE Avg↑ | LQER | MBQ | 全精度 |
|-----------|---------|------|-----|--------|
| Qwen2VL-2B W4A6 | 58.74 | 55.92 | 54.73 | 62.97 |
| InternVL2-8B W4A6 | 68.13 | 65.29 | 65.00 | 70.60 |
| Qwen2VL-72B W4A6 MMMU/OCR | 58.11/76.60 | 52.33/59.60 | 52.67/69.70 | 61.44/78.70 |

### 消融实验要点
- 去掉SE或REs均降性能→互补；随机路由/聚类不如学习的→共现关系有意义
- $N_r$=8性价比最优；细化训练一致提升

## 亮点 / 我学到了什么
- Token感知是VLM量化重要维度——MoE补偿是自然且巧妙的组合
- NPMI+谱聚类揭示了token级语义结构
- 72B W4A6恢复5.09%有实际部署价值
- 3.5-4.5倍NPU加速验证硬件效率

## 局限性 / 可改进方向
- MoE引入额外参数和计算；聚类数需设定
- → 与 `ideas/20260316_attention_aware_quant.md` 和 `ideas/20260316_svd_quant_dense_prediction_vlm.md` 密切相关

## 与相关工作的对比
vs SmoothQuant/AWQ: 静态通道缩放忽略token差异；vs LQER: 全局低秩统一处理；vs MBQ: 模态级不到token级

## 与我的研究方向的关联
Token感知量化可直接启发dense prediction VLM量化；MoE低秩框架可扩展到检测/分割backbone

## 评分
- 新颖性: ⭐⭐⭐⭐ Token感知+MoE量化结合新颖，但子技术成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 2B-72B多尺度、多设置(W4A6/W4A8/W3A16)、11个benchmark
- 写作质量: ⭐⭐⭐⭐ 观察→动机→方法叙述流畅
- 对我的价值: ⭐⭐⭐⭐⭐ 与模型压缩/VLM高效部署直接相关
