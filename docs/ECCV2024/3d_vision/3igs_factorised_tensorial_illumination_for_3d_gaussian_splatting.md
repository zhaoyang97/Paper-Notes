# 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: [https://github.com/TangZJ/3iGS](https://github.com/TangZJ/3iGS)  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D Gaussian Splatting, 光照建模, BRDF, 张量分解, 视角依赖效果  

## 一句话总结
通过引入基于张量分解的连续入射光照场和可学习BRDF特征，替代3DGS中独立优化的球谐系数，3iGS显著提升了镜面反射等视角依赖效果的渲染质量，同时保持实时渲染速度。

## 背景与动机
3D高斯体表示（3DGS）已成为从多视角图像重建3D场景并实时渲染新视角的主流方法。然而，3DGS为每个高斯体独立优化球谐系数（SH）来表示出射辐射，这种做法本质上忽略了场# 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: [https://github.com/TangZJ/3?*领域**:](https://github.com/TangZJ/3?*领域**:) 3D视觉 / 新视角合成  
**?*关键词**: 3D Gaussian Splatting, ?
## ???色。但这类逆渲染本身是严重的病态问题，物理参数难以准确估通过引入基亲?## 背景与动机
3D高斯体表示（3DGS）已成为从多视角图像重建3D场景并实时渲染新视角的主流方法。然而，3DGS为每个高斯体独立优化球谐系数（SH）来表示出射辐射，这种???3D高斯体表示??**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: [https://github.com/TangZJ/3?*领域**:](https://github.com/TangZJ/3?*领域**:) 3??*arXiv**: [2408.03753?*?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https?*arXiv**: [2408.03753](htt??*代码**: https?*代码**: https://github.com/TangZJ/3?*领域**: 3D视觉 / 新视角合成  
**?*??*?*关键词**: 3D Gaussian Splatting, ?
## ???色。但这类逆渲染本身是严重的病??## ???色。但这类逆渲染本身是严??3D高斯体表示（3DGS）已成为从多视角图像重建3D场景并实时渲染新视角的主流方法。然而，3DGS为毾?**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: [https://github.com/T??**?**会议**:](https://github.com/T??**?**会议**:) ECCV 2024  
**arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?*代码**: https?*代码**: https://github.com/TangZJ/3?*领域**: 3??*arXiv**: [2408.03753?*?**??**arXiv**: [2408.03753](https?*arXiv**: [2408.03753](htt??*代码**: https?*代码**: https://github.com/TangZJ/3?*领?)**?*??*?*关键词**: 3D Gaussian Splatting, ?  
## ???色。但这类逆渲染本身是严重的病??## ???色。但这类逆渲染本身是严??3D?%## ???色。但这类逆渲染本身是严重的病??**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: [https://github.com/T??**?**会议**:](https://github.com/T??**?**会议**:) ECCV 2024  
**arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?ilter作用**arXiv**: [2408.03753](htt?*代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?*代码**: https???# ???色。但这类逆渲染本身是严重的病??## ???色。但这类逆渲染本身是严??3D?%## ???色。但这类逆渲染本身是严重的病??**?**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: [https://github.com/T??**?**会议**:](https://github.com/T??**?**会议**:) ECCV 2024?  
**代码**: https?*代码**: [https://github.com/T??**?**会议**:](https://github.com/T??**?**会议**:) ECCV 2024  
**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?ilter作用**arXiv**  

**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?*代码**: https???# ???色。但这类逆渲染本身是严重的病??## ???色。但-|**arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*arXiv**: 0**代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024?  
**代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024--**代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024?3**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](https?*arXiv**: [2408----|  
| 
**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](https?*arXiv**: [2408.03753]F启?*代码**: https???**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*arXiv**: [2408.03753?*作者**: Zhe Jun Tang, Tat-Jen Cham  
**代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*RD**代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*arXiv**: 0**代码**: https?*??**代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024--**代码**: https?*代码**: https://github.com/T??**?**会议**: ECCV 2024?3**arXiv**: [2408.03753](h?*arXiv**: [2408.03753](htt??   
**arXiv**??  
- **张量分解带来效率优势**：光照场用VM分解后非常紧凑（150^3体素），查询只需插值，相比GaussianShader（12x慢）速度优势明显（仅3.2x慢）
- **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*RD**代码**: https?*代?-**代码**: http??照网格，无法直接处理无界场景（如室外大场景），需要场景warping技术
**arXiv**??  
- **张量分解带来效率优势**：光照场用VM分解后非常紧凑（150^3体素），查询只需插值，相比GaussianShader（12x慢）速度优势明显（仅3.2x慢）
- **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*RD**代码**: https?*代?-**代码**: http??照网格，无法直接处理无界场景（如室外大场景），需要场景war?与法线? **张量?? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753](https://arxiv.org???*RD**䨡**arXiv**??
- **张量分解带来效率优势**：光照场用VM分解后非常紧凑（150^3体素），查询只需插值，相比GaussianShader（12x慢）速度优势明显（仅3.2x慢）
- **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.??- **张量? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753lender）略有优势，因为? **张量分解带来效率优势**：光照场用VM分解后非常紧凑（150^3体素），查询只需插值，相比GaussianShader（12x慢）速度优势明显（仅3.2x慢）
- **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.??- **张量? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753lender）略有优势，因为? **张量分解带来? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.??- **张量? **即插即用**：方法与3DGS框架高岾- **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.??- **张量? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.com/T?*arXiv**: [2408.03753lender）略有优势，因为? **张量分解带来? **即插即用**：方法与3DGS框架高??*代码**: https?*代?-**代码**: https?*代码**: https://github.??- **张量? **即插即用**９?，公式推导流畅，图示直观
- 价值: ⭐⭐⭐⭐ 为3DGS视角依赖效果提供了有效且高效的解决方案，有实际应用价值
