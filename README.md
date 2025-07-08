# Pruning for Large Language Models


I currently focus on pruning for large language models including
- [Surveys](#Surveys)
- [Structured Pruning](#Structured-Pruning)
- [Unstructured Pruning](#Unstructured-Pruning)
- [Semi-Structured Pruning](#SSemi-tructured-Pruning)
  
<strong> Last Update: 2025/06/20 </strong>




<a name="Surveys" />

## Surveys 
- [2025] Efficient Compressing and Tuning Methods for Large Language Models: A Systematic Literature Review, ACM  [[Paper](https://doi.org/10.1145/3728636)]
- [2025] A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond, arXiv [[Paper](https://arxiv.org/pdf/2503.21614)] [[Code](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning)]
- [2025] A Survey on Efficient Vision-Language Models, arXiv [[Paper](https://arxiv.org/abs/2504.09724)]
- [2025] Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions, arXiv [[Paper](https://arxiv.org/abs/2503.16585)]
- [2024] A Survey on Model Compression for Large Language Models, TACL [[Paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482)] 
- [2024] Efficient Large Language Models: A Survey, TMLR [[Paper](https://arxiv.org/abs/2312.03863)] [[Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]
- [2024] A Survey of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2303.18223)] [[Code](https://github.com/RUCAIBox/LLMSurvey)]
- [2024] A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/10643325)] [[Code](https://github.com/hrcheng1066/awesome-pruning)]
- [2024] Structured Pruning for Deep Convolutional Neural Networks: A Survey, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/10330640)] [[Code](https://github.com/he-y/Awesome-Pruning)]


 <a name="Structured-Pruning" />
 
## Structured Pruning
- [2025] EvoP: Robust LLM Inference via Evolutionary Pruning, arXiv [[Paper](https://arxiv.org/abs/2502.14910)] 
- [2025] Sample-aware Adaptive Structured Pruning for Large Language Models, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33973)] [[Code]( https://github.com/JunKong5/AdaPruner)]
- [2025] SPAP: Structured Pruning via Alternating Optimization and Penalty Methods, arXiv [[Paper](https://arxiv.org/abs/2505.03373)] 
- [2025] Týr-the-Pruner: Unlocking Accurate 50% Structural Pruning for LLMs via Global Sparsity Distribution Optimization, arXiv [[Paper](https://arxiv.org/abs/2503.09657)] 
- [2025] Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing, ICLR [[Paper](https://arxiv.org/abs/2502.15618)]
- [2025] Lightweight and Post-Training Structured Pruning for On-Device Large Lanaguage Models, arXiv [[Paper](https://arxiv.org/abs/2501.15255)]
- [2025] FASP: Fast and Accurate Structured Pruning of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2501.09412)]
- [2024] FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models, NeurIPS [[Paper](https://openreview.net/forum?id=jrSWzgno4W)] 
- [2024] Fluctuation-Based Adaptive Structured Pruning for Large Language Models, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28960)]
- [2024] A Convex-optimization-based Layer-wise Post-training Pruner for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2408.03728)] 
- [2024] LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models, ICML [[Paper](https://arxiv.org/abs/2404.09695)] [[Code](https://github.com/lihuang258/LoRAP)]
- [2024] SlimGPT: Layer-wise Structured Pruning for Large Language Models, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1c44e46358e0fb94dc94ec495a7fb1a-Abstract-Conference.html)] 
- [2024] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning, ICLR [[Paper](https://openreview.net/forum?id=09iOdaeOzp)] [[Code](https://github.com/princeton-nlp/LLM-Shearing)]
- [2024] Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes, arXiv [[Paper](https://arxiv.org/abs/2402.05406)] 
- [2024] Compact Language Models via Pruning and Knowledge Distillation, arXiv [[Paper](https://www.arxiv.org/abs/2407.14679)] 
- [2024] A Deeper Look at Depth Pruning of LLMs, ICML [[Paper](https://openreview.net/forum?id=9B7ayWclwN)]
- [2024] MoreauPruner: Robust Pruning of Large Language Models against Weight Perturbations, arXiv [[Paper](https://arxiv.org/abs/2406.07017)]  [[Code](https://github.com/usamec/double_sparse)]
- [2024] Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models, arXiv [[Paper](https://arxiv.org/abs/2405.20541)] 
- [2024] Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models, ICLR [[Paper](https://openreview.net/forum?id=Tr0lPx9woF)] 
- [2024] BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation, arXiv [[Paper](https://arxiv.org/abs/2402.16880)]
- [2024] ShortGPT: Layers in Large Language Models are More Redundant Than You Expect, arXiv [[Paper](https://arxiv.org/abs/2403.03853)] 
- [2024] NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2402.09773)] 
- [2024] SliceGPT: Compress Large Language Models by Deleting Rows and Columns, ICLR[[Paper](https://arxiv.org/abs/2401.15024)] [[Code](https://github.com/microsoft/TransformerCompression?utm_source=catalyzex.com)]
- [2023] LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery, arXiv [[Paper](https://arxiv.org/abs/2310.18356)]
- [2023] LLM-Pruner: On the Structural Pruning of Large Language Models, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html)] [[Code](https://github.com/horseee/LLM-Pruner)]
- [2023] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning, NeurIPS [[Paper](https://arxiv.org/abs/2310.06694)] [[Code](https://github.com/princeton-nlp/LLM-Shearing)]
- [2023] LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, arXiv [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- [2023] LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, ICML [[Paper](https://proceedings.mlr.press/v202/li23ap.html)] [[Code](https://github.com/yxli2123/LoSparse)]
- [2019] Importance Estimation for Neural Network Pruning, CVPR  [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)]
- [2016] Learning Structured Sparsity in Deep Neural Networks, NIPS  [[Paper](https://proceedings.neurips.cc/paper_files/paper/2016/hash/41bfd20a38bb1b0bec75acf0845530a7-Abstract.html)] [[Code](https://github.com/wenwei202/caffe/tree/scnn)]
- [2015] Learning both Weights and Connections for Efficient Neural Network, NIPS  [[Paper](https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html)]


<a name="Unstructured-Pruning" />

## Unstructured Pruning
- [2025] Wanda++: Pruning Large Language Models via Regional Gradients, ACL [[Paper](https://arxiv.org/abs/2503.04992)]
- [2025] Dynamic Superblock Pruning for Fast Learned Sparse Retrieval, SIGIR [[Paper](https://arxiv.org/abs/2504.17045)]  [[Code](https://github.com/thefxperson/hierarchical_pruning)]
- [2025] Two Sparse Matrices are Better than One: Sparsifying Neural Networks with Double Sparse Factorization, ICLR [[Paper](https://openreview.net/forum?id=DwiwOcK1B7)]  [[Code](https://github.com/usamec/double_sparse)]
- [2024] Fast and Effective Weight Update for Pruned Large Language Models, TMLR [[Paper](https://openreview.net/forum?id=1hcpXd9Jir)] [[Code](https://github.com/fmfi-compbio/admm-pruning)]
- [2024] A Simple and Effective Pruning Approach for Large Language Models, ICLR [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- [2024] Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs, ICLR [[Paper](https://openreview.net/forum?id=1ndDmZdT4g)]  [[Code](https://github.com/zyxxmu/DSnoT)]
- [2024] Pruner-Zero: Evolving Symbolic Pruning Metric From Scratch for Large Language Models, ICML [[Paper](https://openreview.net/forum?id=1tRLxQzdep)] 
- [2024] Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs, ICLR [[Paper](https://arxiv.org/abs/2310.08915)] 
- [2024] A Convex-optimization-based Layer-wise Post-training Pruner for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2408.03728)]
- [2023] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot, ICML [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- [2023] One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models, arXiv [[Paper](https://arxiv.org/pdf/2310.09499v1.pdf)]
- [2023] Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity, VLDB [[Paper](https://dl.acm.org/doi/abs/10.14778/3626292.3626303)]


<a name="Semi-Structured-Pruning" />

## Semi-Structured Pruning
- [2025] Progressive Binarization with Semi-Structured Pruning for LLMs, arXiv [[Paper](https://arxiv.org/abs/2502.01705)] [[Code](https://github.com/XIANGLONGYAN/PBS2P)]
- [2025] Pruning Large Language Models with Semi-Structural Adaptive Sparse Training, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34592)]
- [2024] ADMM Based Semi-Structured Pattern Pruning Framework for Transformer, AIoTC [[Paper](https://ieeexplore.ieee.org/abstract/document/10748287)]
- [2024] Dependency-Aware Semi-Structured Sparsity of GLU Variants in Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2405.01943)]
- [2024] MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models, NeurIPS [[Paper](https://openreview.net/forum?id=Llu9nJal7b)]  [[Code](https://github.com/NVlabs/MaskLLM)]
- [2024] LPViT: Low-Power Semi-structured Pruning for Vision Transformers, ECCV [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73209-6_16)]
- [2023] E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity, arXiv [[Paper](https://arxiv.org/abs/2310.15929)]
- [2021] NxMTransformer: Semi-Structured Sparsification for Natural Language Understanding via ADMM, NeurIPS [[Paper](https://proceedings.neurips.cc/paper/2021/hash/0e4f5cc9f4f3f7f1651a6b9f9214e5b1-Abstract.html)]

