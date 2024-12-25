# Welcome to Modalbed

ModalBed is a PyTorch-based framework designed to facilitate reproducible and solid research in modality generalization, as introduced in [Towards Modality Generalization: A Benchmark and Prospective Analysis](https://arxiv.org/pdf/2412.18277).


_The complete code is coming soon!_

### Continual Update!
ModalBed is an ongoing project that will be continually updated with new results, algorithms, and datasets. Contributions from fellow researchers through pull requests are highly encouraged and welcomed :).

### Available Algorithms

- Feature Concatenation (Concat)
- On-the-fly Gradient Modulation ([OGM](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf))
- Dynamically Learning Modality Gap ([DLMG](https://openreview.net/pdf?id=QbsPz0SnyV))
- Empirical Risk Minimization ([ERM](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
- Inter-domain Mixup ([Mixup](https://arxiv.org/abs/2001.00677))
- Class-conditional DANN ([CDANN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf))
- Style Agnostic Networks ([SagNet](https://arxiv.org/abs/1910.11645))
- Information Bottleneck ([IB_ERM](https://arxiv.org/abs/2106.06607))
- Conditional Contrastive Adversarial Domain ([CondCAD](https://arxiv.org/abs/2201.00057))
- Empirical Quantile Risk Minimization ([EQRM](https://arxiv.org/abs/2207.09944))

### Available Perceptors

- ImageBind ([paper](https://facebookresearch.github.io/ImageBind/paper), [codebase](https://github.com/facebookresearch/ImageBind))
- LanguageBind ([paper](https://arxiv.org/abs/2310.01852), [codebase](https://github.com/PKU-YuanGroup/LanguageBind?tab=readme-ov-file))
- UniBind ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Lyu_UniBind_LLM-Augmented_Unified_and_Balanced_Representation_Space_to_Bind_Them_CVPR_2024_paper.pdf), [codebase](https://github.com/QC-LY/UniBind))

### Available Datasets
- MSR-VTT: MSR-VTT: A Large Video Description Dataset for Bridging Video and Language
- NYUDv2: Indoor Segmentation and Support Inference from RGBD Images
- VGGSound: VGGSound: A Large-scale Audio-Visual Dataset

### Acknowledgement
- [DomainBed](https://github.com/facebookresearch/DomainBed), a suite to test domain generalization algorithms

### Citing ModalBed
If you find this repository useful, please consider giving a star ⭐ and citation
```
@misc{liu2024modalbed,
      title={Towards Modality Generalization: A Benchmark and Prospective Analysis}, 
      author={Xiaohao Liu and Xiaobo Xia and Zhuo Huang and Tat-Seng Chua},
      year={2024},
      eprint={2412.18277},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
