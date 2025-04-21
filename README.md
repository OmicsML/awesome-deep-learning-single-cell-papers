[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/OmicsML/awesome-deep-learning-single-cell-papers/pulls)
# awesome-deep-learning-single-cell-papers

This repository keeps track of the latest papers on single-cell analysis with deep learning methods. We categorize them based on individual tasks.

We will try to make this list updated. If you find any error or any missed paper, please don't hesitate to open an issue or pull request.

## Citation

Be free to refer to our comprehensive survey paper on Deep Learning in Single-cell Analysis:

```bibtex
@article{molho2024deep,
  title={Deep learning in single-cell analysis},
  author={Molho, Dylan and Ding, Jiayuan and Tang, Wenzhuo and Li, Zhaoheng and Wen, Hongzhi and Wang, Yixin and Venegas, Julian and Jin, Wei and Liu, Renming and Su, Runze and others},
  journal={ACM Transactions on Intelligent Systems and Technology},
  volume={15},
  number={3},
  pages={1--62},
  year={2024},
  publisher={ACM New York, NY}
}
```


## For the foundation model for single-cell, more papers are recorded [[HERE]](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers).

- [Book](#book)
- [Single Cell Technology](#single-cell-techonoly)
- [Course](#course)
- [Survey](#survey)
- [Pretrained Model or LLM or Foundation Model](#pretrained-model-or-llm-or-foundation-model)
- [GAN or Diffusion Model](#gan-or-diffusion-model)
- [Multimodal Learning](#multimodal-learning)
- [Single Cell Data Simulation](#single-cell-data-simulation)
- [Interpretability](#interpretability)
- [Spatialtemporal Transcriptomic](#spatialtemporal-transcriptomic)
- [RNA Velocity](#rna-velocity)
- [Molecular Representation Learning](#molecular-representation-learning)
- [Single Cell Perturbation or Drug Response](#single-cell-perturbation-or-drug-response)
- [Cellular Dynamics](#cellular-dynamics)
- [Single Cell Application](#single-cell-application)
- [Tools For Single Cell or Spatial Data](#tools-for-single-cell-or-spatial-data)
- [Single Cell Atlas](#single-cell-atlas)
- [Single Cell Visualization](#single-cell-visualization)
- [Benchmarking](#benchmarking)
- [Metric Design](#metric-design)
- [Subcellular Analysis](#subcellular-analysis)
- [Dimensionality Reduction and Visualization](#dimensionality-reduction-and-visualization)
- [Representation Learning](#representation-learning)
- [Batch Effect Correction](#batch-effect-correction)
- [Tumor Microenvironment-TME](#tumor-microenvironment-tme)
- [Cell-Cell Communication Events](#cell-cell-communication-events)
- [Gene Regulatory Network](#gene-regulatory-network)
- [Imputation](#imputation)
- [Spatial Domain](#spatial-domain)
- [Reference Embedding or Transfer Learning](#reference-embedding-or-transfer-learning)
- [Cell Segmentation](#cell-segmentation)
- [Cell Type Deconvolution](#cell-type-deconvolution)
- [Cell Type Annotation](#cell-type-annotation)
- [Cell Clustering](#cell-clustering)
- [Disease Prediction](#disease-prediction)
- [Multimodal Integration](#multimodal-integration)
- [Multiomics Translation](#multiomics-translation)

## Book
1. [[Single Cell Best Practices]](https://www.sc-best-practices.org/preamble.html), Fabian Theis's Lab
1. [[Basics of Single-Cell Analysis with Bioconductor]](http://bioconductor.org/books/3.15/OSCA.basic/index.html), Bioconductor software based on R

## Single Cell Technology
### Single-Modality

### Multimodality

### Spatial Transcriptomic
1. [2022 Nature Methods] **Museum of spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41592-022-01409-2)

## Course
1. [[CSCI 1850 Deep Learning in Genomics]](https://cs.brown.edu/courses/csci1850/index.html), Brown University
1. [[Machine Learning in Genomics: Dissecting Human Disease Circuitry]](http://stellar.mit.edu/S/course/6/fa21/6.047/materials.html), MIT
1. [[ANALYSIS OF SINGLE CELL RNA-SEQ DATA]](https://broadinstitute.github.io/2019_scWorkshop/index.html), course by Orr Ashenberg, Dana Silverbush, Kirk Gosik
1. [[Analysis of single cell RNA-seq data, www.singlecellcourse.org]](https://scrnaseq-course.cog.sanger.ac.uk/website/index.html) - step-by-step scRNA-seq analysis course. R-based, with code examples, explanations, exercises. From alignment (STAR) and QC (FASTQC) to introduction to R, SingleCellExperiment class, `scater` object, data exploration (reads, UMI), filtering, normalization (`scran`), batch effect removal (`RUV`, `ComBat`, `mnnCorrect`, GLM, `Harmony`), clustering and marker gene identification (`SINCERA`, `SC3`, tSNE, `Seurat`), feature selection (`M3Drop::M3DropConvertData`, `BrenneckeGetVariableGenes`), pseudotime analysis (`TSCAN`, `Monocle`, diffusion maps, `SLICER`, `Ouija`, `destiny`), imputation (`scImpute`, `DrImpute`, `MAGIC`), differential expression (Kolmogorov-Smirnov, Wilcoxon, `edgeR`, `Monocle`, `MAST`), data integration (`scmap`, cell-to-cell mapping, `Metaneighbour`, `mnnCorrect`, `Seurat`'s canonical correllation analysis). Search for scRNA-seq data ([scfind](https://github.com/hemberg-lab/scfind) R package), as well as [Hemberg group’s public datasets](https://hemberg-lab.github.io/scRNA.seq.datasets/). [Seurat chapter](https://scrnaseq-course.cog.sanger.ac.uk/website/seurat-chapter.html). ["Ideal" scRNA-seq pipeline](https://scrnaseq-course.cog.sanger.ac.uk/website/ideal-scrnaseq-pipeline-as-of-oct-2017.html). [Video lectures](https://www.youtube.com/watch?list=PLEyKDyF1qdOYAhwU71qlrOXYsYHtyIu8n&v=56n77bpjiKo). <details>
    <summary>Paper</summary>
    Andrews, Tallulah S., Vladimir Yu Kiselev, Davis McCarthy, and Martin Hemberg. "Tutorial: Guidelines for the Computational Analysis of Single-Cell RNA Sequencing Data." https://doi.org/10.1038/s41596-020-00409-w Nature Protocols, December 7, 2020. 
</details>

## Survey
1. [2023 Biophysics Reviews] **Deep learning in spatial transcriptomics: Learning from the next next-generation sequencing** [[paper]](https://pubs.aip.org/aip/bpr/article/4/1/011306/2879089/Deep-learning-in-spatial-transcriptomics-Learning)


## Pretrained Model or LLM or Foundation Model
**Refer more details to** [[foundation-model-single-cell-papers]](https://github.com/OmicsML/foundation-model-single-cell-papers)
1. [2024 BioRxiv] **scPRINT: pre-training on 50 million cells allows robust gene network predictions** [[paper](https://www.biorxiv.org/content/10.1101/2024.07.29.605556v1)]
1. [2024 ICLR] **BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs** [[paper]](https://openreview.net/forum?id=jJCeMiwHdH)
1. [2023 bioRxiv] **CellPLM: Pre-training of Cell Language Model Beyond Single Cells** [[paper]](https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1)
1. [2023 bioRxiv] **DNABERT-2: EFFICIENT FOUNDATION MODEL AND BENCHMARK FOR MULTI-SPECIES GENOME** [[paper]](https://arxiv.org/pdf/2306.15006.pdf)
1. [2023 bioRxiv] **The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4** [[paper]](https://arxiv.org/abs/2311.07361)
1. [2023 bioRxiv] **Augmenting large language models with chemistry tools** [[paper]](https://arxiv.org/pdf/2304.05376.pdf)
1. [2023 bioRxiv] **GET: a foundation model of transcription across human cell types** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.24.559168v1)
1. [2023 bioRxiv] **Cell2Sentence: Teaching Large Language Models the Language of Biology** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.11.557287v1)
1. [2023 bioRxiv] **Evaluating the Utilities of Large Language Models in Single-cell Data Analysis** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.08.555192v2)
1. [2023 arxiv] **Towards Generalist Biomedical AI** [[paper]](https://arxiv.org/abs/2307.14334)
1. [2023 bioRxiv] **Contextualizing protein representations using deep learning on protein networks and single-cell data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.07.18.549602v1)
1. [2023 Nature] **Large language models encode clinical knowledge** [[paper]](https://www.nature.com/articles/s41586-023-06291-2)
1. [2023 Nature Methods] **Towards foundation models of biological image segmentation** [[paper]](https://www.nature.com/articles/s41592-023-01885-0)
1. [2023 bioRxiv] **DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins** [[paper]](https://www.biorxiv.org/content/10.1101/2023.06.29.543848v1)
1. [2023 arxiv] **Hyena Hierarchy: Towards Larger Convolutional Language Models** [[paper]](https://arxiv.org/abs/2302.10866)
1. [2023 bioRxiv] **Population-level integration of single-cell datasets enables multi-scale analysis across samples** [[paper]](https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1)
1. [2023 bioRxiv] **Large Scale Foundation Model on Single-cell Transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705v2)
1. [2023 Bioinformatics] **Applications of transformer-based language models in bioinformatics: a survey** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36845200/)
1. [2023 Nature] **Transfer learning enables predictions in network biology** [[paper]](https://www.nature.com/articles/s41586-023-06139-9)
1. [2023 arxiv] **BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks** [[paper]](https://arxiv.org/abs/2305.17100)
1. [2023 arxiv] **Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding** [[paper]](https://arxiv.org/abs/2305.12031)
1. [2023 arxiv] **CancerGPT: Few-shot Drug Pair Synergy Prediction using Large Pre-trained Language Models** [[paper]](https://arxiv.org/abs/2304.10946)
1. [2023 iSchience tGPT] **Generative pretraining from large-scale transcriptomes for single-cell deciphering** [[paper]](https://www.sciencedirect.com/science/article/pii/S2589004223006132)
1. [2023 bioRxiv] **GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information** [[paper]](https://arxiv.org/abs/2304.09667)
1. [2023 Github] **OpenBioMed** [[Github]](https://github.com/BioFM/OpenBioMed)
1. [2023 blog] **BioMedLM: a Domain-Specific Large Language Model for Biomedical Text** [[blog]](https://www.mosaicml.com/blog/introducing-pubmed-gpt)
1. [2023 bioRxiv] **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI** [[paper]](https://www.biorxiv.org/content/10.1101/2023.04.30.538439v1)
1. [2023 bioRxiv] **xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.03.24.534055v1)
1. [2023 Nature Biotechnology] **Large language models generate functional protein sequences across diverse families** [[paper]](https://www.nature.com/articles/s41587-022-01618-2)
1. [2022 arxiv] **A single-cell gene expression language model** [[paper]](https://arxiv.org/abs/2210.14330)
1. [2022 Briefings in Bioinformatics] **BioGPT: generative pre-trained transformer for biomedical text generation and mining** [[paper]](https://academic.oup.com/bib/article/23/6/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9&login=true)
1. [2022 Nature Machine Intelligence] **scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data** [[paper]](https://www.nature.com/articles/s42256-022-00534-z)
1. [2022 bioRxiv] **scFormer: a universal representation learning approach for single-cell data using transformers** [[paper]](https://openreview.net/pdf?id=7hdmA0qtr5)
1. [2022 Bioinformatics] **scPretrain: multi-task self-supervised learning for cell-type classification** [[paper]](https://academic.oup.com/bioinformatics/article/38/6/1607/6499287)
1. [2021 PNAS] **Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences** [[paper]](https://www.pnas.org/doi/10.1073/pnas.2016239118)
1. [2021 Bioinformatics] **DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome** [[paper]](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
1. [2021 Arxiv, 576 citations] **Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing** [[paper]](https://arxiv.org/pdf/2004.10964.pdf)
1. [2021 Arxiv, 1111 citations] **Don't Stop Pretraining: Adapt Language Models to Domains and Tasks** [[paper]](https://arxiv.org/pdf/2004.10964.pdf)

## GAN or Diffusion Model
1. [2024 Brief Bioinform] **stDiff: a diffusion model for imputing spatial transcriptomics through single-cell transcriptomics** [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11021815/)
1. [2024 biorxiv] **scDiffEq: drift-diffusion modeling of single-cell dynamics with neural stochastic differential equations** [[paper]](https://www.biorxiv.org/content/10.1101/2023.12.06.570508v1)
1. [2024 biorxiv] **scDiffusion: conditional generation of high-quality single-cell data using diffusion model** [[paper]](https://arxiv.org/abs/2401.03968)
1. [2024 biorxiv] **In Silico Generation of Gene Expression profiles using Diffusion Models** [[paper]](https://www.biorxiv.org/content/10.1101/2024.04.10.588825v1)
1. [2024 Cell] **A programmable reaction-diffusion system for spatiotemporal cell signaling circuit design** [[paper]](https://www.cell.com/cell/pdf/S0092-8674(23)01339-9.pdf)
1. [2023 ICCV] **Scalable Diffusion Models with Transformers** [[paper]](https://arxiv.org/abs/2212.09748)
1. [2023 biorxiv] **From Noise to Knowledge: Probabilistic Diffusion-Based Neural Inference of Gene Regulatory Networks** [[paper]](https://www.biorxiv.org/content/10.1101/2023.11.05.565675v1)
1. [2023 biorxiv Diffusion] **A General Single-Cell Analysis Framework via Conditional Diffusion Generative Models** [[paper]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=7lwkXGEAAAAJ&citation_for_view=7lwkXGEAAAAJ:Se3iqnhoufwC)
1. [2023 biorxiv GAN] **Predicting cell morphological responses to perturbations using generative modeling** [[paper]](https://www.biorxiv.org/content/10.1101/2023.07.17.549216v1)
1. [2023 Nature Diffusion Model] **AI tools are designing entirely new proteins that could transform medicine** [[paper]](https://www.nature.com/articles/d41586-023-02227-y)
1. [2023 biorxiv Diffusion Model] **The Power of Two: integrating deep diffusion models and variational autoencoders for single-cell transcriptomics analysis** [[paper]](https://www.biorxiv.org/content/10.1101/2023.04.13.536789v1)
1. [2023 biorxiv GAN] **Scalable Integration of Multiomic Single Cell Data Using Generative Adversarial Networks** [[paper]](https://www.biorxiv.org/content/10.1101/2023.06.26.546547v2)
1. [2023 biorxiv Diffusion Model] **Spontanously breaking of symmetry in overlapping cell instance segmentation using diffusion models** [[paper]](https://www.biorxiv.org/content/10.1101/2023.07.07.548066v1)
        

## Multimodal Learning
1. [2024 ICLR workshop NLP+Gene Expression] **Joint Embedding of Transcriptomes and Text Enables Interactive Single-Cell RNA-seq Data Exploration via Natural Language** [[paper]](https://openreview.net/forum?id=yWiZaE4k3K)
1. [2024 Nature Biotechnology Image+Gene Expression] **Inferring super-resolution tissue architecture by integrating spatial transcriptomics with histology** [[paper]](https://www.nature.com/articles/s41587-023-02019-9#citeas)
1. [2023 arxiv Image+Gene Expression] **Transformer with Convolution and Graph-Node co-embedding: An accurate and interpretable vision backbone for predicting gene expressions from local histopathological image** [[paper]](https://www.biorxiv.org/content/10.1101/2023.05.28.542669v1)
1. [2023 arxiv multimodal] **MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data** [[paper]](https://arxiv.org/abs/2310.02275)
1. [2023 biorxiv multimodal] **Pathformer: biological pathway informed Transformer model integrating multi-modal data of cancer** [[paper]](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v1)
1. [2023 biorxiv Image+Gene Expression] **Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning** [[paper]](https://arxiv.org/pdf/2306.01859.pdf)
1. [2023 biorxiv Image+Gene Expression] **Single-cell gene expression prediction using H&E images based on spatial transcriptomics** [[paper]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12471/1247105/Single-cell-gene-expression-prediction-using-HE-images-based-on/10.1117/12.2654294.full?SSO=1)

## Single Cell Data Simulation
1. [2025 NM] **scMultiSim: simulation of single-cell multi-omics and spatial data guided by gene regulatory networks and cell–cell interactions** [[paper]](https://www.nature.com/articles/s41592-025-02651-0)
1. [2023 NBT] **scDesign3 generates realistic in silico data for multimodal single-cell and spatial omics** [[paper]](https://www.nature.com/articles/s41587-023-01772-1)
1. [2023 NC] **scReadSim: a single-cell RNA-seq and ATAC-seq read simulator** [[paper]](https://www.nature.com/articles/s41467-023-43162-w)
1. [2023 biorxiv] **GRouNdGAN: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks** [[paper]](https://www.biorxiv.org/content/10.1101/2023.07.25.550225v1)
1. [2022 JCB] **Simulating Single-Cell Gene Expression Count Data with Preserved Gene Correlations by scDesign2** [[paper]](https://www.liebertpub.com/doi/abs/10.1089/cmb.2021.0440)
1. [2021 GB] **scDesign2: a transparent simulator that generates high-fidelity single-cell gene expression count data with gene correlations captured** [[paper]](https://link.springer.com/article/10.1186/s13059-021-02367-2)
1. [2019 Bioinformatics] **A statistical simulator scDesign for rational scRNA-seq experimental design** [[paper]](https://academic.oup.com/bioinformatics/article/35/14/i41/5529133)



## Interpretability
1. [2021 CVPR] **Transformer Interpretability Beyond Attention Visualization** [[paper]](https://arxiv.org/abs/2012.09838)[[github]](https://github.com/hila-chefer/Transformer-Explainability)
1. [2021 ICML] **BERTology Meets Biology: Interpreting Attention in Protein Language Models** [[paper]](https://openreview.net/pdf?id=YWtLZvLmud7)
1. [2019 ACL] **A Multiscale Visualization of Attention in the Transformer Model** [[paper]](https://arxiv.org/pdf/1906.05714.pdf) [[github]](https://github.com/jessevig/bertviz/tree/master)


## Spatialtemporal Transcriptomic
1. [2024 biorxiv] **Gene Trajectory Inference for Single-cell Data by Optimal Transport Metrics** [[paper]](https://www.biorxiv.org/content/10.1101/2022.07.08.499404v3)
1. [2023 biorxiv] **Uncovering developmental time and tempo using deep learning** [[paper]](https://www.nature.com/articles/s41592-023-02083-8)
1. [2023 biorxiv] **scNODE: Generative Model for Temporal Single Cell Transcriptomic Data Prediction** [[paper]](https://www.biorxiv.org/content/10.1101/2023.11.22.568346v1.full.pdf)
1. [2023 biorxiv] **Gene Trajectory Inference for Single-cell Data by Optimal Transport Metrics** [[paper]](https://www.biorxiv.org/content/10.1101/2022.07.08.499404v3)
1. [2023 arxiv survey from CS field] **Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook** [[paper]](https://arxiv.org/abs/2310.10196)
1. [2023 ICML Reference from CS field] **Continuous Spatiotemporal Transformers** [[paper]](https://arxiv.org/abs/2301.13338)
1. [2023 arxiv multimodalities Reference from CS field] **IMAGEBIND: One Embedding Space To Bind Them All** [[paper]](https://arxiv.org/pdf/2305.05665.pdf)
1. [2023 arxiv multimodalities Reference from CS field] **UnIVAL: Unified Model for Image, Video, Audio and Language Tasks** [[paper]](https://arxiv.org/pdf/2307.16184.pdf)
1. [2023 arxiv multimodalities Reference from CS field] **Meta-Transformer: A Unified Framework for Multimodal Learning** [[Meta-Transformer paper]](https://arxiv.org/pdf/2307.10802.pdf)[[viT vision Transformer paper]](https://arxiv.org/abs/2010.11929)[[ImageGPT paper Generative Pretraining From Pixels]](https://proceedings.mlr.press/v119/chen20s.html)
1. [2023 KDD Reference from CS field] **Spatio-temporal Diffusion Point Processes** [[paper]](https://arxiv.org/pdf/2305.12403.pdf)
1. [2023 arxiv Reference from CS field] **Long-Range Transformers for Dynamic Spatiotemporal Forecasting** [[paper]](https://arxiv.org/pdf/2109.12218.pdf)
1. [2023 Nature Communications] **Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions** [[paper]](https://www.nature.com/articles/s41467-021-23518-w)
1. [2023 bioRxiv] **Mapping cells through time and space with moscot** [[paper]](https://www.biorxiv.org/content/10.1101/2023.05.11.540374v1)        
1. [2023 Nature Methods] **Spatiotemporally resolved transcriptomics reveals the subcellular RNA kinetic landscape** [[paper]](https://www.nature.com/articles/s41592-023-01829-8)
1. [2022 bioRxiv] **Spateo: multidimensional spatiotemporal modeling of single-cell spatial transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2022.12.07.519417v1.abstract)
1. [2022 ICLR Reference from CS field] **UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning** [[paper]](https://arxiv.org/abs/2201.04676)
1. [2022 NeurIPS spatial-temporal single-cell -> spatial-temporal video] **Flamingo: a Visual Language Model for Few-Shot Learning** [[paper]](https://arxiv.org/abs/2204.14198)
1. [2022 arxiv, image-gene expression contrastive learning] **CoCa: Contrastive Captioners are Image-Text Foundation Models** [[paper]](https://arxiv.org/pdf/2205.01917.pdf)
1. [2020 ICLR, image-gene expression pretraining] **VL-BERT: Pre-training of Generic Visual-Linguistic Representations** [[paper]](https://arxiv.org/pdf/1908.08530.pdf)
1. [2019 AAAI, image-gene expression pretraining] **Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training** [[paper]](https://arxiv.org/pdf/1908.06066.pdf)

## RNA Velocity
1. [2023 Nature Methods] **Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells** [[paper]](https://www.nature.com/articles/s41592-023-01994-w)



## Molecular Representation Learning
1. [2023 ICLR] **Uni-Mol: A Universal 3D Molecular Representation Learning Framework** [[paper]](https://openreview.net/forum?id=6K2RM6wVqKu)

## Single Cell Perturbation or Drug Response
1. [2024 NeurIPS AIDrugX (Spotlight)] **Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for Therapeutics** [[paper]](https://openreview.net/forum?id=kL8dlYp6IM) [[poster]](https://drive.google.com/file/d/1plypydZCaegbgxyCl-xehFxSgwX6e8So/view)
2. [2024 biorxiv] **Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear methods** [[paper]](https://www.biorxiv.org/content/biorxiv/early/2024/09/21/2024.09.16.613342.full.pdf)
1. [2024 ICLR] **Biologically Interpretable VAE with Supervision for Transcriptomics Data Under Ordinal Perturbations** [[paper]](https://www.nature.com/articles/s41592-023-02144-y)
1. [2024 Nature Methods] **scPerturb: harmonized single-cell perturbation data** [[paper]](https://www.nature.com/articles/s41592-023-02144-y)
1. [2023 biorxiv] **Unagi: Deep Generative Model for Deciphering Cellular Dynamics and In-Silico Drug Discovery in Complex Diseases** [[paper]](https://assets.researchsquare.com/files/rs-3676579/v1_covered_2dc4a452-a1f2-47a2-acb1-f816276a9e07.pdf?c=1702865288)
1. [2023 NeurIPS] **Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder** [[paper]](https://openreview.net/forum?id=DzaCE00jGV)
1. [2023 Nature Methods] **Learning single-cell perturbation responses using neural optimal transport** [[paper]](https://www.nature.com/articles/s41592-023-01969-x)
1. [2023 Nature Methods] **Neural optimal transport predicts perturbation responses at the single-cell level** [[paper]](https://www.nature.com/articles/s41592-023-01968-y)
1. [2023 Mol Syst Biol] **Predicting cellular responses to complex perturbations in high-throughput screens** [[paper]](https://pubmed.ncbi.nlm.nih.gov/37154091/)
1. [2023 biorxiv] **Learning Perturbation-specific Cell Representations for Prediction of Transcriptional Response across Cellular Contexts** [[paper]](https://www.biorxiv.org/content/10.1101/2023.03.20.533433v1)
1. [2023 Nature] **Dissecting cell identity via network inference and in silico gene perturbation** [[paper]](https://www.nature.com/articles/s41586-022-05688-9)
1. [2023 biorxiv Diffusion Model] **The Power of Two: integrating deep diffusion models and variational autoencoders for single-cell transcriptomics analysis** [[paper]](https://www.biorxiv.org/content/10.1101/2023.04.13.536789v1)
1. [2023 ICLR] **Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information** [[paper]](https://openreview.net/forum?id=ICYasJBlZNs)
1. [2022 arxiv] **PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations** [[paper]](https://www.biorxiv.org/content/10.1101/2022.07.20.500854v2)
1. [2022 arxiv] **CausalBench: A Large-scale Benchmark for Network Inference from Single-cell Perturbation Data** [[paper]](https://arxiv.org/abs/2210.17283)
1. [2022 NeurIPS] **Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution** [[paper]](https://arxiv.org/pdf/2204.13545.pdf)
1. [2022 biorxiv] **GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations** [[paper]](https://www.biorxiv.org/content/10.1101/2022.07.12.499735v2)
1. [2021 biorxiv] **Learning interpretable cellular responses to complex perturbations in high-throughput screens** [[paper]](https://www.biorxiv.org/content/10.1101/2021.04.14.439903v2)
1. [2019 Nature Methods] **scGen predicts single-cell perturbation responses** [[paper]](https://www.nature.com/articles/s41592-019-0494-8)


## Cellular Dynamics
1. [2023 Genome Biology] **scTour: a deep learning architecture for robust inference and accurate prediction of cellular dynamics** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02988-9)
        
        

## Single Cell Application
1. [2023 medrxiv] **Single-cell RNA sequencing of human tissue supports successful drug targets** [[paper]](https://www.medrxiv.org/content/10.1101/2024.04.04.24305313v1)
1. [2023 Nature Methods] **Machine learning in rare disease** [[paper]](https://www.nature.com/articles/s41592-023-01886-z)
1. [2023 Molecular System Biology] **Single-cell biology: what does the future hold?** [[paper]](https://www.embopress.org/doi/full/10.15252/msb.202311799)
1. [2023 Genes] **Single-Cell Analysis in the Omics Era: Technologies and Applications in Cancer** [[paper]](https://www.mdpi.com/2073-4425/14/7/1330)
1. [2023 Nature Communications] **ASGARD is A Single-cell Guided Pipeline to Aid Repurposing of Drugs** [[paper]](https://www.nature.com/articles/s41467-023-36637-3)
1. [2023 Nature Reviews Clinical Oncology] **Advancing CAR T cell therapy through the use of multidimensional omics data** [[paper]](https://www.nature.com/articles/s41571-023-00729-2)

## Tools For Single Cell or Spatial Data
[[Tool Summary]](https://www.scrna-tools.org/tools)
1. [2024 biorxiv] **Scvi-hub: an actionable repository for model-driven single cell analysis** [[paper]](https://www.biorxiv.org/content/10.1101/2024.03.01.582887v1.full.pdf)
1. [2024 Nature Methods] **SpatialData: an open and universal data framework for spatial omics** [[paper]](https://www.nature.com/articles/s41592-024-02212-x)
1. [2023 Nucleic Acids Research] **DeepBIO: an automated and interpretable deep-learning platform for high-throughput biological sequence prediction, functional annotation and visualization analysis** [[paper]](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkad055/7041952)
1. [2023 Github] **SpatialTis: an ultra-fast spatial analysis toolkit for large-scale spatial single-cell data.** [[github]](https://github.com/Mr-Milk/SpatialTis)
1. [2023 biorxiv] **CellContrast: Reconstructing Spatial Relationships in Single-Cell RNA Sequencing Data via Deep Contrastive Learning** [[paper]](https://www.biorxiv.org/content/10.1101/2023.10.12.562026v1)


## Single Cell Atlas
1. [2023 Nature] **A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain** [[paper]](https://www.nature.com/articles/s41586-023-06812-z)
1. [2023 Nature] **A spatially resolved single-cell genomic atlas of the adult human breast** [[paper]](https://www.nature.com/articles/s41586-023-06252-9)
1. [2023 Nature Medicine] **An integrated cell atlas of the lung in health and disease** [[paper]](https://www.nature.com/articles/s41591-023-02327-2)
1. [2022 Nucleic Acids Research] **Aquila: a spatial omics database and analysis platform** [[paper]](https://academic.oup.com/nar/article/51/D1/D827/6761736)
1. [[Cellxgene Datasets: 546 datasets by 2022]](https://cellxgene.cziscience.com/datasets)
1. [2022 Nature Methods] **Benchmarking atlas-level data integration in single-cell genomics** [[paper]](https://www.nature.com/articles/s41592-021-01336-8.pdf)
1. [2022 bioRxiv] **A unified analysis of atlas single cell data** [[paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503038v1.full.pdf)
1. [2022 Nature Biotechnology] **Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis** [[paper]](https://www.nature.com/articles/s41587-021-01006-2)
1. [2022 bioRxiv] **Supervised spatial inference of dissociated single-cell data with SageNet** [[paper]](https://www.biorxiv.org/content/10.1101/2022.04.14.488419v1)  
1. [2022 Nature Communications] **Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space** [[paper]](https://www.nature.com/articles/s41467-022-33758-z)


## Single Cell Visualization
1. [[Chanzuckerberg: An interactive explorer for single-cell transcriptomics data]](https://github.com/chanzuckerberg/cellxgene)
1. [[UCSC Cell Browser]](http://cells.ucsc.edu/)
1. [[Cytoscape]](https://cytoscape.org/)
1. [[UCSC Xena]](https://xena.ucsc.edu/)
1. [[ASAP: Automated Single-cell Analysis Pipeline]](https://asap.epfl.ch/)
1. [[GenePattern]](https://notebook.genepattern.org/)
1. [[Loopy Browser]](https://loopybrowser.com/)

## Benchmarking
1. [2024 MoML@Mila] **** [[CMT submission]](https://cmt3.research.microsoft.com/MoML2024/Submission/Summary/13) [[preprint]](https://www.biorxiv.org/content/10.1101/2024.06.12.598655v2) [[poster]](https://drive.google.com/file/d/1LYdITeFY5iX07zyXPGVEjMpYjuHMrneS/view) [[conference]](https://portal.ml4dd.com/moml-2024)
2. [2023 biorxiv] **Benchmarking the translational potential of spatial gene expression prediction from histology** [[paper]](https://www.biorxiv.org/content/10.1101/2023.12.12.571251v1)
1. [2023 bioRxiv] **Systematic benchmarking of imaging spatial transcriptomics platforms in FFPE tissues** [[paper]](https://www.biorxiv.org/content/10.1101/2023.12.07.570603v1)
1. [2023 bioRxiv] **Benchmarking multi-omics integration algorithms across single-cell RNA and ATAC data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.11.15.564963v1)
1. [2023 bioRxiv] **BEND: Benchmarking DNA Language Models on biologically meaningful tasks** [[paper]](https://arxiv.org/abs/2311.12570)
1. [2023 Genome Biology] **Benchmarking algorithms for joint integration of unpaired and paired single-cell RNA-seq and ATAC-seq data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03073-x)
1. [2023 Nature Communications] **A comprehensive benchmarking with practical guidelines for cellular deconvolution of spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41467-023-37168-7)
1. [2023 bioRxiv] **Universal preprocessing of single-cell genomics data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.14.543267v1.full.pdf)
1. [2023 Genome Biology] **Meta-analysis of (single-cell method) benchmarks reveals the need for extensibility and interoperability** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02962-5)
1. [2023 Nature Communications] **A comprehensive benchmarking with practical guidelines for cellular deconvolution of spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41467-023-37168-7)
1. [2023 bioRxiv] **Benchmarking the Autoencoder Design for Imputing Single-Cell RNA Sequencing Data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.02.16.528866v1.abstract)
1. [2023 bioRxiv] **Benchmarking Algorithms for Gene Set Scoring of Single-cell ATAC-seq Data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.01.14.524081v1)
1. [2022 Nature Communications] **Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data** [[paper]](https://www.nature.com/articles/s41467-022-30755-0)
1. [2022 Nature Methods] **Benchmarking atlas-level data integration in single-cell genomics** [[paper]](https://www.nature.com/articles/s41592-021-01336-8.pdf)
1. [2022 Nature Methods] **Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution** [[paper]](https://www.nature.com/articles/s41592-022-01480-9)
1. [2022 BioRxiv] **Benchmarking Automated Cell Type Annotation Tools for Single-cell ATAC-seq Data** [[paper]](https://www.biorxiv.org/content/10.1101/2022.10.05.511014v1)
1. [2022 Briefings in Bioinformatics] **Benchmarking methods for detecting differential states between conditions from multi-subject single-cell RNA-seq data** [[paper]](https://academic.oup.com/bib/article/23/5/bbac286/6649780)
1. [2022 Nucleic Acids Research] **scIMC: a platform for benchmarking comparison and visualization analysis of scRNA-seq data imputation methods** [[paper]](https://academic.oup.com/nar/article/50/9/4877/6582166)
1. [2021 Frontiers in Genetics] **Evaluating the Reproducibility of Single-Cell Gene Regulatory Network Inference Algorithms** [[paper]](https://www.frontiersin.org/articles/10.3389/fgene.2021.617282/full)
1. [2021 Nature Communications] **A benchmark study of simulation methods for single-cell RNA sequencing data** [[paper]](https://www.nature.com/articles/s41467-021-27130-w)
1. [2021 Genome Biology] **Benchmarking UMI-based single-cell RNA-seq preprocessing workflows** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02552-3)
1. [2020 Nature Methods] **Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data** [[paper]](https://www.nature.com/articles/s41592-019-0690-6)
1. [2020 Genome Biology] **A benchmark of batch-effect correction methods for single-cell RNA sequencing data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9)
1. [2020 Nature Biotechnology] **A multicenter study benchmarking single-cell RNA sequencing technologies using reference samples** [[paper]](https://www.nature.com/articles/s41587-020-00748-9)
1. [2019 Nature Methods] **Benchmarking single cell RNA-sequencing analysis pipelines using mixture control experiments** [[paper]](https://www.nature.com/articles/s41592-019-0425-8)

## Metric Design
1. [2019 Narure Methods] **A test metric for assessing single-cell RNA-seq batch correction** [[paper]](https://www.nature.com/articles/s41592-018-0254-1)

## Subcellular Analysis
1. [2024 Nature Communications] **BIDCell: Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data** [[paper]](https://www.nature.com/articles/s41467-023-44560-w)
1. [2023 Nature Methods] **Spatiotemporally resolved transcriptomics reveals the subcellular RNA kinetic landscape** [[paper]](https://www.nature.com/articles/s41592-023-01829-8)
1. [2023 biorxiv] **Bering: joint cell segmentation and annotation for spatial transcriptomics with transferred graph embeddings** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.19.558548v1.full.pdf)
1. [2023 Bioinformatics] **FISHFactor: a probabilistic factor model for spatial transcriptomics data with subcellular resolution** [[paper]](https://academic.oup.com/bioinformatics/article/39/5/btad183/7114027)
1. [2023 Science] **Spatially resolved single-cell translatomics at molecular resolution** [[paper]](https://www.science.org/doi/10.1126/science.add3067)
1. [2023 Nature Methods] **Subcellular omics: a new frontier pushing the limits of resolution, complexity and throughput** [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10049458/pdf/nihms-1881939.pdf)
1. [2022 BioRxiv] **Bento: A toolkit for subcellular analysis of spatial transcriptomics data** [[paper]](https://www.biorxiv.org/content/10.1101/2022.06.10.495510v1)
1. [2022 BioRxiv] **Subcellular spatially resolved gene neighborhood networks in single cells** [[paper]](https://www.biorxiv.org/content/10.1101/2022.08.03.502409v1)
1. [2022 bioRxiv] **Statistical analysis supports pervasive RNA subcellular localization and alternative 3’ UTR regulation** [[paper]](https://www.biorxiv.org/content/biorxiv/early/2022/10/27/2022.10.26.513902.full.pdf)
1. [2019 Cell] **Atlas of Subcellular RNA Localization Revealed by APEX-Seq** [[paper]](https://www.sciencedirect.com/science/article/pii/S0092867419305550?via%3Dihub)



## Dimensionality Reduction and Visualization
1. [2023 Genome Research] **Complex hierarchical structures in single-cell genomics data unveiled by deep hyperbolic manifold learning** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36849204/)
1. [2021 Nature Communications] **Deep generative model embedding of single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces** [[paper]](https://www.nature.com/articles/s41467-021-22851-4)
1. [2018 Nature Communications] **Interpretable dimensionality reduction of single cell transcriptome data with deep generative models** [[paper]](https://www.nature.com/articles/s41467-018-04368-5)


## Representation Learning
1. [2025 arxiv] **SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics** [[paper]](https://arxiv.org/abs/2412.01124)
1. [2023 Nature Machine Intelligence] **Reusability report: Learning the transcriptional grammar in single-cell RNA-sequencing data using transformers** [[paper]](https://www.nature.com/articles/s42256-023-00757-8)
1. [2023 Genome Biology] **Correcting gradient-based interpretations of deep neural networks for genomics** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02956-3)
1. [2023 Nature Methods] **SIMBA: single-cell embedding along with features** [[paper]](https://www.nature.com/articles/s41592-023-01899-8)
1. [2023 bioRxiv] **Towards Universal Cell Embeddings: Integrating Single-cell RNA-seq Datasets across Species with SATURN** [[paper]](https://www.biorxiv.org/content/10.1101/2023.02.03.526939v1?rss=1)
1. [2021 Current Opinion in Systems Biology] **Graph representation learning for single-cell biology** [[paper]](https://www.sciencedirect.com/science/article/pii/S2452310021000329)
1. [2020 Nature Communications] **Realistic in silico generation and augmentation of single-cell RNA-seq data using generative adversarial networks** [[paper]](https://www.nature.com/articles/s41467-019-14018-z)
1. [2019 Nature Methods] **Data denoising with transfer learning in single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-019-0537-1)
1. [2018 Nature Methods] **Deep generative modeling for single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-018-0229-2)


## Batch Effect Correction
1. [2023 Bioinformatics] **CLAIRE: contrastive learning-based batch correction framework for better balance between batch mixing and preservation of cellular heterogeneity** [[paper]](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad099/7055295)
1. [2020 Genomy Biology] **A benchmark of batch-effect correction methods for single-cell RNA sequencing data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9)
1. [2020 Nature Biotechnology] **A multicenter study benchmarking single-cell RNA sequencing technologies using reference samples** [[paper]](https://www.nature.com/articles/s41587-020-00748-9)
1. [2019 Nature Methods, **Harmony**] **Fast, sensitive and accurate integration of single-cell data with Harmony** [[paper]](https://www.nature.com/articles/s41592-019-0619-0)
1. [2018 Nature Biotechnology, **CCA**] **Integrating single-cell transcriptomic data across different conditions, technologies, and species** [[paper]](https://www.nature.com/articles/nbt.4096)
1. [2018 Nature Biotechnology, **Mutual Nearest Neighbors**] **Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors** [[paper]](https://www.nature.com/articles/nbt.4091)
1. [2018 Nature Methods] **A test metric for assessing single-cell RNA-seq batch correction** [[paper]](https://www.nature.com/articles/s41592-018-0254-1)
1. [2017 Nature Biotechnology] **Multiplexed droplet single-cell RNA-sequencing using natural genetic variation** [[paper]](https://www.nature.com/articles/nbt.4042)

## Tumor Microenvironment-TME
1. [2023 bioRxiv] **Identifying Spatial Co-occurrence in Healthy and InflAmed tissues (ISCHIA)** [[paper]](https://www.biorxiv.org/content/10.1101/2023.02.13.526554v1)
1. [2023 bioRxiv] **Predicting tumor immune microenvironment and checkpoint therapy response of head & neck cancer patients from blood immune single-cell transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2023.01.17.524455v1)
1. [2022 Nature Biomedical Engineering] **Graph deep learning for the characterization of tumour microenvironments from spatial protein profiles in tissue specimens** [[paper]](https://www.nature.com/articles/s41551-022-00951-w)
1. [2022 Nature Communications] **SOTIP is a versatile method for microenvironment modeling with spatial omics data** [[paper]](https://www.nature.com/articles/s41467-022-34867-5)



## Cell-Cell Communication Events
1. [2024 Nature Methods] **Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes** [[paper]](https://www.nature.com/articles/s41592-023-02124-2)
1. [2024 Nature Reviews Genetics] **The diversification of methods for studying cell–cell interactions and communication** [[paper]](https://www.nature.com/articles/s41576-023-00685-8)
1. [2024 bioRxiv] **Large-scale characterization of cell niches in spatial atlases using bio-inspired graph learning** [[paper]](https://www.biorxiv.org/content/10.1101/2024.02.21.581428v2.full.pdf)
1. [2024 Pac Symp Biocomput] **PEPSI: Polarity measurements from spatial proteomics imaging suggest immune cell engagement** [[paper]](https://pubmed.ncbi.nlm.nih.gov/38160302/)
1. [2023 Cell Systems] **Single-cell A/B testing for cell-cell communication** [[paper]](https://www.sciencedirect.com/science/article/pii/S2405471223001503)
1. [2023 Nature Biotechnology] **Inferring cell–cell communication at single-cell resolution** [[paper]](https://www.nature.com/articles/s41587-023-01834-4)
1. [2022 bioRxiv] **scTensor detects many-to-many cell–cell interactions from single cell RNA-sequencing data** [[paper]](https://www.biorxiv.org/content/10.1101/2022.12.07.519225v1)
1. [2022 Nature Biotechnology] **Modeling intercellular communication in tissues using spatial graphs of cells** [[paper]](https://www.nature.com/articles/s41587-022-01467-z)
1. [2022 bioRxiv] **Decoding functional cell–cell communication events by multi-view graph
learning on spatial transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2022.06.22.496105v1)
1. [2021 Bioinformatics] **Identifying signaling genes in spatial single-cell expression data** [[paper]](https://pubmed.ncbi.nlm.nih.gov/32886099/)
1. [2020 Nature Methods] **NicheNet: modeling intercellular communication by linking ligands to target genes** [[paper]](https://www.nature.com/articles/s41592-019-0667-5)
1. [2020 Nature Communications] **Predicting cell-to-cell communication networks using NATMI** [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7538930/pdf/41467_2020_Article_18873.pdf)
1. [2018 Nature] **Single-cell reconstruction of the early maternal–fetal interface in humans** [[paper]](https://www.nature.com/articles/s41586-018-0698-6)


## Gene Regulatory Network
1. [2023 arxiv] **DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets** [[paper]](https://arxiv.org/pdf/2302.04178.pdf)
1. [2023 Bioinformatics] **STGRNS: an interpretable transformer-based method for inferring gene regulatory networks from single-cell transcriptomic data** [[paper]](https://academic.oup.com/bioinformatics/article/39/4/btad165/7099621)
1. [2022 Nature Machine Intelligence] **Inferring transcription factor regulatory networks from single-cell ATAC-seq data based on graph neural networks** [[paper]](https://www.nature.com/articles/s42256-022-00469-5)
1. [2022 Nature Biotechnology] **Multi-omics single-cell data integration and regulatory inference with graph-linked embedding** [[paper]](https://www.nature.com/articles/s41587-022-01284-4.pdf)
1. [2022 Biorxiv] **scMEGA: Single-cell Multiomic Enhancer-based Gene Regulatory Network Inference** [[paper]](https://www.biorxiv.org/content/10.1101/2022.08.10.503335v1)
1. [2022 Bioinformatics] **High-performance single-cell gene regulatory network inference at scale: the Inferelator 3.0** [[paper]](https://academic.oup.com/bioinformatics/article/38/9/2519/6533443)
1. [2022 Briefings in Bioinformatic] **SIGNET: single-cell RNA-seq-based gene regulatory network prediction using multiple-layer perceptron bagging** [[paper]](https://academic.oup.com/bib/article/23/1/bbab547/6484519)
1. [2020 Nature Methods] **Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data** [[paper]](https://www.nature.com/articles/s41592-019-0690-6)
1. [2019 Genome Biology] **Single-cell transcriptomics unveils gene regulatory network plasticity** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1713-4)
1. [2017 Cell Syst] **Gene Regulatory Network Inference from Single-Cell Data Using Multivariate Information Measures** [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624513/)


## Imputation
1. [2018 Nature Communications] **An accurate and robust imputation method scImpute for single-cell RNA-seq data** [[paper]](https://www.nature.com/articles/s41467-018-03405-7)
1. [2019 Genome Biology] **DeepImpute: an accurate, fast, and scalable deep neural network method to impute single-cell RNA-seq data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1837-6)
1. [2018 Cell] **Recovering Gene Interactions from Single-Cell Data Using Data Diffusion** [[paper]](https://www.cell.com/cell/fulltext/S0092-8674(18)30724-4)
1. [2018 Genome Biology] **VIPER: variability-preserving imputation for accurate gene expression recovery in single-cell RNA sequencing studies** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1575-1)
1. [2021 PLOS Computational Biology] **G2S3: A gene graph-based imputation method for single-cell RNA sequencing data** [[paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009029)
1. [2021 Nature Communications] **scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses** [[paper]](https://www.nature.com/articles/s41467-021-22197-x#Sec23)
1. [2021 iScience] **Imputing single-cell RNA-seq data by combining graph convolution and autoencoder neural networks** [[paper]](https://www.cell.com/iscience/fulltext/S2589-0042(21)00361-8)
1. [2022 PLOS ONE] **Single-cell specific and interpretable machine learning models for sparse scChIP-seq data imputation** [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0270043)

## Spatial Domain
1. [2023 Nature Genetics] **SPICEMIX enables integrative single-cell spatial modeling of cell identity** [[paper]](https://www.nature.com/articles/s41588-022-01256-z)
1. [2023 bioRxiv] **CellCharter: a scalable framework to chart and compare cell niches across multiple samples and spatial -omics technologies** [[preprint]](https://www.biorxiv.org/content/10.1101/2023.01.10.523386v1)
1. [2022 Genome Research] **A model-based constrained deep learning clustering approach for spatially resolved single-cell data** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36198490/)
1. [2022 Nature Communications Biology] **Deciphering tissue structure and function using spatial transcriptomics** [[Review paper]](https://www.nature.com/articles/s42003-022-03175-5)
1. [2022 Genome Biology] **Statistical and machine learning methods for spatially resolved transcriptomics data analysis** [[Review paper]](https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-022-02653-7.pdf)
1. [2022 Nature Communications] **Deciphering spatial domains from spatially resolved transcriptomics with adaptive graph attention auto-encoder** [[paper]](https://www.nature.com/articles/s41467-022-29439-6)
1. [2022 Nature Computational Science] **Cell clustering for spatial transcriptomics data with graph neural networks** [[paper]](https://www.nature.com/articles/s43588-022-00266-5)
1. [2022 Frontiers in Genetics] **Analysis and Visualization of Spatial Transcriptomic Data** [[paper]](https://arxiv.org/pdf/2110.07787.pdf)
1. [2021 Nature Methods] **SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network** [[paper]](https://www.nature.com/articles/s41592-021-01255-8)
1. [2021 Nature Biotechnology] **Spatial transcriptomics at subspot resolution with BayesSpace** [[paper]](https://www.nature.com/articles/s41587-021-00935-2)
1. [2021 Biorxiv] **Unsupervised Spatially Embedded Deep Representation of Spatial Transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2021.06.15.448542v2)
1. [2021 Genome Biology] **Giotto: a toolbox for integrative analysis
and visualization of spatial expression data** [[Tool]](https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-021-02286-2.pdf)
1. [2021 Biorxiv] **Define and visualize pathological architectures of human tissues from spatially resolved transcriptomics using deep learning** [[paper]](https://www.biorxiv.org/content/10.1101/2021.07.08.451210v2)
1. [2020 Biorxiv] **stLearn: integrating spatial location, tissue morphology and gene expression to find cell types, cell-cell interactions and spatial trajectories within undissociated tissues** [[paper]](https://www.biorxiv.org/content/10.1101/2020.05.31.125658v1)
1. [2018 Nature Methods] **SpatialDE: Identification of Spatially Variable Genes** [[paper]](https://www.nature.com/articles/nmeth.4636)
1. [2018 Nature Biotechnology] **Identification of Spatially Associated Subpopulations by Combining scRNAseq and Sequential Fluorescence In Situ Hybridization Data** [[paper]](https://www.nature.com/articles/nbt.4260)
1. [2008 Journal of Statistical Mechanics] **Fast unfolding of community hierarchies in large networks** [[paper]](https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008)


## Reference Embedding or Transfer Learning
1. [2019 Nature Methods] **Data denoising with transfer learning in single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-019-0537-1)
1. [2018 Nature Methods] **Deep generative modeling for single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-018-0229-2)
1. [2020 Bioinformatics] **Conditional out-of-distribution generation for unpaired data using transfer VAE** [[paper]](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5&login=false)
1. [2021 Nature Biotechnology] **Mapping single-cell data to reference atlases by transfer learning** [[paper]](https://www.nature.com/articles/s41587-021-01001-7)
1. [2021 Molecular Systems Biology] **Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models** [[paper]](https://www.embopress.org/doi/full/10.15252/msb.20209620)
1. [2022 bioRxiv Preprint] **Biologically informed deep learning to infer gene program activity in single cells** [[preprint]](https://www.biorxiv.org/content/10.1101/2022.02.05.479217v2)

## Cell Segmentation
1. [2023 biorxiv] **Bering: joint cell segmentation and annotation for spatial transcriptomics with transferred graph embeddings** [[paper]](https://www.biorxiv.org/content/10.1101/2023.09.19.558548v1.full.pdf)
1. [2022 Cytometry A] **MIRIAM: A machine and deep learning single-cell segmentation and quantification pipeline for multi-dimensional tissue images** [[paper]](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cyto.a.24541)[[code]](https://github.com/Coffey-Lab/MIRIAM)(MIRIAM)
1. [2021 Nature Biotechnology] **Cell segmentation in imaging-based spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-021-01044-w) 
1. [2021 Biorxiv] **Scellseg: a style-aware cell instance segmentation tool with pre-training and
contrastive fine-tuning** [[paper]](https://www.biorxiv.org/content/10.1101/2021.12.19.473392v1) [[code]](https://github.com/cellimnet/scellseg-publish)
1. [2021 Nature Biotechnology] **Cell segmentation in imaging-based spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-021-01044-w) [[code]](https://github.com/kharchenkolab/Baysor)(Baysor)
1. [2021 Nature Biotechnology] **Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning** [[paper]](https://www.nature.com/articles/s41587-021-01094-0) [[code]](https://github.com/vanvalenlab/intro-to-deepcell)(Memser)
1. [2021 Nature Methods] **Cellpose: a generalist algorithm for cellular segmentation** [[paper]](https://www.nature.com/articles/s41592-020-01018-x) [[code]](https://www.github.com/mouseland/cellpose)(Cellpose)
1. [2021 Molecular Systems Biology]**Joint cell segmentation and cell type annotation for spatial transcriptomics** [[paper]](https://pubmed.ncbi.nlm.nih.gov/34057817/) [[code]](https://github.com/wollmanlab/JSTA) (JSTA)
1. [2020 Nature Communications]**A convolutional neural network segments yeast microscopy images with high accuracy** [[paper]](https://www.nature.com/articles/s41467-020-19557-4) [[code]](http://github.com/lpbsscientist/YeaZ-GUI)
1. [2020 Medical Image Analysis] **DeepDistance: A multi-task deep regression model for cell detection in inverted microscopy images** [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300840?via%3Dihub) (DeepDistance)
1. [2016 Computational Biology]**Deep Learning Automates the Quantitative Analysis of Individual Cells in Live-Cell Imaging Experiments** [[paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005177) [[code]](https://github.com/vanvalenlab/deepcell-tf) (Deepcell)

## Cell Type Deconvolution
1. [2023 Genome Biology] **Smoother: a unified and modular framework for incorporating structural dependency in spatial omics data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03138-x)
1. [2023 BioRxiv] **RETROFIT: REFERENCE-FREE DECONVOLUTION OF CELL-TYPE MIXTURES IN SPATIAL TRANSCRIPTOMICS** [[paper]](https://www.biorxiv.org/content/10.1101/2023.06.07.544126v1)
1. [2023 BioRxiv] **STdGCN: accurate cell-type deconvolution using graph convolutional networks in spatial transcriptomic data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.03.10.532112v1)
1. [2023 BioRxiv] **Spotless: a reproducible pipeline for benchmarking cell type deconvolution in spatial transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2023.03.22.533802v1)
1. [2022 Nature Biotechnology] **High-resolution alignment of single-cell and spatial transcriptomes with CytoSPACE** [[paper]](https://www.nature.com/articles/s41587-023-01697-9)
1. [2022 Nature Communications] **Reference-free cell type deconvolution of multi-cellular pixel-resolution spatially resolved transcriptomics data** [[paper]](https://www.nature.com/articles/s41467-022-30033-z)
1. [2022 Nature Biotechnology] **DestVI identifies continuums of cell types in spatial transcriptomics data** [[paper]](https://www.nature.com/articles/s41587-022-01272-8)
1. [2022 Biorxiv] **Accurate cell type deconvolution in spatial transcriptomics using a batch effect-free strategy** [[paper]](https://www.biorxiv.org/content/10.1101/2022.12.15.520612v1)
1. [2022 Nature Biotechnology] **Spatially informed cell-type deconvolution for spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-022-01273-7#Sec2)
1. [2022 Nature Cancer] **Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology** [[paper]](https://doi.org/10.1038/s43018-022-00356-3)
1. [2022 Nature Communications] **Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data** [[paper]](https://www.nature.com/articles/s41467-022-28020-5#Sec2)
1. [2022 Nature Biotechnology] **Cell2location maps fine-grained cell types in spatial transcriptomics** [[paper]](https://doi.org/10.1038/s41587-021-01139-4)
1. [2021 Briefings in Bioinformatics] **DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence** [[paper]](https://doi.org/10.1093/bib/bbaa414)
1. [2021 Genome Research] **Likelihood-based deconvolution of bulk gene expression data using single-cell references** [[paper]](https://www.genome.org/cgi/doi/10.1101/gr.272344.120.)
1. [2021 Genome Biology] **SpatialDWLS: accurate deconvolution of spatial transcriptomic data** [[paper]](https://doi.org/10.1186/s13059-021-02362-7)
1. [2021 Nucleic Acids Research] **SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics spots with single-cell transcriptomes** [[paper]](https://doi.org/10.1093/nar/gkab043)
1. [2021 Nature Biotechnology] **Robust decomposition of cell type mixtures in spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-021-00830-w)
1. [2019 Nature Communications] **Accurate estimation of cell-type composition from gene expression data** [[paper]](https://doi.org/10.1038/s41467-019-10802-z)
1. [2019 Science] **Slide-seq: A scalable technology for measuring genome-wide expression at high spatial resolution** [[paper]](https://doi.org/10.1126/science.aaw1219)

## Cell Type Annotation 

1. [2025 BioRxiv] **CASSIA: a multi-agent large language model for reference free, interpretable, and automated cell annotation of single-cell RNA-sequencing data** [[paper]](https://www.biorxiv.org/content/10.1101/2024.12.04.626476v2) [[code]](https://github.com/ElliotXie/CASSIA/tree/main)
1. [2025 BioRxiv] **Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data** [[paper]](https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1) [[code]](https://github.com/cafferychen777/mLLMCelltype)
1. [2023 biorxiv] **Scaling cross-tissue single-cell annotation models** [[paper]](https://www.biorxiv.org/content/10.1101/2023.10.07.561331v1.full.pdf) 
1. [2023 Nature Methods] **Multi-layered maps of neuropil with segmentation-guided contrastive learning** [[paper]](https://www.nature.com/articles/s41592-023-02059-8) 
1. [2023 Nature Methods] **Cue: a deep-learning framework for structural variant discovery and genotyping** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36959322/) 
1. [2023 Nature Communications] **Transformer for one stop interpretable cell type annotation** [[paper]](https://www.nature.com/articles/s41467-023-35923-4) 
1. [2023 Nature Biotech] **TACCO unifies annotation transfer and decomposition of cell identities for single-cell and spatial omics** [[paper]](https://www.nature.com/articles/s41587-023-01657-3) 
1. [2022 Nature Method] **Annotation of spatially resolved single-cell data with STELLAR** [[paper]](https://www.nature.com/articles/s41592-022-01651-8) 
NOTE: annotated reference cell graph + query cell graph
1. [2022 Brief Bioinform] **scIAE: an integrative autoencoder-based ensemble classification framework for single-cell RNA-seq data** [[paper]](https://doi.org/10.1093/bib/bbab508)
1. [2022 Nature Communications] **scGCN is a graph convolutional networks algorithm for knowledge transfer in single cell omics** [[paper]](https://doi.org/10.1038/s41467-021-24172-y)
1. [2022 Science] **Cross-tissue immune cell analysis reveals tissue-specific features in humans** [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7612735/#S9)
1. [2022 Bioinformatics] **CellMeSH: probabilistic cell-type identification using indexed literature** [[paper]](https://doi.org/10.1093/bioinformatics/btab834)
1. [2022 Cancers] **Transformer for Gene Expression Modeling (T-GEM): An Interpretable Deep Learning Model for Gene Expression-Based Phenotype Predictions** [[paper]](https://pubmed.ncbi.nlm.nih.gov/36230685/)
1. [2021 Nucleic Acids Research] **scDeepSort: a pre-trained cell-type annotation method for single-cell transcriptomics using deep learning with a weighted graph neural network** [[paper]](https://doi.org/10.1093/nar/gkab775)
1. [2021 BMC Bioinformatics] **Single-cell classification using graph convolutional networks** [[paper]](https://doi.org/10.1186/s12859-021-04278-2)
1. [2021 Genome Research] **Semisupervised adversarial neural networks for single-cell classification** [[paper]](https://genome.cshlp.org/content/31/10/1781)
1. [2020 BMC Bioinformatics] **EnClaSC: a novel ensemble approach for accurate and robust cell-type classification of single-cell transcriptomes** [[paper]](https://doi.org/10.1186/s12859-020-03679-z)
1. [2020 Bioinformatics] **ACTINN: automated identification of cell types in single cell RNA sequencing** [[paper]](https://doi.org/10.1093/bioinformatics/btz592)
1. [2020 Nature Communications] **SciBet as a portable and fast single cell type identifier** [[paper]](https://doi.org/10.1038/s41467-020-15523-2)
1. [2019 Nucleic Acids Research] **SuperCT: a supervised-learning framework for enhanced characterization of single-cell transcriptomic profiles** [[paper]](https://doi.org/10.1093/nar/gkz116)
1. [2019 Nucleic Acids Research] **CHETAH: a selective, hierarchical cell type identification method for single-cell RNA sequencing** [[paper]]( https://doi.org/10.1093/nar/gkz543)
1. [2019 Bioinformatics] **scMatch: a single-cell gene expression profile annotation tool using reference datasets** [[paper]](https://doi.org/10.1093/bioinformatics/btz292)
1. [2019 Cell Systems] **SingleCellNet: A Computational Tool to Classify Single Cell RNA-Seq Data Across Platforms and Across Species** [[paper]](https://doi.org/10.1016/j.cels.2019.06.004)
1. [2019 Genome Biology] **SingleCellNet: cPred: accurate supervised method for cell-type classification from single-cell RNA-seq data** [[paper]](https://doi.org/10.1186/s13059-019-1862-5)



## Cell Clustering
1. [2023 Bioinformatics] **scBGEDA: deep single-cell clustering analysis via a dual denoising autoencoder with bipartite graph ensemble clustering** [[paper]](https://www.biorxiv.org/content/10.1101/2023.01.15.524109v1](https://academic.oup.com/bioinformatics/article/39/2/btad075/7025496))
1. [2023 bioRxiv] **G3DC: a Gene-Graph-Guided selective Deep Clustering method for single cell RNA-seq data** [[paper]](https://www.biorxiv.org/content/10.1101/2023.01.15.524109v1)
1. [2022 BMC Bioinformatics] **SC3s: efficient scaling of single cell consensus clustering to millions of cells** [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05085-z)
1. [2022 Bioinformatics] **GNN-based embedding for clustering scRNA-seq data** [[paper]](https://doi.org/10.1093/bioinformatics/btab787)
1. [2022 AAAI] **ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations** [[paper]]( https://aaai-2022.virtualchair.net/poster_aaai5060)
1. [2022 Briefings in Bioinformatics] **Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph neural network** [[paper]]( https://doi.org/10.1093/bib/bbac018)
1. [2022 Bioinformatics] **scGAC: a graph attentional architecture for clustering single-cell RNA-seq data** [[paper]]( https://doi.org/10.1093/bioinformatics/btac099)
1. [2022 Nature Computational Science] **Cell clustering for spatial transcriptomics data with graph neural networks** [[paper]]( https://www.nature.com/articles/s43588-022-00266-5)
1. [2021 Nature Communications] **Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data** [[paper]]( https://www.nature.com/articles/s41467-021-22008-3)
1. [2020 NAR Genomics and Bioinformatics] **Deep soft K-means clustering with self-training for single-cell RNA sequence data** [[paper]]( https://doi.org/10.1093/nargab/lqaa039)
1. [2020 Nature Communications] **Deep learning enables accurate clustering with batch effect removal in single-cell RNA-seq analysis** [[paper]](https://www.nature.com/articles/s41467-020-15851-3)[[website]](https://eleozzr.github.io/desc/)[[github]](https://github.com/eleozzr/desc)
1. [2019 Nature Machine Intelligence] **Clustering single-cell RNA-seq data with a model-based deep learning approach** [[paper]]( https://www.nature.com/articles/s42256-019-0037-0)

<!---
## Cell Trajectory 
1. [2017 Nature Communications] **Reconstructing cell cycle and disease progression using deep learning** [[paper]](https://www.nature.com/articles/s41467-017-00623-3)
--->

## Disease Prediction
1. [2024 Nature Biotechnology] **Can single-cell biology realize the promise of precision medicine?** [[paper]](https://www.nature.com/articles/s41587-024-02138-x)
1. [2018 IJCAI] **Hybrid Approach of Relation Network and Localized Graph Convolutional Filtering for Breast Cancer Subtype Classification** [[paper]](https://www.ijcai.org/Proceedings/2018/490)
1. [2021 NPJ Digital Medicine] **DeePaN - A deep patient graph convolutional network integratingclinico-genomic evidence to stratify lung cancers benefiting from immunotherapy** [[paper]](https://www.nature.com/articles/s41746-021-00381-z)
1. [2022 Biocumputing] **CloudPred: Predicting Patient Phenotypes From Single-cell RNA-seq** [[paper]](https://www.worldscientific.com/doi/abs/10.1142/9789811250477_0031)
1. [2022 CHIL '20: Proceedings of the ACM Conference on Health, Inference, and Learning] **Disease state prediction from single-cell data using graph attention networks** [[paper]](https://dl.acm.org/doi/10.1145/3368555.3384449)

## Multimodal Integration
1. [2024 Nature Methods] **Search and match across spatial omics samples at single-cell resolution** [[Paper]](https://www.nature.com/articles/s41592-024-02410-7)
1. [2023 Nature Biotechnology] **Integration of spatial and single-cell data across modalities with weakly linked features** [[Paper]](https://www.nature.com/articles/s41587-023-01935-0)
1. [2023 Nature Communications] **scDREAMER for atlas-level integration of single-cell datasets using deep generative model paired with adversarial classifier** [[Paper]](https://www.nature.com/articles/s41467-023-43590-8)
1. [2023 biorxiv] **Automated single-cell omics end-to-end framework with data-driven batch inference** [[Paper]](https://www.biorxiv.org/content/10.1101/2023.11.01.564815v1)
1. [2023 Nature Biotechnology] **Integration of multi-modal single-cell data** [[Paper]](https://www.nature.com/articles/s41587-023-01826-4)
1. [2023 Nature Biotechnology] **Integration of spatial and single-cell data across modalities with weakly linked features** [[Paper]](https://www.nature.com/articles/s41587-023-01935-0)
1. [2023 Briefings in Bioinformatics] **A universal framework for single-cell multi-omics data integration with graph convolutional networks** [[Paper]](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbad081/7079707)
1. [2022 PMLR] **CVQVAE: A representation learning based method for multi-omics single cell data integration** [[Paper]](https://proceedings.mlr.press/v200/liu22a.html)
1. [2022 Nature Biotechnology] **Multi-omics single-cell data integration and regulatory inference with graph-linked embedding** [[Paper]](https://www.nature.com/articles/s41587-022-01284-4)
1. [2022 Nature Communications] **Clustering of single-cell multi-omics data with a multimodal deep learning method** [[Survey]](https://www.nature.com/articles/s41467-022-35031-9)
1. [2022 Genome Biology] **A benchmark study of deep learning-based multi-omics data fusion methods for cancer** [[Survey]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02739-2)
1. [2018 ICML] **MAGAN: Aligning biological manifolds** [[paper]](https://proceedings.mlr.press/v80/amodio18a.html)
1. [2019 PLoS computational biology] **Building gene regulatory networks from scATAC-seq and scRNA-seq using linked self organizing maps** [[paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006555)
1. [2020 Bioinformatics] **SCIM: universal single-cell matching with unpaired feature sets** [[paper]](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i919/6055906)
1. [2021 Nature communications] **Multi-domain translation between single-cell imaging and sequencing data using autoencoders** [[paper]](https://www.nature.com/articles/s41467-020-20249-2)
1. [2021 PLoS Computational Biology] **Imputation of spatially-resolved transcriptomes by graph-regularized tensor completion** [[paper]](https://pubmed.ncbi.nlm.nih.gov/33826608/)
1. [2021 Genome biology] **Cobolt: integrative analysis of multimodal single-cell sequencing data** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02556-z)
1. [2021 Cell reports methods] **A mixture-of-experts deep generative model for integrated analysis of single-cell multiomics data** [[paper]](https://www.sciencedirect.com/science/article/pii/S2667237521001235)
1. [2021 Briefings in Bioinformatics] **Deep-joint-learning analysis model of single cell transcriptome and open chromatin accessibility data** [[paper]](https://academic.oup.com/bib/article/22/4/bbaa287/5985290)
1. [2021 Bioinformatics] **Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data** [[paper]](https://academic.oup.com/bioinformatics/article/37/22/4091/6283577)
1. [2022 Nature Biotechnology] **scJoint integrates atlas-scale single-cell RNA-seq and ATAC-seq data with transfer learning** [[paper]](https://www.nature.com/articles/s41587-021-01161-6)
1. [2022 Bioinformatics] **SMILE: mutual information learning for integration of single-cell omics data** [[paper]](https://academic.oup.com/bioinformatics/article/38/2/476/6384571)
1. [2022 SIGKDD] **Graph Neural Networks for Multimodal Single-Cell Data Integration** [[paper]](https://dl.acm.org/doi/10.1145/3534678.3539213)
1. [2022 Genome biology] **scDART: integrating unmatched scRNA-seq and scATAC-seq data and learning cross-modality relationship simultaneously** [[paper]](https://link.springer.com/article/10.1186/s13059-022-02706-x)
1. [2019 Biorxiv] **A Joint Model of RNA Expression and Surface Protein Abundance in Single Cells** [[paper]](https://www.biorxiv.org/content/10.1101/791947v1.abstract)
1. [2021 Biorxiv] **DeepMAPS: Single-cell biological network inference using heterogeneous graph transformer** [[paper]](https://www.biorxiv.org/content/10.1101/2021.10.31.466658v2)
1. [2022 Biorxiv] **Adaptative Machine Translation between paired Single-Cell Multi-Omics Data** [[paper]](https://www.biorxiv.org/content/10.1101/2021.01.27.428400v2)
1. [2022 Biorxiv] **Multigrate: single-cell multi-omic data integration** [[paper]](https://www.biorxiv.org/content/10.1101/2022.03.16.484643v1.abstract)
1. [2019 NeurIPS multi-lingual pretraining for multi-omics] **Cross-lingual Language Model Pretraining** [[Paper]](https://arxiv.org/pdf/1901.07291.pdf)


## Multiomics Translation
1. [2024 Nature Communications] **scButterfly: a versatile single-cell cross-modality translation method via dual-aligned variational autoencoders** [[paper]](https://www.nature.com/articles/s41467-024-47418-x)
1. [2023 arxiv scHyena] **scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain** [[paper]](https://arxiv.org/abs/2310.02713)
1. [2023 bioRxiv scTranslator] **A pre-trained large language model for translating single-cell transcriptome to proteome** [[paper]](https://www.biorxiv.org/content/10.1101/2023.07.04.547619v1)
1. [2023 Advanced Science] **Efficient Generation of Paired Single-Cell Multiomics Profiles by Deep Learning** [[paper]](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301169)
1. [2022 JCB] **Multimodal Single-Cell Translation and Alignment with Semi-Supervised Learning** [[paper]](https://www.liebertpub.com/doi/full/10.1089/cmb.2022.0264)
1. [2022 Nature Machine Intelligence sciPENN] **A multi-use deep learning method for CITE-seq and single-cell RNA-seq data integration with cell surface protein prediction and imputation** [[paper]](https://www.nature.com/articles/s42256-022-00545-w)
1. [2022 RECOMB] **Semi-supervised Single-Cell Cross-modality Translation Using Polarbear** [[paper]](https://dl.acm.org/doi/abs/10.1007/978-3-031-04749-7_2)
1. [2020 PNAS] **BABEL enables cross-modality translation between multiomic profiles at single-cell resolution** [[paper]](https://www.pnas.org/doi/abs/10.1073/pnas.2023070118)



