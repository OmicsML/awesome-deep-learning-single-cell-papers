# awesome-deep-learning-single-cell-papers

This repository keeps track of the latest papers on the single cell analysis with deep learning methods. We categorize them based on individual tasks.

We will try to make this list updated. If you found any error or any missed paper, please don't hesitate to open an issue or pull request.

## Modality Prediction


## Modality Matching


## Joint Embedding
1. [2022 Genome Biology] **scDART: integrating unmatched scRNA-seq and scATAC-seq data and learning cross-modality relationship simultaneously** [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02706-x)

## Multimodal Integration


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
1. [2022 Genome Biology] **Statistical and machine learning methods for spatially resolved transcriptomics data analysis** [[Review paper]](https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-022-02653-7.pdf)
1. [2022 Nature Communications] **Deciphering spatial domains from spatially resolved transcriptomics with adaptive graph attention auto-encoder** [[paper]](https://www.nature.com/articles/s41467-022-29439-6)
1. [2022 Nature Computational Science] **Cell clustering for spatial transcriptomics data with graph neural networks** [[paper]](https://www.nature.com/articles/s43588-022-00266-5)
1. [2022 Frontiers in Genetics] **Analysis and Visualization of Spatial Transcriptomic Data** [[paper]](https://arxiv.org/pdf/2110.07787.pdf)
1. [2021 Nature Methods] **SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network** [[paper]](https://www.nature.com/articles/s41592-021-01255-8)
1. [2021 Nature Biotechnology] **Spatial transcriptomics at subspot resolution with BayesSpace** [[paper]](https://www.nature.com/articles/s41587-021-00935-2)
1. [2021 Biorxiv] **Unsupervised Spatially Embedded Deep Representation of Spatial Transcriptomics** [[paper]](https://www.biorxiv.org/content/10.1101/2021.06.15.448542v2)
1. [2021 Biorxiv] **Define and visualize pathological architectures of human tissues from spatially resolved transcriptomics using deep learning** [[paper]](https://www.biorxiv.org/content/10.1101/2021.07.08.451210v2)
1. [2020 Biorxiv] **stLearn: integrating spatial location, tissue morphology and gene expression to find cell types, cell-cell interactions and spatial trajectories within undissociated tissues** [[paper]](https://www.biorxiv.org/content/10.1101/2020.05.31.125658v1)
1. [2018 Nature Methods] **SpatialDE: Identification of Spatially Variable Genes** [[paper]](https://www.nature.com/articles/nmeth.4636)
1. [2018 Nature Biotechnology] **Identification of Spatially Associated Subpopulations by Combining scRNAseq and Sequential Fluorescence In Situ Hybridization Data** [[paper]](https://www.nature.com/articles/nbt.4260)
1. [2008 Journal of Statistical Mechanics] **Fast unfolding of community hierarchies in large networks** [[paper]](https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008)

## Reference Embedding / Transfer Learning
1. [2019 Nature Methods] **Data denoising with transfer learning in single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-019-0537-1)
1. [2018 Nature Methods] **Deep generative modeling for single-cell transcriptomics** [[paper]](https://www.nature.com/articles/s41592-018-0229-2)
1. [2020 Bioinformatics] **Conditional out-of-distribution generation for unpaired data using transfer VAE** [[paper]](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5&login=false)
1. [2021 Nature Biotechnology] **Mapping single-cell data to reference atlases by transfer learning** [[paper]](https://www.nature.com/articles/s41587-021-01001-7)
1. [2021 Molecular Systems Biology] **Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models** [[paper]](https://www.embopress.org/doi/full/10.15252/msb.20209620)
1. [2022 bioRxiv Preprint] **Biologically informed deep learning to infer gene program activity in single cells** [[preprint]](https://www.biorxiv.org/content/10.1101/2022.02.05.479217v2)

## Cell Segmentation
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
1. [2022 Nature Biotechnology] **Spatially informed cell-type deconvolution for spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-022-01273-7#Sec2)
1. [2022 Nature Cancer] **Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology** [[paper]](https://doi.org/10.1038/s43018-022-00356-3)
1. [2022 Nature Communications] **Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data** [[paper]](https://www.nature.com/articles/s41467-022-28020-5#Sec2)
1. [2021 Briefings in Bioinformatics] **DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence** [[paper]](https://doi.org/10.1093/bib/bbaa414)
1. [2021 Genome Research] **Likelihood-based deconvolution of bulk gene expression data using single-cell references** [[paper]](https://www.genome.org/cgi/doi/10.1101/gr.272344.120.)
1. [2021 Genome Biology] **SpatialDWLS: accurate deconvolution of spatial transcriptomic data** [[paper]](https://doi.org/10.1186/s13059-021-02362-7)
1. [2021 Nucleic Acids Research] **SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics spots with single-cell transcriptomes** [[paper]](https://doi.org/10.1093/nar/gkab043)
1. [2021 Nature Biotechnology] **Robust decomposition of cell type mixtures in spatial transcriptomics** [[paper]](https://www.nature.com/articles/s41587-021-00830-w)

## Cell Type Annotation 
1.

## Cell Clustering
1. [2022 Bioinformatics] **GNN-based embedding for clustering scRNA-seq data** [[paper]](https://doi.org/10.1093/bioinformatics/btab787)
1. [2022 AAAI] **ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations** [[paper]]( https://aaai-2022.virtualchair.net/poster_aaai5060)
1. [2022 Briefings in Bioinformatics] **Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph neural network** [[paper]]( https://doi.org/10.1093/bib/bbac018)
1. [2022 Bioinformatics] **scGAC: a graph attentional architecture for clustering single-cell RNA-seq data** [[paper]]( https://doi.org/10.1093/bioinformatics/btac099)
1. [2022 Nature Computational Science] **Cell clustering for spatial transcriptomics data with graph neural networks** [[paper]]( https://www.nature.com/articles/s43588-022-00266-5)
1. [2021 Nature Communications] **Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data** [[paper]]( https://www.nature.com/articles/s41467-021-22008-3)
1. [2020 NAR Genomics and Bioinformatics] **Deep soft K-means clustering with self-training for single-cell RNA sequence data** [[paper]]( https://doi.org/10.1093/nargab/lqaa039)
1. [2019 Nature Machine Intelligence] **Clustering single-cell RNA-seq data with a model-based deep learning approach** [[paper]]( https://www.nature.com/articles/s42256-019-0037-0)


## Cell Trajectory 
1. [2017 Nature Communications] **Reconstructing cell cycle and disease progression using deep learning** [[paper]](https://www.nature.com/articles/s41467-017-00623-3)

## Disease Prediction
1. [2018 IJCAI] **Hybrid Approach of Relation Network and Localized Graph Convolutional Filtering for Breast Cancer Subtype Classification** [[paper]](https://www.ijcai.org/Proceedings/2018/490)
1. [2021 NPJ Digital Medicine] **DeePaN - A deep patient graph convolutional network integratingclinico-genomic evidence to stratify lung cancers benefiting from immunotherapy** [[paper]](https://www.nature.com/articles/s41746-021-00381-z)
1. [2022 Biocumputing] **CloudPred: Predicting Patient Phenotypes From Single-cell RNA-seq** [[paper]](https://www.worldscientific.com/doi/abs/10.1142/9789811250477_0031)
1. [2022 CHIL '20: Proceedings of the ACM Conference on Health, Inference, and Learning] **Disease state prediction from single-cell data using graph attention networks** [[paper]](https://dl.acm.org/doi/10.1145/3368555.3384449)




