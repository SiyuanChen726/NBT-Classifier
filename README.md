# **A CNN-based _NBT-Classifier_ facilitates analysing different normal breast tissue compartments on whole slide images**

**Abstract:** Whole slide images (WSIs) are digitized tissue slides increasingly adopted in clinical practice and serve as promising resources for histopathological research through advanced computational methods. Recognizing tissue compartments and identifying regions of interest (ROIs) are fundamental steps in WSI analysis. In contrast to breast cancer, tools for high-throughput analysis of WSIs derived from normal breast tissue (NBT) are limited, despite NBT being an emerging area of research for early detection. We collected 70 WSIs from multiple NBT resources and cohorts, along with pathologist-guided manual annotations, to develop a robust convolutional neural network (CNN)-based classification model, named _NBT-Classifier_, which categorizes three major tissue compartments: epithelium, stroma, and adipocytes. The two versions of _NBT-Classifier_, processing 512 x 512- and 1024 x 1024-pixel input patches, achieved accuracies of 0.965 and 0.977 across three external datasets, respectively. Two explainable AI visualization techniques confirmed the histopathological relevance of the high-attention patterns associated with predicting specific tissue classes. Additionally, we integrated a WSI pre-processing pipeline to localize lobules and peri-lobular regions in NBT, the output from which is also compatible with interactive visualization and built-in image analysis on the QuPath platform. The _NBT-Classifier_ and the accompanying pipeline will significantly reduce manual effort and enhance reproducibility for conducting advanced computational pathology (CPath) studies on large-scale NBT datasets.


## Installation
To get started, clone the repository, install [HistoQC](https://github.com/choosehappy/HistoQC.git) and other required dependencies. 
```
git clone https://github.com/SiyuanChen726/NBT-Classifier.git
cd NBT-Classifier
conda env create -f environment.yml
conda activate tfgpu-env
```

## Implementation
```
```
