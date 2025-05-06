# **Normal breast tissue classifiers assess large-scale tissue compartments with high accuracy**

[Paper]() | [Cite]()

**Abstract:** Cancer research emphasises early detection, yet quantitative methods for normal tissue analysis remain limited. Digitised haematoxylin and eosin (H&E)-stained slides enable computational histopathology, but artificial intelligence (AI)-based analysis of normal breast tissue (NBT) in whole slide images (WSIs) remains scarce. We curated 70 WSIs of NBTs from multiple sources and cohorts with pathologist-guided manual annotations of epithelium, stroma, and adipocytes (https://github.com/cancerbioinformatics/OASIS). Using this dataset, we developed robust convolutional neural network (CNN)-based, patch-level classification models, named NBT-Classifiers, to tessellate and classify NBTs at different scales. Across three external cohorts, NBT-Classifiers trained on 128 x 128 µm and 256 x 256 µm patches achieved AUCs of 0.98–1.00. Two explainable AI-visualisation techniques confirmed the biological relevance of tissue class predictions. Moreover, NBT-Classifiers can be integrated into an end-to-end pre-processing framework to support efficient downstream image analysis in lobular regions. Their high compatibility with QuPath further enables broader application in studies of normal tissues, in the context of breast.


<p align="center">
    <img src="data/NBT.png" width="100%">
</p>

## Installation
To get started, install [HistoQC](https://github.com/choosehappy/HistoQC.git) and NBT-Classifier:
```
git clone https://github.com/choosehappy/HistoQC.git
git clone https://github.com/SiyuanChen726/NBT-Classifier.git
cd NBT-Classifier
conda env create -f environment.yml
conda activate nbtclassifier
```

## Docker
NBT-Classifier supports Docker for reproducible analysis of user histology data, with tutorial examples for both command-line and Jupyter notebook workflows. 

To get the Docker:

`docker pull siyuan726/nbtclassifier:latest`

or use singularity for HPC

`singularity pull docker://siyuan726/nbtclassifier:latest`


Host data is expected to be organised as follows:
```
project/
├── WSIs/slide1.ndpi, slide2.ndpi, slide3.svs, ...
├── QCs/
└── FEATUREs/
```


The Docker Image has an exposed volume (/app) that can be mapped to the host system directory. For example, to mount the current directory:

The following code launches Singularity container with:
- NVIDIA GPU support (--nv)
- Host WSI directory mounted to /app/WSIs (--bind)
- Temporary writable filesystem (--writable-tmpfs)
  
```
singularity shell --nv \ 
--bind /the/host/WSI/directory:/app/project \
--writable-tmpfs 
./nbtclassifier_latest.sif
```


You will see an app folder under "root":
```
/app/
├── NBT-Classifier/
├── HistoQC/
├── Dockerfile
├── project/
|   ├── WSIs/host slides, such as slide1.ndpi, slide2.ndpi, slide3.svs, ...
|   ├── QCs/
|   └── FEATUREs/
└── examples/
    ├── WSIs/17064108_FPE_1.ndpi
    ├── QCs/
    ├── FEATUREs/
    ├── patch_examples/
    ├── QuPath/
    ├── TC512_tsne_HEOverlay.png  
    └── NBTClassifier_512px_externaltesting.csv
```

## Implementation using host data 
First, implement HistoQC to obtain masks of foreground tissue regions:
```
cd /app/HistoQC
python -m histoqc -c NBT -n 3 '/app/project/WSIs/*.ndpi' -o '/app/project/QCs'
```
Note, change `.ndpi` into the exact format of the host WSI files


This step yields:
```
/app/
├── NBT-Classifier/
├── HistoQC/
├── Dockerfile
├── project/
|   ├── WSIs/
|   ├── QCs/
|   │   |── slide1/slide1_maskuse.png, ... 
|   │   |── slide2/slide2_maskuse.png, ...
|   │   |── slide3/slide3_maskuse.png, ...
|   │   └── ... 
|   └── FEATUREs/
└── examples/       
```


Then, use the following script to classify NBT tissue components:
```
cd /app//NBT-Classifier
python main.py \
--wsi_folder /app/project/WSIs \
--mask_folder /app/project/QCs \
--output_folder /app/project/FEATUREs \
--model_type TC_512 \
--patch_size_microns 128 \
--use_multithreading \
--max_workers 8
```

This step yields:
```
/app/
├── NBT-Classifier/
├── HistoQC/
├── Dockerfile
├── project/
|   ├── WSIs/
|   ├── QCs/
|   └── FEATUREs/
|       |── slide1/
│       |   ├──slide1_TC_512_pattern_x_idx.npy     
│       |   ├──slide1_TC_512_pattern_y_idx.npy     
│       |   ├──slide1_TC_512_pattern_im_shape.npy  
│       |   ├──slide1_TC_512_pattern_patches.npy    
│       |   ├──slide1_TC_512_probmask.npy                     # This contains the tissue classification results
│       |   ├──slide1_TC_512.png                              # This visualises the tissue classification map
│       |   ├──slide1_TC_512_patch_all.csv                    # This saves all classified patches
│       |   ├──slide1_TC_512_cls_wsi.json                     # This imports all classified patches into QuPath via the annotation_loader.groovy script
│       |   ├──slide1_TC_512_epi_(18,0,0,8704,6208)-mask.png  # This imports detected lobuels into QuPath via the mask2annotation.groovy script
│       |   ├──slide1_TC_512_patch_roi.csv                    # This saves the selected patches from ROIs containing lobules and peri-lobular stroma
│       |   ├──slide1_TC_512_cls_roi.json                     # This imports selected patches into QuPath using the annotation_loader.groovy script
│       |   └──slide1_TC_512_bbx.png                          # This visualises the selected ROIs
│       └── ...
└── examples/

```


Alternatively, tessellating and classifying NBTs using larger patches of 1024x1024 pixels:
```
cd /app//NBT-Classifier
python main.py \
--wsi_folder /app/project/WSIs \
--mask_folder /app/project/QCs \
--output_folder /app/project/FEATUREs \
--model_type TC_1024 \
--patch_size_microns 256 \
--use_multithreading \
--max_workers 8
```


 
For a full implementation of **_NBT-Classifier_**, please take a look at [notebook pipeline](pipeline.ipynb). 
