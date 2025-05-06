# **Normal breast tissue classifiers assess large-scale tissue compartments with high accuracy**

[Paper]() | [Cite]()

**Abstract:** Cancer research emphasises early detection, yet quantitative methods for normal tissue analysis remain limited. Digitised haematoxylin and eosin (H&E)-stained slides enable computational histopathology, but artificial intelligence (AI)-based analysis of normal breast tissue (NBT) in whole slide images (WSIs) remains scarce. We curated 70 WSIs of NBTs from multiple sources and cohorts with pathologist-guided manual annotations of epithelium, stroma, and adipocytes (https://github.com/cancerbioinformatics/OASIS). Using this dataset, we developed robust convolutional neural network (CNN)-based, patch-level classification models, named NBT-Classifiers, to tessellate and classify NBTs at different scales. Across three external cohorts, NBT-Classifiers trained on 128 x 128 µm and 256 x 256 µm patches achieved AUCs of 0.98–1.00. Two explainable AI-visualisation techniques confirmed the biological relevance of tissue class predictions. Moreover, NBT-Classifiers can be integrated into an end-to-end pre-processing framework to support efficient downstream image analysis in lobular regions. Their high compatibility with QuPath further enables broader application in studies of normal tissues, in the context of breast.


<p align="center">
    <img src="data/NBT.png" width="100%">
</p>

## Installation
To get started, clone the repository, install [HistoQC](https://github.com/choosehappy/HistoQC.git) and other required dependencies. 
Then install NBT-Classifier:
```
git clone https://github.com/SiyuanChen726/NBT-Classifier.git
cd NBT-Classifier
conda env create -f environment.yml
conda activate nbtclassifier
```

## Data structure

Data is expected to be organised as follows:
```
project/
├── NBT-Classifier/
├── HistoQC/
└── project-data/
    ├── WSIs/slide1.ndpi, slide2.ndpi, slide3.svs, ...
    ├── QCs/
    └── FEATUREs/
```

## Implementation
First, implement HistoQC to obtain masks of foreground tissue regions:
```
cd ./HistoQC
python -m histoqc -c NBT -n 3 '../project-data/WSIs/*.ndpi' -o '../project-data/QCs'
```
This step yields:
```
project/
├── NBT-Classifier/
├── HistoQC/
└── project-data/
    ├── WSIs/slide1.ndpi, slide2.ndpi, slide3.svs, ...
    ├── QCs/              
    │   └── slide1/slide1_maskuse.png  
    └── FEATUREs/         
```

Then, use the following script to classify NBT tissue components:
```
cd ../NBT-Classifier
python main.py \
--wsi_folder ../project-data/WSIs \
--mask_folder ../project-data/QCs \
--output_folder ../project-data/FEATUREs \
--model_type TC_512 \
--patch_size_microns 128 \
--use_multithreading \
--max_workers 8
```

This step yields:
```
prj_BreastAgeNet/
├── WSIs
├── QC/KHP
│   ├── slide1/slide1_maskuse.png
│   └── ...
├── Features/KHP
│   ├── slide1/slide1_TC_512_probmask.npy     # This is the tissue classification results
│   ├── slide1/slide1_TC_512.png              # This visualises the tissue classification map
│   ├── slide1/slide1_TC_512_All.csv          # This saves all classified patches
│   ├── slide1/slide1_TC_512_cls.json         # This imports all classified patches into QuPath using the annotation_loader.groovy script
│   ├── slide1/slide1_TC_512_epi_(downsample,0,0,width,height)-mask.png      # This imports detected lobuels into QuPath using the mask2annotation.groovy script
│   ├── slide1/slide1_TC_512_patch.csv        # This saves the selected patches
│   ├── slide1/slide1_TC_512_ROIdetection.json     # This imports selected patches into QuPath using the annotation_loader.groovy script
│   ├── slide1/slide1_TC_512_bbx.png          # This visualises the selected ROIs
│   └── ...


project/
├── NBT-Classifier/
├── HistoQC/
└── project-data/
    ├── WSIs/slide1.ndpi, slide2.ndpi, slide3.svs, ...
    ├── QCs/              
    │   └── slide1/slide1_maskuse.png  
    └── FEATUREs/
        └── slide1/
            ├──
            ├──
            ├──
            ├──
            ├──
            ├──
            ├──
            ├──
            └──


'17064108_FPE_1.ndpi_epi_(18,0,0,8704,6208)-mask.png'

 17064108_FPE_1_TC_512_All.csv
 17064108_FPE_1_TC_512_bbx.png
 17064108_FPE_1_TC_512_cls.json
 17064108_FPE_1_TC_512_cls_roi.json
 17064108_FPE_1_TC_512_cls_wsi.json
'17064108_FPE_1_TC_512_epi_(18,0,0,8704,6208)-mask.png'
 17064108_FPE_1_TC_512_patch_all.csv
 17064108_FPE_1_TC_512_patch.csv
 17064108_FPE_1_TC_512_patch_roi.csv
 17064108_FPE_1_TC_512_pattern_im_shape.npy
 17064108_FPE_1_TC_512_pattern_patches.npy
 17064108_FPE_1_TC_512_pattern_x_idx.npy
 17064108_FPE_1_TC_512_pattern_y_idx.npy
 17064108_FPE_1_TC_512.png
 17064108_FPE_1_TC_512_probmask.npy
 17064108_FPE_1_TC_512_ROIdetection.json
```
 
 
For a full implementation of **_NBT-Classifier_**, please take a look at [notebook pipeline](pipeline.ipynb). 
