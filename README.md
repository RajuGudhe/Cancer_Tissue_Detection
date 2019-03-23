# Cancer_Tissue_Detection

This is the implementation of Dense Siamese Network for the detection of cancer tissues in histopathology scans.

# Datasets
## PatchCamelyon(PCam)


The PatchCamelyon(PCam) is a new benchmark dataset for medical image classification. It consists of 3,27,680 color images of size 96 × 96 pixels patches extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating the presence of tumor cells.



![Example Annotated Images from PCam dataset](https://cdn-images-1.medium.com/max/800/1*MAAzPX-f5uejYE2e2jsXeA.png)

# Download 

The data is provided under the CCO License, Data download  [
link](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB).

# Usage
The dataset is available in HDF5 files with train, valid, test split. Each set contains the data and target file. In this work the data is preprocessed into PyTorch ImageFolder format and the structure of data folder is as follows:
```bash
├── PCam-data
	├── train
	│   ├── tumor
	│   ├── no-tumor
    ├── valid
	│   ├── tumor
	│   ├── no-tumor
	├── test
	    ├── tumor
	    ├── no-tumor

```

## Citations
Biblatex entry:
```bash
@ARTICLE{Veeling2018-qh,
  title         = "Rotation Equivariant {CNNs} for Digital Pathology",
  author        = "Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and
                   Cohen, Taco and Welling, Max",
  month         =  jun,
  year          =  2018,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "1806.03962"
}
```

# Acknowledgements
- Data image folder is created based on [
this](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) PyTorch tutorial.
- Data creation was inspired from [this](https://medium.com/@meghana97g/classification-of-tumor-tissue-using-deep-learning-fastai-77252ae16045) blog.

