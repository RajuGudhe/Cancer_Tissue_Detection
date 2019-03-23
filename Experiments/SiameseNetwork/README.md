
# Siamese Network with triplet loss

This is the PyTorch implemtation of Siamese Network.  Siamese Networks can learn useful data descriptors that can be further used as features for classification. The Siamese Network consists of two identical Convolution Neural Networks that are trained by sharing same weights. 
The architecture used in this work is shown below.
<p align="center">
 <img src="./images/dense_arch.png" alt="Drawing" width="50%">
</p>

## Requirements

* Python 3
* PyTorch (newest version)
* tqdm
* tensorboard_logger
## Code
Currently the code supports CIFAR-10 dataset, Will upload the code to work with PCam datasets(Need to resolve few bugs while loading the data and test the code).
To run the code :
 ```
python main.py --help
```



# References


[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015

[3] Alexander Hermans, Lucas Beyer, Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737), 2017

