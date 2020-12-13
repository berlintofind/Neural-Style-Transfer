# Neural-Style-Transfer
Neural Style Transfer Application with VGG-19

By Tianyang Zhao

## Table of Contents
1. Introduction
2. Models
3. Results
4. Reference

## Introduction
This repository contains the Neural Style Transfer Application. The goal of the model is to use a "content" image (C) and a "style" image (S), to create a "generated" image (G).

#### Packages Version
1. Python 3.6.9 
2. tensorflow-gpu 1.13.1
3. numpy 1.19.1
4. CUDA Version 10.1
5. scipy 1.2.1
6. keras-gpu 2.3.1


## Models
Neural Style Transfer allows the use to generate his own artistic pictures with the help of a "style" image. 

VGG-19, a 19-layer version of the VGG network will be used in the model. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers). It's widely used in image related problems. This step is a transfer learning actually. To save the sapce of my Github, the pre-trained is not uploaded in this repository. Users could download it from [MatConvNet.](https://www.vlfeat.org/matconvnet/pretrained/)

Different from most of the neural networks, this application is built in the optimization algorithm whhihc updates the pixel values rather than the neural network's parameters. Optimizing the total cost function results in synthesizing new images.


To test the model on your own images，the use only need to upload his own images and change the name with his own image in line 21 and 124 of NeuralStyle.py.

#### Note
image size requirement: wideth 800, height 600

## Result

the generated picture of "louvre" and "claude-monet":
![image](https://github.com/berlintofind/YOLO_v2_Objective_Detection/blob/master/out/test.jpg)

## Reference
1. [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015) A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
2. [Harish Narayanan Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
3. [Log0,TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
4. [Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition MatConvNet.](https://arxiv.org/pdf/1409.1556.pdf)
