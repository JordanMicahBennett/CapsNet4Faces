# Face Classification with Capsule Network built in Tensorflow

A Tensorflow implementation of CapsNet(Capsules Net) apply on the Labelled Faces in the Wild (LFW) dataset based on [thibo73800's Traffic Sign Classifier](https://github.com/thibo73800/capsnet-traffic-sign-classifier)

This implementation is based on this paper: <b>Dynamic Routing Between Capsules</b> (https://arxiv.org/abs/1710.09829) from Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton.

The code for the CapsNet is located in the following file: <b>caps_net.py</b> while the whole model is created inside the <b>model.py</b> file. The two main methods used to build the CapsNet are  <b>conv_caps_layer</b> and <b>fully_connected_caps_layer</b>


## Requirements
- Python 3
- NumPy 1.13.1
- Tensorflow 1.3.0
- docopt 0.6.2
- Sklearn: 0.18.1
- Matplotlib

## Install

    $> git clone https://github.com/krishnr/CapsNet4Faces.git
   
## Train

    $> python train_for_faces.py

During the training, the checkpoint is saved by default into the outputs/checkpoints/ folder. The exact path and name of the checkpoint is print during the training.

## Test

In order to measure the accuracy and the loss on the Test dataset you need to used the test.py script as follow:

    $> python test_all_faces.py outputs/checkpoints/ckpt_name

## Metrics

<b>Accuracy: </b>
<ul>
    <li>Train: 100%</li>
    <li>Test: 93.7%</li>
</ul>

Checkpoints and tensorboard files are stored inside the <b>outputs</b> folder.



