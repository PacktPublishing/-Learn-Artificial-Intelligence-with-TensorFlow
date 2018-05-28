# Section 2: Computer Vision

Contains the code for building and training a convolutional neural network on
the CIFAR-10 dataset.

The first time this code is run, the CIFAR-10 dataset will be downloaded for you. 

Brief file descriptions:
- `cifar10_estimator.py`: custom tf.estimator.Estimator implemention of the CNN. This is the main file that should be run. It uses `cifar10_input` to get the data and build the input pipeline, and uses `cifar10_model` to link the logits, loss, and training operations together in the custom estimator's `model_fn`.
- `cifar10_input.py`: contains the code for checking whether the user has the CIFAR-10 dataset, and downloading it for them if it is not found.
- `cifar10_model.py`: contains all the model-building code, such as the convolutional layers, the loss function, the optimizer, etc.

Example command to train the model:
```bash
./cifar_10_estimator.py --data_dir=$PWD/../data/cifar --batch_size=128
```

## Additional Resources

### 2.1: Convolutional Neural Networks

- [CS231n CNNs for Visual Recognition](http://cs231n.github.io/convolutional-networks/).
- [Chapter 9 of Deep Learning](https://www.deeplearningbook.org/contents/convnets.html): Convolutional Networks.
- [A Guide to TF Layers](https://www.tensorflow.org/tutorials/layers): Building a Convolutional Neural Network.

### 2.2: Preprocessing, Pooling, and Batch Normalization

- [Images with TensorFlow](https://www.tensorflow.org/api_guides/python/image).
- [Batch Normalization: Accelerate Deep Network Training by Reducing Internal Covariate Shift](arxiv.org/abs/1502.03167) by Sergey Ioffe and Christian Szegedy.

### 2.3/2.4: Training a CNN on CIFAR-10

- [Convolutional Neural Networks](https://www.tensorflow.org/tutorials/deep_cnn) TensorFlow tutorial with CIFAR-10. The code for this section began as the tutorial code, but was refactored to use more up-to-date tensorflow practices and improve readability.
- [TensorFlow Estimators](https://www.tensorflow.org/programmers_guide/estimators) Programmer's Guide.
- [Importing Data](https://www.tensorflow.org/programmers_guide/datasets) Programmer's Guide.
- [Creating Custom Estimators](https://www.tensorflow.org/get_started/custom_estimator).

