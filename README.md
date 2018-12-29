# fastai-tf-fit
Fit your Tensorflow model using fastai and PyTorch

## Installation
```bash
pip install git+https://github.com/fastai/tf-fit.git
```

## Features
This project is an extension of fastai to allow training of Tensorflow models with a similar interface of fastai. It uses fastai `DataBunch` objects so the interface is exactly the same for loading data. For training, the `TfLearner` has many of the same features as the fastai `Learner`. Here is a list of the currently supported features.
* Training Tensorflow models with constant learning rate and weight decay
* Training using the [1cycle policy](https://docs.fast.ai/train.html#fit_one_cycle)
* Learning rate finder
* Fit with callbacks with access to hyper parameter updates
* Discriminative learning rates
* Freezing layers from having parameters trained
* [True weight decay option](https://arxiv.org/abs/1711.05101)
* L2 regularization (true_wd=False)
* [Removing weight decay from batchnorm layers option (bn_wd=False)](https://arxiv.org/abs/1706.02677)
* Momentum
* Option to train batchnorm layers even if the layer is frozen (train_bn=True)
* Model saving and loading
* Default image data format is channels * hieght * width

## To do
This project is a work in progress so there may be missing features or obscure bugs.
* Get predictions function
* Tensorflow train/eval functionality for dropout and batchnorm in eager mode
* Pip and conda packages

## Examples

### Setup
Setup fastai data bunch, optimizer, loss function, and metrics.
```python
from fastai.vision import *
from fastai_tf_fit import *

path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512).normalize(cifar_stats)

opt_fn = tf.train.AdamOptimizer

loss_fn = tf.losses.sparse_softmax_cross_entropy

def categorical_accuracy(y_pred, y_true):
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, tf.keras.backend.argmax(y_pred, axis=-1)))
metrics = [categorical_accuracy]
```

### Using tf.keras.Model
```python
class Simple_CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.conv3 = tf.keras.layers.Conv2D(10, kernel_size=3, strides=(2,2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=1)
    def call(self, xb):
        xb = tf.nn.relu(self.bn1(self.conv1(xb)))
        xb = tf.nn.relu(self.bn2(self.conv2(xb)))
        xb = tf.nn.relu(self.bn3(self.conv3(xb)))
        xb = tf.nn.pool(xb, (4,4), 'AVG', 'VALID', data_format="NCHW")
        xb = tf.reshape(xb, (-1, 10))
        return xb

model = Simple_CNN()
```



### Using Keras functional API
```python
inputs = tf.keras.layers.Input(shape=(3,32,32))
x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')(inputs)
x = tf.keras.layers.BatchNormalization(axis=1)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')(x)
x = tf.keras.layers.BatchNormalization(axis=1)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(10, kernel_size=3, strides=(2,2), padding='same')(x)
x = tf.keras.layers.BatchNormalization(axis=1)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), padding='same')(x)
x = tf.keras.layers.Reshape((10,))(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
```

### Training
Create TfLearner object
```python
learn = TfLearner(data, model, opt_fn, loss_fn, metrics=metrics, true_wd=True, bn_wd=True, wd=defaults.wd, train_bn=True)
```

Learning rate finder.
```python
learn.lr_find()
learn.recorder.plot()
```

Train the model for 3 epochs with a learning rate of 3e-3 and weight decay of 0.4.
```python
learn.fit(3, lr=3e-3, wd=0.4)
```

Fit the model using 1cycle policy with a cycle length of 10 using a discriminative learning rate.
```python
learn.fit_one_cycle(10, max_lr=slice(6e-3, 3e-3))
```

Freeze, unfreeze, and freeze to last layers from training.
```python
learn.freeze()
```
```python
learn.unfreeze()
```
```python
learn.freeze_to(-1)
```

Save and load model weights.
```python
learn.save('cnn-1')
```
```python
learn.load('cnn-1')
```

### Metrics
Plot learning rate and momentum schedules.
```python
learn.recorder.plot_lr(show_moms=True)
```

Plot train and validation losses.
```python
learn.recorder.plot_losses()
```

Plot metrics.
```python
learn.recorder.plot_metrics()
```
