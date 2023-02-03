# netfieldv1

netfieldv1 is a simple platform for Convolutional Neural Networks applied on the classification of 3-channeled images. The platform is designed using [Python3] without the usage of high level APIs for deep learning, but solely based on the [NumPy] (for CPU) and [CuPy] (for GPU mainly) libraries. The following documentation refers to machines equipped with a Debian GNU/Linux OS.

### Capabilities

Its training iterations can be performed per minibatch, and its layers are fully vectorized. Currently the available layer types are:

  - Convolutional
  - Max Pooling
  - Average Pooling
  - Flattening
  - Fully Connected

The platform uses the Categorical Cross Entropy cost function for the models validation, and offers a configurable tool for a graphical representation of model validation metrics like accuracy and loss. Also the platform backs up the learnable parameters during a training session depending on the backup period argument which the user chooses through the respective configuration file.
Currently the available activation functions are:

   - ReLU
   - Sigmoid
   - Softmax (used on the Output Layer)

Available Optimization methods:
    
   - Gradient Descent (vanilla)
   - Gradient Descent with Momentum
   - AdaGrad
   - RMSProp
   - ADAM

It supports GPU acceleration on CUDA enabled GPUs. 

### Requirements and Installation

netfieldv1 uses a number of libraries to work properly:

  - [NumPy] - NumPy is a tool that helps us perform optimized tensor manipulations in a linear algebra based intuition, thus it's all over the place when it comes to neural networks like CNNs
  - [idx2numpy] - Used for parsing IDX files
  - [CuPy] - CuPy is used as the GPU equivalent of NumPy, it can be used on NVIDIA GPUs that include CUDA cores to reduce execution time
  - [Matplotlib] - A great tool to project our model validation metrics and also to view our image of interest
  - [pandas] - This is used to list our model validation metrics in a .csv file and proves to be incredibly convenient when it comes to organizing a set of results.
  - [Pillow] - This is used to project an image (used after a classification to show the result)

To install these dependencies this guide uses [pip], a terminal based package installer for Python. Now to begin, open a linux terminal and follow the next instructions accordingly to install all the dependencies.

#### Step 1

The following installs pip.
```sh
$ sudo apt install python3-pip
```

#### Step 2

Now to install each dependency using pip.
```sh
$ python3 -m pip install numpy
$ python3 -m pip install idx2numpy
$ python3 -m pip install matplotlib
$ python3 -m pip install pandas
$ python3 -m pip install Pillow
```

#### Step 3

This step is necessary if one plans to use GPU acceleration.

a) Carefully follow the instructions given by https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html to install the CUDA toolkit
<br>
b) Now to install [CuPy]
```sh
$ python3 -m pip install cupy
```

and we're ready.

### Technical Overview

As one wills to specify how the platform will work, the program first needs some input files. To begin, it needs a configuration file that provides the platform with the paths of all the other files which will be used to specify the training or/and classification session, along with some fields which specify how the dataset will be preprocessed. That configuration file will be named as `<config>`. The rest of the files that `<config>` specifies are: the .json file `<cnn_config>` which holds the CNNs architecture and some other minor options, the .txt file `<class_names>` which holds each corresponding label for the classification problem, the path to the learnable parameters stored in a .npy file `<lparams>`, and the dataset file paths are specified by 4 distinct .txt files, `<training_features>`, `<training_target_variables>`, `<val_features>` and `<val_target_variables>`. `<config>` holds a special field called `"dataset_parsing_mode"`. This field either takes as an input the string `"four_files"`, which corresponds to the case of the dataset being splitted into four files, `<training_features_file>`, `<training_target_variables_file>`, `<val_features_file>` and `<val_target_variables_file>`. These files can either be stored as npy or IDX file formats. Now the second and final option for the same field, is the string `"agent_wise"`. Using this option the platform should instead be fed with a dataset that is splitted by each agent/example in npy files (IDX file format will not be supported for the `"agent_wise"` option). The latter parsing method is obviously computationally more demanding and generally less convenient so it's more preferable to use the first input in that field.
Now each file has a very specific syntax and a set of inputs that needs to be taken into account:
<br>
`<config>`: .json file, provides the paths of all the required files that fully specify the training sequence along with some options. Some of these options will not be necessary if we use netfieldv1 to classify an image,
```
{
  "dataset_parsing_mode": <acceptable values: {"four_files", "agent_wise"}, specifies in which way the dataset will be parsed>,

  "training_set":
  {
    "training_features": <path of the <training_features> .txt file>,
    "training_target_variables": <path of the <training_target_target_variables> .txt file>
  },

  "val_set":
  {
    "val_features": <path of the <val_features> .txt file>,
    "val_target_variables": <path of a <val_target_variables> .txt file>
  },

  "normalize_input": <int, acceptable values: {0, 1}, determines if the input will be normalized, we use 1 to perform normalization on the input, otherwise we use 0>,
  "cnn_config": <path to the <cnn_config> .json file>,
  "class_names": <path to the <class_names> .txt file>,
  "lparams_save_dir": <path to the <lparams_save_dir>, this is used in case if we want to specify the backup location of the learnable parameters during a training session>,
  "lparams_load_file": <path of the <lparams> .npy file, if left blank then new ones are generated using Uniform Xavier distributions>
}
```
Now on that file, if we use the option `"dataset_parsing_mode": "four_files"` then all the .txt files that specify the locations of the dataset files need to have exactly one line each. These files are specified by the fields `"training_set/training_features"`, `"training_set/training_target_variables"`, `"val_set/val_features"` and `"val_set/val_target_variables"`. Or in other words, using this options, each file looks like this
<br>
`<training_features>`:
```
<path_to_the_file_that_contains_all_the_training_features>
```
`<training_target_variables>`:
```
<path_to_the_file_that_contains_all_the_training_target_variables>
```
`<val_features>`:
```
<path_to_the_file_that_contains_all_the_val_features>
```
`<val_target_variables>`:
```
<path_to_the_file_that_contains_all_the_val_target_variables>
```
Otherwise in case we use the option `"dataset_parsing_mode": "agent_wise"` then each line of each file corresponds to the index of an example, be it either a training or a validation one. This time the .txt files should look like this
<br>
`<training_features>`:
```
<path_to_the_training_feature_file_0>
<path_to_the_training_feature_file_1>
<path_to_the_training_feature_file_2>
...
<path_to_the_training_feature_file_m-1>
```
`<training_target_variables>`:
```
<path_to_the_training_target_variable_file_0>
<path_to_the_training_target_variable_file_1>
<path_to_the_training_target_variable_file_2>
...
<path_to_the_training_target_variable_file_m-1>
```
`<val_features>`:
```
<path_to_the_val_feature_file_0>
<path_to_the_val_feature_file_1>
<path_to_the_val_feature_file_2>
...
<path_to_the_val_feature_file_r-1>
```
`<val_target_variables>`:
```
<path_to_the_val_target_variable_file_0>
<path_to_the_val_target_variable_file_1>
<path_to_the_val_target_variable_file_2>
...
<path_to_the_val_target_variable_file_r-1>
```
the positive integers m, r correspond to the cardinality of the training and validation sets respectively. The delimiter that's used is obviously the newline character.
<br>
`<cnn_config>`: .json file, contains all the CNNs hyperparameters along with other important options
```
{
  "training":
  {
    "epochs": <number of training epochs>,
    "backup_period": <number of training epochs required for each backuping up of the learnable parameters>,
    "shuffle_input": <acceptable values: {0, 1}, specifies if we want to apply shuffling to the input indices>,
    "minibatch_size": <preferable minibatch size, the final minibatch is produced as a remainder rather as the preferable size>,
    <OPTIMIZATION_ALGORITHM>
    "cost_function": <the name of the cost function>
  },

  "input":
  {
    "input_height": <height of input RGB image>,
    "input_width": <width of input RGB image>,
    "input_channels": <acceptable values: {1, 3}, multitude of channels of input image>
  },

  <CONV_PART_HIDDEN_LAYERS>

  "flattening":
  {
  },

  <FC_PART_HIDDEN_LAYERS>

  "softmax_output":
  {
  }
}
```
The `<OPTIMIZATION_ALGORITHM>`, `<CONV_PART_HIDDEN_LAYERS>`, `<FC_PART_HIDDEN_LAYERS>` chunks can vary.
<br>
`<OPTIMIZATION_ALGORITHM>`: chunk of `<cnn_config>` that specifies the optimization algorithm, and can take exactly one of the following forms
  - Regular Gradient Descent
```
    "learning_method":
    {
      "name": "regular_gd",
      "args":
      {
        "learning_rate": <float type>
      }
    },
```
  - Regular Gradient Descent with Momentum
```
    "learning_method":
    {
      "name": "regular_gd_with_momentum",
      "args":
      {
        "learning_rate": <float type>,
        "momentum_coeff": <float type, this is the momentum coefficient>
      }
    },
```
  - AdaGrad Gradient Descent
```
    "learning_method":
    {
      "name": "adagrad_gd",
      "args":
      {
        "epsilon": <float type, this is the global learning rate, preferable value is 0.001>,
        "c": <float type, used for numerical stability, preferable value is 0.0000001>
      }
    },
```
  - RMSProp Gradient Descent
```
    "learning_method":
    {
      "name": "rmsprop_gd",
      "args":
      {
        "epsilon": <float type, this is the global learning rate, preferable value is 0.001>,
        "rho": <float type, decay rate. Preferable value is 0.9>,
        "c": <float type, used for numerical stability, preferable value is 0.0000001>
      }
    },
```
  - ADAM Gradient Descent
```
    "learning_method":
    {
      "name": "adam_gd",
      "args":
      {
        "epsilon": <float type, this is the step size, preferable value is 0.001>,
        "rho1": <float type, this is the moment estimate 1, preferable value is 0.9>,
        "rho2": <float type, this is the moment estimate 1, preferable value is 0.999>,
        "c": <float type, used for numerical stability, preferable value is 0.0000001>
      }
    },
```
`<CONV_PART_HIDDEN_LAYERS>`: chunk of `<cnn_config>` that specifies the CNNs architecture after the input layer and before the flattening layer. Can take any permutation from the following set of layers,
  - Convolutional Layer
```
  "convolutional":
  {
    "output_channels": <int type, this is the number of the output channels, or equivalently the number of filter for the layer>,
    "filter_size": <int type, this is the length of each square filter/convolution kernel applied to the current convolutional layer>,
    "padding": <int type, the size of the padding>,
    "stride": <int type, the convolutions stride>,
    "dropout_rate": <float type, percentage of output neurons we want to drop (or substitude with zero)>
    <ACTIVATION_FUNCTION>
  },
```
  - Average Pooling Layer
```
  "avgpooling":
  {
    "size": <int type, the size of the average pooling scan acted upon each channel slice of the input>,
    "padding": <int type, the size of the padding>,
    "stride": <int type, the convolutions stride>,
    "dropout_rate": <float type, percentage of output neurons we want to drop (or substitude with zero)>
  },
```
  - Max Pooling Layer
```
  "maxpooling":
  {
    "size": <int type, the size of the max pooling scan acted uppon each inputs channel slice>,
    "padding": <int type, the size of the padding>,
    "stride": <int type, the convolutions stride>,
    "dropout_rate": <float type, percentage of output neurons we want to drop (or substitude with zero)>
  },
```
`<FC_PART_HIDDEN_LAYERS>`: chunk of `<cnn_config>` that specifies the CNNs architecture after the flattening layer and before the output layer. Can take any permutation from the following set of layers,
```
  "fully_connected":
  {
    "output_size": <int type, the size of the output given an example>,
    "dropout_rate": 0.5,
    <ACTIVATION_FUNCTION>
  },
```
`<ACTIVATION_FUNCTION>`: chunk of layer types that include learnable parameters. Can take exactly one of the following forms
  - Sigmoid Activation Function
```
    "activation_function": "sigmoid"
```
  - ReLU Activation Function
```
    "activation_function": "relu"
```
`<class_names>`: This .txt file contains each class name in the following way
```
<class_name_0>
<class_name_1>
<class_name_2>
...
<class_name_c-1>
```
where c is the multitude of classes for the classification problem.
`<lparams>`: .npy pickled file that contains the learnable parameters for a given CNN architecture.

### Directory Layout

The hierarchy of directories, inside the netfieldv1 directory, is the following:
```
.
├── classnames # Holds the <classnames> files.
│   └── MNIST_classnames.txt
├── cnn_config # Holds the <cnn_config> files
│   └── convnet1.json
├── config # This directory holds the <config> files.
│   ├── option1_classification.json
│   └── option1.json
├── datasets
│   └── MNIST
│       ├── val_features_loc.txt
│       ├── val_set
│       │   ├── t10k-images-idx3-ubyte
│       │   └── t10k-labels-idx1-ubyte
│       ├── val_target_variables_loc.txt
│       ├── training_features_loc.txt
│       ├── training_set
│       │   ├── train-images-idx3-ubyte
│       │   └── train-labels-idx1-ubyte
│       └── training_target_variables_loc.txt
├── lparams # This is where we store the learnable parameter files along with a special .csv file that contains the history of a training sessions metrics.
│   └── metrics_history.csv
├── src
│   ├── classify.py # We use this to classify an image.
│   ├── cnnplot.py # We use this to visualize the historical metrics for a training session stored in the metrics_history.csv .
│   ├── cnnprint.py
│   ├── cnn.py
│   ├── gradient_checking.py
│   ├── gradient_descent.py
│   ├── layerexec
│   │   ├── actfunction.py
│   │   ├── avgpooling.py
│   │   ├── convolutional.py
│   │   ├── dropout.py
│   │   ├── flattening.py
│   │   ├── fullyconnected.py
│   │   ├── __init__.py
│   │   ├── maxpooling.py
│   │   └── pad.py
│   ├── lparams_init.py
│   ├── model_efficiency_metric.py
│   ├── parser.py
│   ├── tensor_module.py
│   └── train.py # We use this to train a model.
├── gradient_checking.sh
├── train.sh
└── classify.sh
```
the MNIST dataset is already on the `datasets` directory, and it's splitted into 4 files, in the way that it can be parsed using the option `dataset_parsing_mode: "four_files"` on our `<config>` file.

### Training a Model

To train a model first we need to have the dataset split by either each agent/example, or split in four files in the way that was clarified on the Technical Overview section. Then we need to store the paths of these files in the respective .txt files and we should normalize the features before the training starts by setting the option `"normalize_input"` of `<config>` to 1. Also we need to specify the CNN using the `<cnn_config>` file, set the label names on `<class_names>`, and should also set `<lparams_save_dir>` so that we'd enable the occassional backing up of the learnable parameters (depending on the `"backup_period"` option on the `<cnn_config>` file). As for the `<lparams_load_file>`, if we run the platform for the first time it should be left as an empty string, otherwise we could just specify the location of the already backed up parameters. In case you want to pull old parameters, just be sure to have the same architecture as before! And to begin the training session just open a terminal, navigate to the location of netfieldv1, and execute
```bash
$ python3 ./src/train.py <path for the <config> file> --<processing_unit> 
```
The `<processing_unit>` argument can either take the string `cpu` or `gpu`. Now if we've got the [CuPy] framework already installed, we should use `gpu`, otherwise we use the `cpu` option.

### Classifying a Set of Images

To classify an image or set of images, enter the path of the already trained learnable parameters backed up file on the `<config>` file, prepare a set of images as an ndarray of shape (examples, height, width) or (examples, height, width, channels) and execute the following line on a terminal,
```bash
$ python3 ./src/classify.py <path for the <config> file> <path to image set> --<processing_unit>
```

### Deep Learning Theory

A vectorized version of the backpropagation algorithm that has been implemented on this platform is derived and specified in [[BP][bp]].


[pip]: <https://www.tecmint.com/install-pip-in-linux/>
[Matplotlib]: <https://matplotlib.org/>
[NumPy]: <https://numpy.org/>
[idx2numpy]: <http://yann.lecun.com/exdb/mnist/>
[CuPy]: <https://cupy.chainer.org/>
[pandas]: <https://pandas.pydata.org/>
[Python3]: <https://www.python.org/>
[Pillow]: <https://pillow.readthedocs.io/>
[BP]: <https://drive.google.com/file/d/1O0ROuO3EQ0fxcG2K_GNI6ywh-bs1CGxp/view?usp=sharing>





























