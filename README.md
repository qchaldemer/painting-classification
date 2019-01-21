## 1. Project Motivation
- I am very interested in painting and I have always been wondering whether art experts were using only the painting to determine the style of the art piece or additional information such as the date, the name of the painter and so on.
- To understand it, I have trained a neural network using only the paintings to classify paitings among the following categories: 
    - Art Nouveau
    - Baroque
    - Expressionism
    - Impressionism
    - Post-Impressionism
    - Rococo
    - Romanticism
    - Surrealism
    - Symbolism

## 2. Overview of the results
- Performance on test set is 49%. Validation and training are Train 67.1% and 51%
- Optimal error rate is 30% because some categories are very similar such as impressionism and post impressionism, without additional information than the pixels like painter name and date a person knowledgeable about art would reach 70% accuracy on this dataset (calculated on 50 examples from the validation set)
- Final neural network model used transfer learning based on a vgg19 with a fully connected layer containing 12595 hidden units trained on 1600 examples implemented with Pytorch
- I used many techniques to reduce avoidable biais and variance
    - avoidabale biais: increase the model size (additional layer in the classifier, additional neurons in the layer, tested other architecture with resnet50), tested different regularization parameters
    - variance: added more training data and early stopping

## 3. File Descriptions
- painting_classifier.ipynb: jupyter notebook containing the training, prediction, error analysis of the DL model

- checkpoint5.pth: the saved version of the model (download it here: https://drive.google.com/open?id=1C75piZ_YxpFzOQyhif_ERpaWov5PC4gd and save it in the same folder as the app/)

- app/: contains the python script for an application in the terminal that predicts the painting style of a painting

## 4. How to Interact with your project

- Upload an image of a painting in app/example for e.g. 100152.jpg
- Basic usage: run python predict.py --image example/100152.jpg
- Options:
    - Return top 5 most likely classes: python predict.py --image example/100152.jpg --top_k 5 

## 5. Installations

absl-py               0.4.1
annoy                 1.12.0
args                  0.1.0
astor                 0.7.1
atari-py              0.1.1
backcall              0.1.0
bleach                2.1.4
Box2D-kengz           2.3.3
cachetools            2.1.0
certifi               2018.8.13
chardet               3.0.4
click                 6.7
clint                 0.5.1
cloudpickle           0.5.3
cupy                  4.3.0
cupy-cuda91           4.3.0
cycler                0.10.0
cymem                 1.31.2
Cython                0.28.5
cytoolz               0.9.0.1
dask                  0.18.2
decorator             4.3.0
dill                  0.2.8.2
dlib                  19.15.0
entrypoints           0.2.3
fastrlock             0.3
Flask                 1.0.2
floyd-cli             0.11.17
ftfy                  4.4.3
future                0.16.0
gast                  0.2.0
grpcio                1.14.2
gym                   0.10.5
gym-retro             0.5.6
h5py                  2.8.0
html5lib              1.0.1
idna                  2.7
ijson                 2.3
incremental           17.5.0
intel-openmp          2019.0
ipykernel             4.8.2
ipython               6.5.0
ipython-genutils      0.2.0
ipywidgets            7.4.0
itsdangerous          0.24
jedi                  0.12.1
Jinja2                2.10
jsonschema            2.6.0
jupyter               1.0.0
jupyter-client        5.2.3
jupyter-console       5.2.0
jupyter-core          4.4.0
jupyterlab            0.33.8
jupyterlab-launcher   0.11.2
kaggle                1.4.6
Keras                 2.2.0
Keras-Applications    1.0.2
Keras-Preprocessing   1.0.1
kiwisolver            1.0.1
Markdown              2.6.11
MarkupSafe            1.0
marshmallow           2.15.4
matplotlib            2.2.3
menpo                 0.8.1
mistune               0.8.3
mkl                   2019.0
mkl-devel             2018.0.3
mkl-include           2018.0.3
mpmath                1.0.0
msgpack               0.5.6
msgpack-numpy         0.4.3.1
murmurhash            0.28.0
nbconvert             5.3.1
nbformat              4.4.0
networkx              2.1
nltk                  3.3
notebook              5.6.0
numpy                 1.15.1
opencv-contrib-python 3.4.0.12
pandas                0.23.4
pandocfilters         1.4.2
parso                 0.3.1
path.py               11.0.1
pathlib2              2.3.2
pexpect               4.6.0
pickleshare           0.7.4
Pillow                5.2.0
pip                   10.0.1
plac                  0.9.6
plotly                3.1.1
preshed               1.0.1
prometheus-client     0.3.1
prompt-toolkit        1.0.15
protobuf              3.6.1
ptyprocess            0.6.0
pydot                 1.2.4
pyemd                 0.5.1
pyglet                1.3.2
Pygments              2.2.0
pynvrtc               9.2
PyOpenGL              3.1.0
PyOpenGL-accelerate   3.1.0
pyparsing             2.2.0
Pyphen                0.9.4
python-dateutil       2.7.3
python-Levenshtein    0.12.0
pytz                  2018.5
PyWavelets            0.5.2
PyYAML                3.13
pyzmq                 17.1.2
qtconsole             4.4.1
raven                 6.9.0
regex                 2017.4.5
requests              2.19.1
requests-toolbelt     0.8.0
retrowrapper          0.2.1
retrying              1.3.3
scikit-image          0.14.0
scikit-learn          0.19.2
scikit-umfpack        0.3.1
scipy                 1.1.0
seaborn               0.9.0
Send2Trash            1.5.0
setuptools            39.1.0
simplegeneric         0.8.1
six                   1.11.0
sklearn               0.0
spacy                 2.0.12
sympy                 1.2
tabulate              0.8.2
tensorboard           1.9.0
tensorboardX          1.2
tensorflow            1.9.0
termcolor             1.1.0
terminado             0.8.1
testpath              0.3.1
textacy               0.6.2
tflearn               0.3.2
thinc                 6.10.3
toolz                 0.9.0
torch                 0.5.0a0+a24163a
torchtext             0.3.1
torchvision           0.2.1
tornado               5.1
tqdm                  4.25.0
traitlets             4.3.2
typing                3.6.6
ujson                 1.35
Unidecode             1.0.22
urllib3               1.22
uWSGI                 2.0.17
virtualenv            16.0.0
wcwidth               0.1.7
webencodings          0.5.1
Werkzeug              0.14.1
wheel                 0.31.1
widgetsnbextension    3.4.0
wrapt                 1.10.11
xgboost               0.80
zmq                   0.0.0

## 6. Licensing, Authors, Acknowledgements, etc.
- Dataset is from the kaggle competition painter by numbers and corresponds to the train_1.zip (https://www.kaggle.com/c/painter-by-numbers/data)
- Capstone project of Udacity DS Nanodegree program