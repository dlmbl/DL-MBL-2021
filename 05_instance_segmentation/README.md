# Exercise 5: Instance Segmentation

## Setup

Create a `conda` environment for this exercise and activate it:

```
conda create -n 05_instance_segmentation python
conda activate 05_instance_segmentation
```

After entering this virtual environment, install pytorch, ipykernel and ipywidgets by using the following command

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install ipykernel

conda install -c conda-forge ipywidgets

jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

Start Jupyter within this environment:

```
jupyter notebook
```

...and continue with the instructions in the notebook.

## Setup for the Google Colab
---
### Google Colab brief intro

**Requirements** To use Colab, you must have a Google account with an associated Google Drive.

**Reminder** Ressources on colab are not guaranteed and therefore there might be times where some ressources cannot get allocated. If you're idle for 90 minutes or your connection time exceeds the maximum of 12 hours, the colab virtual machine will disconnect. This means that unsaved progress such as model parameters are lost. 

### 2.1 In the Google Driver

Open the Google Chrome browser and open your google driver when you log into your google account.

Create an empty folder with name you wish (e.g, image-anaylsis-tutorial) and upload all .ipynb files and other necessary files into that folder.

### 2.2 In the notebook

Use GPU:  just follow **Edit > Notebook settings** or **Runtime>Change runtime type** and select GPU as Hardware accelerator.

Start with the following code in your notebook which will mount the driver and change the current working directory to the given path. Remember to set the **path** variable according to the folder name you created in **step 2.1**.

```
from google.colab import drive
drive.mount('/content/drive')
import os
path = "/content/drive/My Drive/image-analysis-tutorial" # path="/content/drive/My Drive/<folder name>"
os.chdir(path)
```

Run the above cell and grant permissions according to the instructions shown.


Then install the following packages in the notebook, remember to check that the imagug version is 0.4.0.

```
!pip uninstall albumentations -y
!pip install tqdm h5py zarr pillow numpy imgaug==0.4.0 mahotas #imgaug has dependency on previous packages
!pip install scikit-image
!pip install tensorboard
!pip install torchsummary
```

## 3. Introduction to the exercise
---
Only the **epithelia_segmentation_challenge.ipynb** needs some extra code implementation.

- 1.foreground_segmentation.ipynb

    data: example_toy_data

- 2.instance_segmentation.ipynb

    data: data_kaggle_test

- 3.epithelia_segmentation_challenge.ipynb (**need extra code implementation**)

    data: data_epithelia

    solution: example_epithelia_segmentation.ipynb

- 4.tile_and_stitch.ipynb

    data: data_kaggle_test

    pretrained_net: net_60000

- 5.Image_Analysis_intro.ipynb

    data: CIFAR-10

Please look into the respective .ipynb file to see the details. For some exercise, it would be better to run on GPU. 


