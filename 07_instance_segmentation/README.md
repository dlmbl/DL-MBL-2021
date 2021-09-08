## Setup
---
## In the terminal

Create a `conda` environment for this exercise and activate it:

```
conda create -n 07_instance_segmentation python
conda activate 07_instance_segmentation
```
If conda activate <env_name> doesn't work, use

```
source activate 07_instance_segmentation
```
After entering this virtual environment, install pytorch, ipykernel and ipywidgets by using the following command

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install ipykernel

conda install -c conda-forge ipywidgets

jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

and continue with the instructions in the notebook.

## In the notebook

Start with the following code:
```
%conda install -c conda-forge matplotlib 
%conda install -c anaconda scipy
!pip install tqdm h5py zarr pillow numpy imgaug==0.4.0 mahotas #imgaug has dependency on previous packages
!pip install scikit-image
!pip install tensorboard
!pip install torchsummary
```

## Introduction to the exercise
---

- 1.foreground_segmentation.ipynb

    data: example_toy_data

- 2.instance_segmentation.ipynb

    data: data_kaggle_test

- 3.epithelia_segmentation_challenge.ipynb

    data: data_epithelia

    solution: example_epithelia_segmentation.ipynb

- 4.tile_and_stitch.ipynb

    data: data_kaggle_test

    pretrained_net: net_60000

Please look into the respective .ipynb file to see the details. For some exercise, it would be better to run on GPU. 
Only the epithelia_segmentation_challenge.ipynb need some extra code implementation