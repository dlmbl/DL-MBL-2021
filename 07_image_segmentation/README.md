# Exercise 7: Image Segmentation

## Setup

Create a conda environment for this exercise and activate it:
```
conda create -n exercise7 -c conda-forge pytorch-cpu jupyter imageio scipy tensorboard torchvision-cpu matplotlib
```

The above environment installs a cpu pytorch version, so you can train your model only on the cpu. This is possible, but requires some patience.
You can install a gpu version via
```
conda create -n exercise7 -c pytorch -c conda-forge pytorch jupyter imageio scipy tensorboard torchvision matplotlib cudatoolkit=<XY.Z>
```
where you need to replace `<XY.Z>` with the correct CUDA version for your system, see https://pytorch.org/ for details.

Activate it:
```
conda activate exercise7
```

You can now run the exercise in jupyter.
