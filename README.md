# LiDAR Data Tools
While working with the LiDAR-bonnetal framework to develop deep learning models for semantic 
segmentation of UAV LiDAR data, some code was written to aid and speed up the work process. That code has been collected 
into this repo. This repo can thus be viewed as a complement to the LiDAR-Bonnetal framework and contains scripts and 
help-functions for pre-processing, visualizing, and analyzing LiDAR data and for evaluating model predictions. 

## Installation and set-up

It is highly recommended to use a virtual environment. 

#### Install requirements

```sh
 pip3 install -r requirements.txt
```

#### Install package 

First, go to the root of the project (lidar-data-tools), then run:


```sh
 pip3 install . 
```

If you want to edit the files, instead run:

```sh
 pip3 install -e . 
```