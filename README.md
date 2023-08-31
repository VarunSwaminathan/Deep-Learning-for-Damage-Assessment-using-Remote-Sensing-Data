# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Building Damage Assessment Model

**Jump to: [Model](#model) | [Data Sources](#data-sources) | [Setup](#setup) | [Instructions to predict for any test image](#instructions-to-predict-for-any-test-image) | [Results](#results) |**


The objective of the "Deep Learning for Damage Assessment using Remote Sensing Data" project is to create a framework that can rapidly and precisely identify and evaluate the extent of damage caused by natural disasters or human-made incidents by analyzing high-resolution satellite imagery with the assistance of deep learning techniques. The rationale behind this project is the urgency for prompt and accurate damage assessment following disasters. The conventional method of damage assessment is manual analysis of satellite images by experts, which is both time-consuming and prone to errors. The adoption of deep learning techniques can enhance the speed and precision of damage assessment, assist in disaster response and recovery efforts, and ultimately reduce fatalities and minimize the economic consequences of disasters.

## Data Sources
The following datasets were used for training the model.
1. SpaceNet dataset 2. xBD dataset 3. CrowdAI dataset: 

## Instructions to predict for any test image

Navigate to folder "\run" run then execute:
```
python pred_py.py --before_PTH "path/to/image/before" --after_PTH "path/to/image/after" --result_PTH "results.png"
```
or for a default example just run
```
python pred_py.py
```

This generates an output image and results will be displayed and a percentage of estimated damage is printed in the console


## Results
We show the result of prediction below:
![Segmented and damage assessment percentage value](./run/results.PNG)
