# AdvCam-Hide-Adv-with-Natural-Styles

Code for "Adversarial Camouflage: Hiding Physical-World Attacks with Natural Styles"

## Installation
We highly recommend using [conda](https://www.anaconda.com/distribution/).
```sh
conda create -n advcam_env python=3.6
source activate advcam_env
```
After activating virtual environment:
```sh
git clone https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles
cd AdvCam-Hide-Adv-with-Natural-Styles
pip install requirement.txt
```
**Note: We use tensorflow v1 in the code, incompatible with python>3.6 when we test.**

## Usage
#### Quick Start
Running our given example:
```sh
sh run.sh
```
#### Basic Usage
* Path to images
  * Put target image, style image, and their segmentations in folders. We set the name of target/style image and their segmentation are same by default. 
  * Modify run.sh and advcam_main.py if you change the path to images.
  * We define the segmentation in following way, you can change it in utils.py

Segmentation type | RGB value
------------ | -------------
UnAttack | <(128,128,128)
Attack | >(128,128,128)

* Parameters
  *Parameters can be speicified in either run.sh or advcam_main.py.

* Run the follow scipt
```sh
sh run.sh
```
## Usage example

A few motivating examples of how AdvCam can be used. 
#### Target Image
Target image (predicted as "Street sign" with 0.904 confidence)
![Target image](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles/blob/master/results/ori_stop_sign.png)
#### Adv Images
Three adverarial images generated by AdvCam with natural adversarial perturbation.
![Adv image](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles/blob/master/results/adv_group.png)
#### Physical Test
![Physical test](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles/blob/master/results/AdvCam-gif2.gif)
More generation and test details can be found in video [AdvCam](https://www.youtube.com/watch?v=gk3NHY_gpvg)

## Acknowledgments
We use [Yang's code](https://github.com/LouieYang/deep-photo-styletransfer-tf) for style transfer part.


## Citation



