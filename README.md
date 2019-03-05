# Predicting Depth Hole Locations from a Single RGB Image

Requires an NVIDIA GPU, Python 3, [CUDA CuDNN](https://developer.nvidia.com/cudnn), [PyTorch 1.0](http://pytorch.org), and [OpenCV](http://www.opencv.org).
<br>
Other libraries such as [visdom](https://github.com/facebookresearch/visdom) and [colorama](https://pypi.org/project/colorama/) are also optionally used in the code.

![General Pipeline](https://github.com/atapour/hole_prediction.dev/blob/master/imgs/architecture.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Network Architecture

## Method:

_"Depth images naturally sensed in the real world through active or passive depth sensing devices are often corrupted or contain large missing regions (holes). This is why in many learning based applications using depth images (e.g., depth completion), synthetic images are used. In such a scenario, synthetic pixel-perfect depth images can be used as ground truth but corrupted depth images (with holes) of the same scenes are required as inputs. Rather than randomly cutting out sections of the image, we opt for creating realistic holes with the characteristics of those found in real-world depth images, which occur in stereo correspondence due to the existence of featureless or shiny surfaces, unclear object separation and distant objects, among others. To produce these semantically meaningful holes, a separate model is needed to predict depth holes by means of pixel-wise classification. The objective is to produce a hole mask, which represents regions in the depth image likely to contain holes. Since within synthetic datasets, only complete pixel-perfect depth is available, simulating corrupted depth, similar to what is naturally sensed in the real world, is important."_


Things need to change from this point on:
[[Atapour-Abarghouei and Breckon, Proc. CVPR, 2018](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)]

---


![](https://github.com/atapour/hole_prediction.dev/blob/master/imgs/sampleResults.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of the results of the approach

---
## Instructions to train the model:

* First and foremost, this repository needs to be cloned:

```
$ git clone https://github.com/atapour/hole_prediction.dev.git
$ cd hole_prediction.dev
```

* A dataset needs to be prepared to be used for training. In our experiments, we use the [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset. However, any other dataset containing rectified stereo images is suitable for training this model. We have provided a simple python script (`data_processing/data_processing_kitti.py`) that processes the raw KITTI data and generates a data root in accordance with our custom dataset class (`data/aligned_dataset.py`). However, feel free to modify the dataset class to fit your own data structure. Our custom dataset follows the following directory structure:

```
Custom Dataset
├── RGB
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   ├──   ...
├── GT
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   ├──   ...

```
* The training code utilizes [visdom](https://github.com/facebookresearch/visdom) ot display training results and plots. To these, run `python -m visdom.server` and click the URL http://localhost:8097.

* To train the model, run the following command:

```
$ python train.py --name=name --data_root=path/to/data
```

* All the arguments for the training are passed from the file `train_arguments.py`

---
## Instructions to test the model:

* In order to easily test the model, we provide a set of pre-trained weights `pretrained_weights/hole_predictor.pth`. Additionally, a small number of example images are provided for testing.

* To test the model, run the following command:

```
$ python test.py --data_root=./examples --test_checkpoint_path=./pretrained_weights/hole_predictor.pth --results_path=./results
```
---

The output results are written in a directory taken as an argument to the test harness ('./results' by default):
* the file with the suffix "_rgb" is the original input image.
* the file with the suffix "_hole" is the output mask image, demonstrating where holes would be if a depth image was obtained from the corresponding input RGB image.

---

This work is created as part of the project published in the following. Note that this code is not the entire code for the project, but only the depth prediction project with an improved architecture.

## Reference:

[Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)
(A. Atapour-Abarghouei, T.P. Breckon), In Proc. Conf. Computer Vision and Pattern Recognition, 2018. [[pdf](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)] [[demo](https://vimeo.com/260393753)]

```
@InProceedings{abarghouei18monocular,
  author = 		{Atapour-Abarghouei, A. and Breckon, T.P.},
  title = 		{Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation},
  booktitle = 	{Proc. Computer Vision and Pattern Recognition},
  pages =		{1-8},
  year = 		{2018},
  month = 		{June},
  publisher = 	{IEEE},
  keywords = 		{monocular depth, generative adversarial network, GAN, depth map, disparity, depth from single image},
}

```
---
