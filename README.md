# mobileye-project
## Detecting traffic lights and the distance to them on runtime within given video 

### phase 1:
Detection of source lights in an image using convolution with customized high- and low-pass filters.

### phase 2:
Generating and training CNN using the products of the previous stage as input, to conclude all the traffic lights in the image (using tensorflow). 

### phase 3:
Estimating the distance to each detected traffic light from the camera picturing the images of interest, involving geometric and linear algebra calculations.

### phase 4:
Integrating all previous parts into a functional and intuitive SW product. 

### Libraries/Technologies Used:
* python 3.7
* numpy
* matplotlib
* scipy
* imgaug
* tensorflow
