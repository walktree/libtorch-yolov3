# libtorch-yolov3
A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. It's fast, easy to be integrated to your production, and CPU and GPU are both supported. Enjoy ~

This project is inspired by the [pytorch version](https://github.com/ayooshkathuria/pytorch-yolo-v3), I rewritten it with C++.

## Requirements
1. LibTorch v1.0.0
2. Cuda
3. OpenCV (just used in the example)


## To compile
1. cmake3
2. gcc 5.4 +



```
mkdir build && cd build
cmake3 -DCMAKE_PREFIX_PATH="your libtorch path" ..

# if there are multi versions of gcc, then tell cmake which one your want to use, e.g.:
cmake3 -DCMAKE_PREFIX_PATH="your libtorch path" -DCMAKE_C_COMPILER=/usr/local/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ ..
```


## Running the detector

The first thing you need to do is to get the weights file for v3:

```
cd models
wget https://pjreddie.com/media/files/yolov3.weights 
```

On Single image:
```
./yolo-app ../imgs/person.jpg
```

As I tested, it will take 25 ms on GPU ( 1080 ti ). please run inference job more than once, and calculate the average cost.
