## Description
This work is used for reproduce MTCNN,a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).**WIDER Face** for face detection and **Celeba** for landmark detection(This is required by original paper.But I found some labels were wrong in Celeba. So I use [this dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) for landmark detection).

## Dependencies
* Tensorflow 1.2.1
* TF-Slim
* Python 3.5
* Ubuntu 16.04
* Cuda 8.0

## Prepare For Training Data
### prepare Pnet data(no landmark data)
1. Download Wider Face Training part only from Official Website , unzip to replace `WIDER_train` and put it into `prepare_data` folder.
2. Run `prepare_data/gen_12net_data.py` to generate training data(Face Detection Part) for **PNet**.
3. Run `gen_imglist_pnet.py` to merge positive, negative and part data.
4. Run `gen_PNet_tfrecords.py` to generate tfrecord for **PNet**.
### prepare Rnet data(no landmark data)
1. After training **PNet**, run `gen_hard_example_R.py` to generate training data(Face Detection Part) for **RNet**.
2. Run `gen_RNet_pos_tfrecords.py` to generate pos tfrecords for **RNet**.
3. Run `gen_RNet_part_tfrecords.py` to generate part tfrecords for **RNet**.
4. Run `gen_RNet_neg_tfrecords.py` to generate neg tfrecords for **RNet**.

5. **total 3 tfrecords for RNet training**

### prepare ONet data(no landmark version)
1. After training **RNet**, run `gen_hard_example_O.py` to generate training data(Face Detection Part) for **ONet**.
2. Run `gen_ONet_pos_tfrecords.py` to generate pos tfrecords for **ONet**.
3. Run `gen_ONet_part_tfrecords.py` to generate part tfrecords for **ONet**.
4. Run `gen_ONet_neg_tfrecords.py` to generate neg tfrecords for **ONet**.

5. **total 3 tfrecords for ONet training**


## training
1. Run `train_models/train_PNet.py` to train PNet.
2. Run `train_models/train_RNet.py` to train RNet.
3. Run `train_models/train_ONet.py` to train ONet.

## Some Detail
* Two version of model was trained, first version has no landmark.
* When training **PNet**,I merge four parts of data(pos,part,neg) into one tfrecord,since their total number radio is almost 1:1:3.But when training **RNet** , I generate 3 tfrecords,since their total number is not balanced.During training,I read 16 samples from pos and part tfrecord, and read 32 samples from neg tfrecord to construct mini-batch. When training **ONet**,I generate four tfrecords,since their total number is not balanced.During training,I read 16 samples from pos,part and landmark tfrecord and read 32 samples from neg tfrecord to construct mini-batch.
* It's important for **PNet** and **RNet** to keep high recall radio.When using well-trained **PNet** to generate training data for **RNet**,I can get 14w+ pos samples.When using well-trained **RNet** to generate training data for **ONet**,I can get 19w+ pos samples.
* Since **MTCNN** is a Multi-task Network,we should pay attention to the format of training data.The format is:
 
  [path to image][cls_label][bbox_label][landmark_label]
  
  For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

  For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].
  
  For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
  For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].  

* Since the training data for landmark is less.I use transform,random rotate and random flip to conduct data augment(the result of landmark detection is not that good).

## Result

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

**Result on FDDB**
![result10.png](https://i.loli.net/2017/08/30/59a6b875f1792.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)
