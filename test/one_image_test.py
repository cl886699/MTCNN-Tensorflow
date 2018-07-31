#coding:utf-8
import sys
import time
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
test_mode = "ONet"
thresh = [0.8, 0.6, 0.7]
min_face_size = 60
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_yin_model/PNet/PNet', '../data/MTCNN_yin_model/RNet/RNet', '../data/MTCNN_yin_model/ONet/ONet']
epoch = [30, 22, 22]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
'''
gt_imdb = []
#gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
#imdb_ = dict()"
#imdb_['image'] = im_path
#imdb_['label'] = 5
path = "lala"

for item in os.listdir(path):
    gt_imdb.append(os.path.join(path,item))



test_data = TestLoader(gt_imdb)
all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
'''
image_path = 'lala/single_face.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480))
t1 = time.time()
boxes_c,landmarks = mtcnn_detector.detect(img)
t = time.time() - t1
print('total time',  t)
for i in range(boxes_c.shape[0]):
    bbox = boxes_c[i, :4]
    score = boxes_c[i, 4]
    corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    # if score > thresh:
    cv2.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
    cv2.putText(img, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('show', img)
    # exit press any key
    cv2.waitKey(0)  


'''
count = 0
for imagepath in gt_imdb:
    #print(imagepath)
    image = cv2.imread(imagepath)
    image = cv2.resize(im, (640, 480))
    for bbox in all_boxes[count]:
        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
   
    for landmark in landmarks[count]:
        for i in range(len(landmark)/2):
            cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
   
        
    count = count + 1
    #cv2.imwrite("result_landmark/%d.png" %(count),image)
    cv2.imshow("lala", image)
    cv2.waitKey(0)    


for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''
