# -*- coding: utf-8 -*
import numpy as np  
import os  
import cv2
import caffe  
from _argparse import argparse
from pyseeta import Aligner
from pyseeta.common import*
from norm import*

class FaceBoxes(object):
    def __init__(self):   
        self.args = argparse()
        gpu_id = int(self.args.gpu)   
        if gpu_id<0: 
            caffe.set_mode_cpu()
        else:
            caffe.set_device(gpu_id)
        
        assert os.path.exists(self.args.deploy_det), 'file {} is not found'.format(self.args.deploy_det)
        assert os.path.isfile(self.args.weights_det), 'file {} is not found'.format(self.args.weights_det)

        self.net = caffe.Net(self.args.deploy_det, self.args.weights_det, caffe.TEST)  
        # self.Onet_p = ONet_Points(self.args.deploy_det_land, self.args.weights_det_land)
        print ('\t deploy_det:{} is used'.format(self.args.deploy_det))
        print ('\t weights_det:{} is used'.format(self.args.weights_det))
        self.aligner=Aligner(self.args.seeta_land)

    def detect(self, img):
        h, w=img.shape[:2]
        height_new, width_new = int(self.args.image_scale*h), int(self.args.image_scale*w)
        # height_new = width_new = 1024
        self.net.blobs['data'].reshape(1, 3, height_new, width_new)
        im = cv2.resize(img, (width_new, height_new))
        transformed_image = 0.00390625 * (im.astype(np.float32, copy=False) - np.array([104, 117, 123])).transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = transformed_image
        out = self.net.forward()  
        boxes = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
        
        # cls = out['detection_out'][0, 0, :, 1]
        
        conf = out['detection_out'][0, 0, :, 2]

        bboxes = np.hstack((boxes, conf[:, np.newaxis])).astype(np.float32, copy=False)
        bboxes = [bbox for bbox in bboxes if bbox[4]>0]
        if len(bboxes) == 0:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = []
        for box in bboxes:
            face = Face()
            face.left = box[0]
            face.top = box[1]
            face.right = box[2]
            face.bottom = box[3]
            face.score = box[4]
            landmark = self.aligner.align(gray, face)  # seetaface1 landmarks
            landmarks.append(landmark)

        return np.array(bboxes), np.array(landmarks)

    def maxFace(self, image, boxes, points):
        assert image is not None
        if len(boxes) == 0:
            return None
        else:   
            max_box_area = 0  
            idx_tmp = 0
            for idx, box in enumerate(boxes):     
                box_area = (box[2] - box[0])*(box[3]-box[1])
                if max_box_area < box_area:
                    max_box_area = box_area 
                    idx_tmp = idx           
            p = points[idx_tmp]
            maxface = get_norm2(image, boxes[idx_tmp], p, image_size=self.args.norm_size)
            # maxface = get_norm2(image, boxes[idx_tmp], p, image_size='118, 118')
            # maxface = get_norm(image, p, cropSize=(112, 112), ec_mc_y=48, ec_y=40)
            return maxface, idx_tmp


if __name__ == '__main__':
    fb = FaceBoxes()
    cap= cv2.VideoCapture(0)
    flage, image = cap.read()
    while True:
        flage, image = cap.read()
        if image is None :
            break       
        ret = fb.detect(image)
        if ret:
            bboxes, points = ret
            print (points)
            for idx, box in enumerate(bboxes):
                box = np.array(box)
                box[:4] = [int(b) for b in box[:4]]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (230, 255, 0), 2)
                for p in points[idx]:
                    print (p)

                    cv2.circle(image, (p[0], p[1]), 2, (230, 255, 0), -1)
        cv2.imshow('image', image)
        if 27 == cv2.waitKey(1):
            break
