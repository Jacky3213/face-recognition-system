import caffe
import cv2
import numpy as np
from _argparse import argparse


class Spereface(object):
    def __init__(self):
        self.args = argparse()
        gpu_id = int(self.args.gpu)   
        if gpu_id<0: 
            caffe.set_mode_cpu()
        else:
            caffe.set_device(gpu_id)

        self.net = caffe.Net(self.args.deploy_recog, self.args.weights_recog, caffe.TEST)
        print ('\t deploy_recog:{} is used'.format(self.args.deploy_recog))
        print ('\t weights_recog:{} is used'.format(self.args.weights_recog))

    def extract_feature(self, faceImg):
        assert faceImg is not None
        input_shape = self.net.blobs['data'].data.shape

        transformed_image = 0.0078125*(faceImg.astype(np.float32, copy=False) - 127.5).transpose((2, 0, 1))
        # transformed_image = 0.00390625*(faceImg.astype(np.float32, copy=False) - 128).transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = transformed_image
        out = self.net.forward()
        f = out[self.args.layer][0]
        f = (f / np.linalg.norm(f)).flatten()      
        return f

    def extract_feature_flip(self, faceImg):
        faceImg_flip = cv2.flip(faceImg, 1)
        f0 = self.extract_feature(faceImg)
        f1 = self.extract_feature(faceImg_flip)
        f = np.concatenate((f0, f1))
        f = (f / np.linalg.norm(f)).flatten()      
        return f

def feature_sim(f1, f2):
    return np.dot(f1, f2.T)
    
if __name__ == '__main__':
    sp = Spereface('models/spherefacem2-sgd_deploy.prototxt', 'models/sphere-m2-sgd_train_iter_260000.caffemodel')

    im1 = cv2.imread('zhuxiaoming_0.bmp')
    im2= cv2.imread('zhuxiaoming_1.bmp')

    t1 = cv2.getTickCount()
    f1 = sp.extract_feature(im1)
    t2 = cv2.getTickCount()
    t = 1000 * (t2 - t1) / cv2.getTickFrequency()
    f2 = sp.extract_feature(im2)
    sim = feature_sim(f1, f2)
    print (f1, f2)
    print (type(f1))
    print (sim)
    print (t, 'ms')