# -*- coding: utf-8 -*
import numpy as np  
import os  
import cv2
from face import*
from sphereface import*



def model_init():
    fb = FaceBoxes()
    recog = Spereface() 
    return fb, recog

def load_features(rootPath):
    print ('load registed features...')
    folders = os.listdir(rootPath)
    features_names = {}
    for folder in folders:
        each_bin_path = os.path.join(rootPath, folder)
        bin_names = os.listdir(each_bin_path)   
        bin_names = [tmp for tmp in bin_names if 'bin' in tmp[-4:]]
        # name = bin_names[0][:-6]
        name = bin_names[0].split('_')[0]
        for bin in bin_names:
            bin_path = os.path.join(each_bin_path, bin)
            print (bin_path)
            face_feature = np.fromfile(bin_path, dtype=np.float32)
            features_names[name] = face_feature
    return features_names

'''
每个人注册一张近红外人脸
'''
def fea_compare(feature, features_names):
    sim_names = {}
    for key, value in features_names.items():       
        score = np.dot(feature, value.T)      
        sim_names[key] = score
    sim_name = sorted(sim_names.items(), key=lambda x:x[1], reverse=True)  # warning, now sim_name is a list   
    return sim_name[0]