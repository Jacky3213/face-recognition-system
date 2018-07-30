import numpy as np  
import os  
import cv2
from process import*
'''
In the folder of each person, 
jpg: image with person
bmp: normalized face image
bin: face image feature file
'''
def register_all(rootPath):
    fb, recog = model_init()
    folders = os.listdir(rootPath)
    print (folders)
    for folder in folders:
        each_im_path = os.path.join(rootPath, folder)
        im_names = os.listdir(each_im_path)
        for im_name in im_names:
            im_path = os.path.join(each_im_path, im_name)
            if 'jpg' not in im_path:
                continue
            print (im_path)
            image = cv2.imread(im_path)
            ret = fb.detect(image)
            if ret is None:
                continue
            boxes, points = ret    
            max_face, _ = fb.maxFace(image, boxes, points)
            if max_face is not None:          
                face_path = im_path.replace('jpg', 'bmp')  # save cropped face
                cv2.imwrite(face_path, max_face)
                face_feature = recog.extract_feature(max_face)
                print ('shape', face_feature.shape)  ##
                feature_path = im_path.replace('jpg', 'bin')   #save face feature
                face_feature.tofile(feature_path)    ##warning
                print ('register complete!')
    print ('ok')
    return 0

if __name__ == '__main__':
    # image('./images/')
    register_all('images_NIR')


     