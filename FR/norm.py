import cv2
import numpy as np
import math
from skimage import transform as trans



def get_norm(image, plandmarks, cropSize=(128, 128), ec_mc_y=48, ec_y=40):
    '''
    image: input image
    plandmarks: ((x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4))
    cropSize:  crop face size, default:(128, 128)
    ec_mc_y: distance between two eyes' center and mouth corner center, default:48
    ec_y: distance between two eyes, default:40
    '''

    midEye = ((plandmarks[0][0] + plandmarks[1][0])/2.0, (plandmarks[0][1] + plandmarks[1][1])/2.0)
    midMouth = ((plandmarks[3][0] + plandmarks[4][0])/2.0, (plandmarks[3][1] + plandmarks[4][1])/2.0)
    distance_ec_mc = ((midEye[0] - midMouth[0])**2 + (midEye[1] - midMouth[1])**2)**0.5
    scale = ec_mc_y / distance_ec_mc
    angle = math.atan2((plandmarks[1][1] - plandmarks[0][1]), (plandmarks[1][0] - plandmarks[0][0]))

    M = np.zeros((2, 3))
    M[0][0] = scale*math.cos(angle)
    M[0][1] = scale*math.sin(angle)
    M[1][0] = -scale*math.sin(angle)
    M[1][1] = scale*math.cos(angle)
    M[0][2] = -(midEye[0]*M[0][0] + midEye[1]*M[0][1]) + cropSize[0] / 2
    M[1][2] = -(midEye[0]*M[1][0] + midEye[1]*M[1][1]) + ec_y
    norm = cv2.warpAffine(image, M, cropSize)
    return norm

def get_norm2(img, bbox=None, landmark=None, **kwargs):
    if isinstance(img, str):
        img = read_image(img, **kwargs)
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        # assert len(image_size)==2
        # assert image_size[0]==112
        # assert image_size[0]==112 or image_size[1]==96
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
    else: #do align using landmark
        assert len(image_size)==2

        #src = src[0:3,:]
        #dst = dst[0:3,:]


        #print(src.shape, dst.shape)
        #print(src)
        #print(dst)
        #print(M)
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

        #tform3 = trans.ProjectiveTransform()
        #tform3.estimate(src, dst)
        #warped = trans.warp(img, tform3, output_shape=_shape)
        return warped