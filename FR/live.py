# -*- coding: utf-8 -*
import numpy as np  
import os  
import cv2
from process import*

def test_live(root_path):
    fb, recog_model = model_init()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    features_names = load_features(root_path)
    print ('processing...')

    while True:
        ret, frame = cap.read()
        if not ret:
            print ('No frame captured!')
            break
  
        t0 = cv2.getTickCount()     
        det_p = fb.detect(frame)
        t1 = cv2.getTickCount()
        det_land_time = 1000*(t1 - t0)/cv2.getTickFrequency()
        if det_p is not None:
            boxes, points = det_p
            maxface, max_id = fb.maxFace(frame, boxes, points)
            '''
            Do something on the BGR image
            '''
            # ## 
            # maxface_gray = cv2.cvtColor(maxface, cv2.COLOR_BGR2GRAY)
            # height, width = maxface.shape[:2]
            # for i in range(height):
            #     for j in range(width):
            #         maxface[i, j][0] = maxface[i, j][0]//2
            #         maxface[i, j][1] = maxface[i, j][1]//2
            #         maxface[i, j][2] = maxface[i, j][2]//2
            ##

            t0_feat = cv2.getTickCount()
            maxface_feature = recog_model.extract_feature(maxface) # 
            t1_feat = cv2.getTickCount()
            feat_time = 1000*(t1_feat - t0_feat)/cv2.getTickFrequency()


            sim_name_maxface = fea_compare(maxface_feature, features_names)

            for idx, box in enumerate(boxes):     
                conf = int(box[4]*100)/100
                box = [int(i) for i in box[:4]]
                color = (230, 255, 0)
                if idx == max_id:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (box[0], box[1]), (box[2],box[3]), color, 2) # Draw rectangle on  all the detected faces
                
                point = points[idx]
                for p in point:
                    cv2.circle(frame, (p[0], p[1]), 4, color, -1)  
                
                
            t = cv2.getTickCount()
            fps =int(1.0 / ((t - t0) / cv2.getTickFrequency()))

            # show something on the image
            cv2.putText(frame, 'fps:'+str(fps), (20,20), 1, 1.2, (220, 220, 0), 2)
            cv2.putText(frame, 'det_time:'+str(int(det_land_time))+ ' ms', (20, 40), 1, 1.2, (220, 220, 0), 2)     
            cv2.putText(frame, 'etr_time:'+str(int(feat_time))+ ' ms', (20, 60), 1, 1.2, (220, 220, 0), 2)
            # cv2.putText(frame, 'score:'+ '%.2f'%(100*sim_name_maxface[1]), (20, 100), 1, 1.2, (230, 220, 0), 2)

            '''
            merge maxface with the register face
            '''
            register_name = sim_name_maxface[0]
            bmp_path_root = os.path.join(root_path, register_name)
            bmp_files = os.listdir(bmp_path_root)
            bmp_files = [bmp for bmp in bmp_files if 'bmp' in bmp]
            register_img = cv2.imread(os.path.join(bmp_path_root, bmp_files[0]))

            ###
            if sim_name_maxface[1]>0.1:
                box_max = boxes[max_id]
                box_max = [int(i) for i in box_max[:4]]
                w = box_max[2] - box_max[0]
                cv2.rectangle(frame, (box_max[0], box_max[3]), (box_max[0] + w, box_max[3] + 40), (0, 0, 255), -1)
                cv2.putText(frame, sim_name_maxface[0], (box_max[0], box_max[3] + 15), 1, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, '%.2f'%(100*sim_name_maxface[1]), (box_max[0], box_max[3] + 35), 1, 1.5, (255, 255, 255), 2)

            cv2.imshow('merge', np.concatenate((register_img, maxface), axis=1))
        cv2.imshow('result', frame)
        if cv2.waitKey(1) == 27:
            break
    return 0

     