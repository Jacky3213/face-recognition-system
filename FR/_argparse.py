from argparse import ArgumentParser

def argparse():
        ## face detect and landmark
        parser = ArgumentParser('faceboxes + seetaface landmarks demo')

        parser.add_argument('--gpu', default=-1, type=str, help='gpu id, -1 for cpu')
        parser.add_argument('--image_scale', dest='image_scale', default=0.8, type=float)
        parser.add_argument('--deploy_det', dest='deploy_det', help='detection prototxt',
                            default='models/ff.prototxt', type = str)
        parser.add_argument('--weights_det', dest='weights_det', help='weights for detection',
                            default='models/ff.caffemodel', type = str)  
        ## seetaface landmarks
        parser.add_argument('--seeta_land', dest='seeta_land', help='seeta model path',
                            default='models/seeta_fa_v1.1.bin', type = str)
        ## face recognition 
        parser.add_argument('--layer', default='fc5/sphere', type=str)
        parser.add_argument('--norm_size', default='112,96', type=str, help='normalized size of a face, height and width')
        parser.add_argument('--deploy_recog', dest='deploy_recog', help='face recogniton model',                      
                            default = './models/xx.prototxt', type = str)
        parser.add_argument('--weights_recog', dest='weights_recog', help='face recogniton model',                         
                        default = './models/xx.caffemodel', type = str)

        return parser.parse_args()


