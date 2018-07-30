from register import*
from live import*


if __name__ == '__main__':
    root_path = '../data/NIR'  ##  images_NIR    20180625faceImages  BGR
    register_all(root_path)
    test_live(root_path)