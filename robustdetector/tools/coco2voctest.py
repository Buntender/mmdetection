from basetest import main
from robustdetector.apis.coco2voctest import single_gpu_test, multi_gpu_test

if __name__ == '__main__':
    main(single_gpu_test, multi_gpu_test)