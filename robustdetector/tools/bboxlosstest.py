from basetest import main
from robustdetector.apis import bboxloss_single_gpu_test, bboxloss_multi_gpu_test

if __name__ == '__main__':
    main(bboxloss_single_gpu_test, bboxloss_multi_gpu_test)