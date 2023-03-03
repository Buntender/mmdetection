import os
import multiprocessing
import time

def _main(dir_name, epoch):
    from basetest import main
    from robustdetector.apis import bboxloss_single_gpu_getfeature, clsloss_single_gpu_getfeature, robust_multi_gpu_getfeature, bboxloss_multi_gpu_getfeature, \
        clsloss_multi_gpu_getfeature
    main(clsloss_single_gpu_getfeature, clsloss_multi_gpu_getfeature, dir_name, epoch)
    # main(bboxloss_single_gpu_getfeature, bboxloss_multi_gpu_getfeature, dir_name, epoch)

if __name__ == '__main__':
    # for epoch in range(21, 24):
    #     for round in range(3):
    #         dir_name = f'ext_feature/test2/loc/epoch_{epoch}/test_{round}/'
    #         if not os.path.isdir(dir_name):
    #             os.makedirs(dir_name)
    #         subprocess = multiprocessing.Process(target=_main, args=(dir_name, epoch))
    #         subprocess.start()
    #         subprocess.join()

    for round in range(0, 2):
        dir_name = f'ext_feature/voc_test1/cls_old/{round}/'
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        subprocess = multiprocessing.Process(target=_main, args=(dir_name, 24))
        subprocess.start()
        subprocess.join()
