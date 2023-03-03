import os
import multiprocessing
import time

def _main(dir_name, epoch, model2):
    from basetest_bimodel import main
    from robustdetector.apis import bboxloss_single_gpu_getfeature_dual, clsloss_single_gpu_getfeature_dual, robust_multi_gpu_getfeature, bboxloss_multi_gpu_getfeature, \
        clsloss_multi_gpu_getfeature
    # main(clsloss_single_gpu_getfeature_dual, clsloss_multi_gpu_getfeature, dir_name, epoch, model2)
    main(bboxloss_single_gpu_getfeature_dual, bboxloss_multi_gpu_getfeature, dir_name, epoch, model2)

if __name__ == '__main__':
    for epoch in range(1, 13):
        for round in range(0, 1):
            dir_name = f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_gentest_bimodel_FreeRobust/epoch_{epoch}/test_{round}/'
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            subprocess = multiprocessing.Process(target=_main, args=(dir_name, 24, f'/media/data4/lkz/mmdetection_stable_on_28_1124/work_dirs/ssd300_voc_FreeRobust/epoch_{epoch}.pth'))
            subprocess.start()
            subprocess.join()

    # for round in range(0, 1):
    #     dir_name = f'ext_feature/voc_clean_Freerobust/clean_robust/cls_old/{round}/'
    #     if not os.path.isdir(dir_name):
    #         os.makedirs(dir_name)
    #     # subprocess = multiprocessing.Process(target=_main, args=(dir_name, 24, '/media/data4/lkz/mmdetection_stable_on_28_1124/work_dirs/ssd300_voc_FreeRobust_train0/latest.pth'))
    #     subprocess = multiprocessing.Process(target=_main, args=(dir_name, 24, '/media/data4/lkz/mmdetection_stable_on_28_1124/work_dirs/ssd300_voc/latest.pth'))
    #     subprocess.start()
    #     subprocess.join()
