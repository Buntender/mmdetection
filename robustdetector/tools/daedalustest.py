from basetest import main
from robustdetector.apis import daedalus_single_gpu_test_save, daedalus_single_gpu_test_load

#TODO make clean save load
if __name__ == '__main__':
    # main(daedalus_single_gpu_test_save, None)
    main(daedalus_single_gpu_test_load, None)