from basetest import main
from robustdetector.apis import clsloss_single_gpu_test, clsloss_multi_gpu_test

if __name__ == '__main__':
    main(clsloss_single_gpu_test, clsloss_multi_gpu_test)