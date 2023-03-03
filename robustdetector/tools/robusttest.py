from basetest import main
from robustdetector.apis import robust_single_gpu_test, robust_multi_gpu_test

if __name__ == '__main__':
    main(robust_single_gpu_test, robust_multi_gpu_test)