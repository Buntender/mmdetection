from basetest import main
from mmdet.apis import single_gpu_test, multi_gpu_test

def wrapper(func):
    def inner(*kargs, **kwargs):
        kwargs.pop("dir_name")
        return func(*kargs, **kwargs)
    return inner

if __name__ == '__main__':
    main(wrapper(single_gpu_test), wrapper(multi_gpu_test))