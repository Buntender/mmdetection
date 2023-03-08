import os
openpath = "data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
savepath = "robustdetector/utils"

start = 0
end = start + 200
ptr = 0

with open(openpath) as rawfile:
    with open(f"{savepath}/test_from{start}_to{end}.txt", "w")as resfile:
        line = rawfile.readline()
        while line:
            ptr += 1
            if ptr > start:
                resfile.write(line)
            if ptr >= end:
                break
            line = rawfile.readline()