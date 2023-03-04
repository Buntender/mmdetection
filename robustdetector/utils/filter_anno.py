with open("data/VOCdevkit/VOC2012/ImageSets/Main/person_trainval.txt") as rawfile:
# with open("data/VOCdevkit/VOC2007/ImageSets/Main/person_test.txt") as rawfile:
    with open("robustdetector/utils/person_only_anno_2012_trainval.txt", "w")as resfile:
        line = rawfile.readline()
        while line:
            line = line.split(" ")
            if line[1].find("-1") == -1:
                resfile.write(line[0]+'\n')
            line = rawfile.readline()

