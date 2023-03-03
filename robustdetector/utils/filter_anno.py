# with open("/media/data4/lkz/mmdetection_stable_on_28_allattacks/data/VOCdevkit/VOC2007/ImageSets/Main/person_trainval.txt") as rawfile:
with open("/media/data4/lkz/mmdetection_stable_on_28_allattacks/data/VOCdevkit/VOC2007/ImageSets/Main/person_test.txt") as rawfile:
    with open("person_only_anno_2007_test.txt", "w")as resfile:
        line = rawfile.readline()
        while line:
            line = line.split(" ")
            if line[1].find("-1") == -1:
                resfile.write(line[0]+'\n')
            line = rawfile.readline()

