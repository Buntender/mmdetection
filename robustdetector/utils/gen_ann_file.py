import os
# path = "work_dirs_data0/CATA_VOC_FRCNN_SSD/ap0"
path = "work_dirs_data0/CATA_VOC_FRCNN_YOLO/ap5"

with open(f"{path}/../anno.txt", 'w') as file:
    for dirpath, dirnames, filenames in os.walk(path):
        file.write("\n".join([filename[:-4] for filename in filenames]))