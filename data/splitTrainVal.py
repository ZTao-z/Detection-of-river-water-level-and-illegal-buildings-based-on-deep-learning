from pathlib import Path, PurePath

resultPath = "./video/buildingwater/ImageSets/Main/"
def splitDataset(path, filename):
    p = Path(path)
    files = [x for x in p.iterdir() if x.is_file()]
    count = 0
    with open(resultPath+filename+'trainval.txt', 'w+') as f:
        with open(resultPath+filename+'train.txt', 'w+') as ft:
            with open(resultPath+filename+'val.txt', 'w+') as fv:
                for file in files:
                    f.write(file.stem + '\n')
                    if count % 5 == 4:
                        fv.write(file.stem + '\n')
                    else:
                        ft.write(file.stem + '\n')
                    count += 1

splitDataset('./video/buildingwater/Annotations', '')
