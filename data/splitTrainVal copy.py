from pathlib import Path, PurePath

resultPath = "./video/buildingwater/ImageSets/Main/"
def splitDataset(path, filename):
    p = Path(path)
    files = [x for x in p.iterdir() if x.is_file()]
    count = 0
    with open(resultPath+filename+'trainval0.txt', 'w+') as f:
        with open(resultPath+filename+'train0.txt', 'w+') as ft:
            with open(resultPath+filename+'val0.txt', 'w+') as fv:
                for file in files:
                    f.write(file.stem + '\n')
                    if file.stem.find('v1') > -1 or file.stem.find('v2') > -1 or file.stem.find('v4') > -1 or file.stem.find('v5') > -1 or file.stem.find('v6') > -1:
                        ft.write(file.stem + '\n')
                    elif file.stem.find('v3') > -1:
                        fv.write(file.stem + '\n')
                    count += 1

splitDataset('./video/buildingwater/Annotations', '')
