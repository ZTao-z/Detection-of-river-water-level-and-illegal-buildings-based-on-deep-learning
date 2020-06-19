from pathlib import Path, PurePath
import cv2

if __name__ == '__main__':
    p = Path('./piaofu/piao/shenhe/JPEGImages/')
    files = [x for x in p.iterdir() if x.is_file()]
    for file in files:
        try:
            print(file.name)
            img = cv2.imread('./piaofu/piao/shenhe/JPEGImages/%s' % file.name, cv2.IMREAD_COLOR)
        except Exception:
            print(file.name)