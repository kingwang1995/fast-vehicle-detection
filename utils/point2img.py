import os
import math
import argparse
import numpy as np
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_path", type=str, default="../data/points", help="path to point cloud")
    parser.add_argument("--minx", type=float, default=0, help="min x")
    parser.add_argument("--maxx", type=float, default=12000., help="max x")
    parser.add_argument("--miny", type=float, default=4600., help="min y")
    parser.add_argument("--maxy", type=float, default=11000., help="max y")
    parser.add_argument("--img_size", type=int, default=416, help="img size")
    parser.add_argument("--img_path", type=str, default="../data/test", help="save img")
    opt = parser.parse_args()
    print(opt)

    filelist = os.listdir(opt.point_path)
    print(filelist)
    for file in filelist:
        file_name = file.split('.')[0]
        xlist = []
        ylist = []
        zlist = []
        with open(os.path.join(opt.point_path, file), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                    pass
                x, y, z = [math.fabs(float(i)) for i in lines.split(' ')]
                # if x >= opt.minx  and x < opt.maxx and y >= opt.miny and y < opt.maxy and z >= 50 and z < 2000:
                xlist.append(x)
                ylist.append(y)
                zlist.append(z)
        w = []
        h = []
        img = np.zeros([opt.img_size, opt.img_size])
        for i in range(len(xlist)):
            c = int((xlist[i] - opt.minx) / (opt.maxx - opt.minx) * opt.img_size)
            r = int((ylist[i] - opt.miny) / (opt.maxy - opt.miny) * (opt.img_size / 2))
            w.append(c)
            h.append(r)
            gray = int(zlist[i] / 2000. * 256)
            if gray > img[r, c]:
                img[r, c] = gray
        img_name = os.path.join(opt.img_path, file_name + '.jpg')
        cv2.imwrite(img_name, img)
        cv2.waitKey()
