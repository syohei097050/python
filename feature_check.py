# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from operator import itemgetter


spine1 = ["IP", "master", "network", "opencv", "visual"]
spine2 = ["bunsan", "eisei", "IPnext", "IPpro", "IPv6", "mac", "mobile", "operate", "packet", "sikiho"]
spine3 = ["asp", "critical", "gnu", "html", "java", "tcp", "window", "xhtml", "xp", "memory", "mysql", "network", "operate", "OS"]
spine4 = ["radius", "vpn", "xoops", "2003", "dis", "h264", "h323", "linkers", "php", "php5", "project"]
spine5 = ["program", "unix", "unix2", "virus", "eclipse", "mac", "obj", "operate"]
shelf = ["spine1", "spine2", "spine3", "spine4", "spine5"]
shelf_num = ["shelf1", "shelf2", "shelf3", "shelf4", "shelf5"]

#HSV空間でのクラスタリング
def check(shelf, spine, shelf_num):

    all_features = []#元画像の特徴点座標を格納する配列
    count = 0
    for line in open('/Users/syohei/Google ドライブ/大澤・梅澤研究室/大学院/データ/all_pixels/IP0.5_50_0_shelf.txt').readlines():
        data = line[:-1].split(' ')
        all_features.append([int(data[0]), int(data[1])])

    all_features =  list(set(map(tuple, all_features)))#重複の削除
    all_features.sort(key=itemgetter(0))#昇順に並べ替え(別にしなくてもいいけど確認しやすくするために)
    print all_features
    print len(all_features)

    mask_features = []#色抽出した画像の特徴点座標を格納する配列
    count = 0
    for line in open('/Users/syohei/Google ドライブ/大澤・梅澤研究室/大学院/データ/hsv1_mask2/IP0.5_50_0_shelf.txt').readlines():
        data = line[:-1].split(' ')
        mask_features.append([int(data[0]), int(data[1])])

    mask_features =  list(set(map(tuple, mask_features)))
    mask_features.sort(key=itemgetter(0))
    print mask_features
    print len(mask_features)

    match = []  
    #mask_featuresの座標がall_featuresの座標に含まれるかチェック
    for mask in mask_features:
        for all_pix in all_features:
            if mask == all_pix:
                match.append(mask)

    print match
    print len(match)

    recall = float(len(match)) / float(len(mask_features))#再現率を求める

    print recall    

if __name__=="__main__":

    #画像を入力  
    #image = cv2.imread('/Users/Syohei_Yamauchi/Google ドライブ/大澤・梅澤研究室/大学院/image/spine1/visual.jpg')
    #image = cv2.imread('/image/samples/' + shelf + '/' + spine + '.jpg')
    #image = cv2.imread('/image/samples/' + shelf[0] + '/' + spine1[4] + '.jpg') 
    check(shelf[0], spine1[0], shelf_num[0])

    """
    for i in range(0, 5):   
        print shelf[0],spine1[i]    
        check(shelf[0], spine1[i])

    for i in range(0, 10):
        print shelf[1],spine2[i]
        check(shelf[1], spine2[i], shelf_num[1])

    for i in range(0, 14):
        print shelf[2],spine3[i]
        check(shelf[2], spine3[i], shelf_num[2])

    for i in range(0, 11):
        print shelf[3],spine4[i]
        check(shelf[3], spine4[i], shelf_num[3])

    for i in range(0, 8):
        print shelf[4],spine5[i]
        check(shelf[4], spine5[i], shelf_num[4])
    """

#cv2.imshow('Quantization', img_dst)
cv2.waitKey(0)
#cv2.destroyAllWindows()
