# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


spine1 = ["IP", "master", "network", "opencv", "visual"]
spine2 = ["bunsan", "eisei", "IPnext", "IPpro", "IPv6", "mac", "mobile", "operate", "packet", "sikiho"]
spine3 = ["asp", "critical", "gnu", "html", "java", "tcp", "window", "xhtml", "xp", "memory", "mysql", "network", "operate", "OS"]
spine4 = ["radius", "vpn", "xoops", "2003", "dis", "h264", "h323", "linkers", "php", "php5", "project"]
spine5 = ["program", "unix", "unix2", "virus", "eclipse", "mac", "obj", "operate"]
shelf = ["spine1", "spine2", "spine3", "spine4", "spine5"]
shelf_num = ["shelf1", "shelf2", "shelf3", "shelf4", "shelf5"]

cluster_size = 5

def threshold(cluster, center):

    #角度を求める
    dig_max = max(cluster[:,0])
    dig_min = min(cluster[:,0])

    #彩度を求める
    sat_max = max(cluster[:,1])
    sat_min = min(cluster[:,1])

    #明度を求める
    val_max = max(cluster[:,2])
    val_min = min(cluster[:,2])

    th_max = [dig_max, sat_max, val_max]
    th_min = [dig_min, sat_min, val_min]

    print 'dig_max = {}'.format(dig_max)
    print 'dig_min = {}'.format(dig_min)
    print 'dig_center = {}'.format(center[0])
    print 'sat_max = {}'.format(sat_max)
    print 'sat_min = {}'.format(sat_min)
    print 'val_max = {}'.format(val_max)
    print 'val_min = {}'.format(val_min)

    th_min = [dig_min, sat_min, val_min]
    th_max = [dig_max, sat_max, val_max]

    return [th_min, th_max]

#HSV空間でのクラスタリング
def clustering(shelf, spine, shelf_num):
    image = cv2.imread('/image/samples/' + shelf + '/' + spine + '.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)            
    Z = hsv.reshape((-1,3))
    HSV = Z.astype(np.float32) 
    
    # K-Means法
    #start1 = time.time() #K-means法の時間計測
        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)        
    K = cluster_size    
    ret,label,center=cv2.kmeans(HSV,                                    
                                K,                                  
                                None,                                  
                                criteria,                                   
                                25,                                 
                                cv2.KMEANS_RANDOM_CENTERS)
     
    #elapsed_time1 = time.time() - start1
    #print ("clustering:{0}".format(elapsed_time1)) + "[sec]"
            
    res = center[label.flatten()]
    print center   

    #各クラスタのデータ
    A = HSV[label.ravel() == 0]
    B = HSV[label.ravel() == 1]
    C = HSV[label.ravel() == 2]
    D = HSV[label.ravel() == 3]
    E = HSV[label.ravel() == 4]
    #F = HSV[label.ravel() == 5]
    #G = HSV[label.ravel() == 6]

    #色相の範囲検索
    region = []
    region.append(threshold(A, center[0]))
    region.append(threshold(B, center[1]))
    region.append(threshold(C, center[2]))
    region.append(threshold(D, center[3]))
    region.append(threshold(E, center[4]))

    print 'Asize = {}'.format(len(A))
    print 'Bsize = {}'.format(len(B))
    print 'Csize = {}'.format(len(C))
    print 'Dsize = {}'.format(len(D))
    print 'Esize = {}'.format(len(E))

    f1 = open('/image/color_match/clustering/mask/cluster5/' + shelf + '/' + spine + '.txt','w')
    f1.write(str(len(A)) + "\n")
    f1.write(str(len(B)) + "\n")
    f1.write(str(len(C)) + "\n")
    f1.write(str(len(D)) + "\n")
    f1.write(str(len(E)) + "\n") 
    f1.close()

    print np.c_[region]
    detect_color(hsv, np.c_[region], shelf, spine, shelf_num)

def detect_color(hsv_image, region, shelf, spine, shelf_num):
    #書棚画像の読み込み
    #shelf_image = cv2.imread('/Users/Syohei_Yamauchi/Google ドライブ/大澤・梅澤研究室/大学院/image/shelf1/IMG_50_0.jpg')
    shelf_image = cv2.imread('/image/samples/' + shelf_num + '/IMG_50_0.jpg')
    hsv_shelf = cv2.cvtColor(shelf_image, cv2.COLOR_BGR2HSV)
    maskA = cv2.inRange(hsv_shelf, region[0][0], region[0][1])
    maskB = cv2.inRange(hsv_shelf, region[1][0], region[1][1])
    maskC = cv2.inRange(hsv_shelf, region[2][0], region[2][1])
    maskD = cv2.inRange(hsv_shelf, region[3][0], region[3][1])
    maskE = cv2.inRange(hsv_shelf, region[4][0], region[4][1])
    masked_imageA = cv2.bitwise_and(shelf_image, shelf_image, mask=maskA)
    masked_imageB = cv2.bitwise_and(shelf_image, shelf_image, mask=maskB)
    masked_imageC = cv2.bitwise_and(shelf_image, shelf_image, mask=maskC)
    masked_imageD = cv2.bitwise_and(shelf_image, shelf_image, mask=maskD)
    masked_imageE = cv2.bitwise_and(shelf_image, shelf_image, mask=maskE)

    """
    dst = cv2.bitwise_or(masked_imageA, masked_imageD)
    dst = cv2.bitwise_or(dst, masked_imageE)
    dst = cv2.bitwise_or(dst, masked_imageC)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/test/' +spine+ '_mask3.jpg', dst)
    """

    dst = cv2.bitwise_or(masked_imageA, masked_imageB)
    dst = cv2.bitwise_or(dst, masked_imageC)
    dst = cv2.bitwise_or(dst, masked_imageD)
    dst = cv2.bitwise_or(dst, masked_imageE)

    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_mask.jpg', dst)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskA.jpg', maskA)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskB.jpg', maskB)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskC.jpg', maskC)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskD.jpg', maskD)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskE.jpg', maskE)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskedA.jpg', masked_imageA)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskedB.jpg', masked_imageB)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskedC.jpg', masked_imageC)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskedD.jpg', masked_imageD)
    cv2.imwrite('/image/color_match/clustering/mask/cluster5/' + shelf+ '/' +spine+ '_maskedE.jpg', masked_imageE)

if __name__=="__main__":

    #画像を入力  
    #image = cv2.imread('/Users/Syohei_Yamauchi/Google ドライブ/大澤・梅澤研究室/大学院/image/spine1/visual.jpg')
    #image = cv2.imread('/image/samples/' + shelf + '/' + spine + '.jpg')
    #image = cv2.imread('/image/samples/' + shelf[0] + '/' + spine1[4] + '.jpg') 
    #clustering_hsv(image)

    """
    for i in range(0, 5):   
        print shelf[0],spine1[i]    
        clustering(shelf[0], spine1[i])
    """

    for i in range(0, 10):
        print shelf[1],spine2[i]
        clustering(shelf[1], spine2[i], shelf_num[1])

    for i in range(0, 14):
        print shelf[2],spine3[i]
        clustering(shelf[2], spine3[i], shelf_num[2])

    for i in range(0, 11):
        print shelf[3],spine4[i]
        clustering(shelf[3], spine4[i], shelf_num[3])

    for i in range(0, 8):
        print shelf[4],spine5[i]
        clustering(shelf[4], spine5[i], shelf_num[4])

#cv2.imshow('Quantization', img_dst)
cv2.waitKey(0)
#cv2.destroyAllWindows()
