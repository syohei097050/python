# -*- coding: utf-8 -*-
"""
背表紙画像の特徴点周辺の画素抽出
"""
import numpy as np
import cv2
import colorsys

#shelf = ["shelf1", "shelf2", "shelf3", "shelf4", "shelf5"]
syukusyo = ["_50%", "_60%", "_70%", "_80%", "_90%"]
image = "IMG_50_0"
spine1 = ["IP", "master", "network", "opencv", "visual"]
spine2 = ["bunsan", "eisei", "IPnext", "IPpro", "IPv6", "mac", "mobile", "operate", "packet", "sikiho"]
spine3 = ["asp", "critical", "gnu", "html", "java", "tcp", "window", "xhtml", "xp", "memory", "mysql", "network", "operate", "OS"]
spine4 = ["radius", "vpn", "xoops", "2003", "dis", "h264", "h323", "linkers", "php", "php5", "project"]
spine5 = ["program", "unix", "unix2", "virus", "eclipse", "mac", "obj", "operate"]
shelf = ["spine1", "spine2", "spine3", "spine4", "spine5"]

"""
def write(features, syukusyo, detector):
	f = open('/image/color_match/clustering/txt/features' + syukusyo + '_' + detector + '.txt','a')
	f.write(str(features))
	f.write('\n')
	f.close()
"""

def detection(shelf, spine):

	#img = cv2.imread('/image/samples/' + shelf + '/' + spine + '.jpg') 
  img = cv2.imread('/Users/syohei/Google ドライブ/大澤・梅澤研究室/大学院/image/spine1/opencv.jpg')

	#AKAZE = cv2.AKAZE_create()
	#BRISK = cv2.BRISK_create()
	#ORB = cv2.ORB_create()
  SIFT = cv2.xfeatures2d.SIFT_create()
	#SURF = cv2.xfeatures2d.SURF_create()

	#key_AKAZE = AKAZE.detect(img)
	#key_BRISK = BRISK.detect(img)
	#key_ORB = ORB.detect(img)
	key_SIFT = SIFT.detect(img)
	#key_SURF = SURF.detect(img)

	features_point = []
	features_point2 = []
	
	for  i in range(len(key_SIFT)):
		keypoint = key_SIFT[i].pt
		features_point.append([int(keypoint[0]), int(keypoint[1])])
		features_point2.append([int(keypoint[1]), int(keypoint[0])])

	features_point = list(set(map(tuple, features_point)))
	print features_point[0]
	print features_point[0][0]
	print len(features_point)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	features = [] #1個前のfor文と一緒にできると思う

	tmp = []
	tmp.append(img[features_point[0][1]][features_point[0][0]])
	tmp.append(img[features_point[1][1]][features_point[1][0]])

	tmp2 = [img[features_point[i][1]][features_point[i][0]] for i in xrange(len(features_point))]
	tmp3 = [img[features_point2[i]] for i in xrange(len(features_point))]

	#print tmp3
	print len(tmp3[0])

	print tmp
	print tmp[0][0]

	for  i in range(len(features_point)):
		features.append(img[features_point[i][1]][features_point[i][0]])

	print type(features_point[0][1])
	#print features
	Z = tmp3.reshape((-1,3))
	print features

	"""
	detected_img = []
	for  i in range(len(features_point)):
		detected_img.append(img[features_point[i][1]][features_point[i][0]])#注意！
	detected_img = np.array(detected_img, dtype=np.uint8)
	print detected_img
	"""

	"""
	hsv = cv2.cvtColor(detected_img, cv2.COLOR_BGR2HSV)
	Z = hsv.reshape((-1,3))
   	Z = Z.astype(np.float32) 

	cos = np.c_[np.cos(np.radians(Z[:,0]*2))*Z[:,1]*(Z[:,2]/255)]
   	sin = np.c_[np.sin(np.radians(Z[:,0]*2))*Z[:,1]*(Z[:,2]/255)]
   	HSV = np.c_[cos,sin,Z[:,2]]
   	elapsed_time1 = time.time() - start1
   	print ("calcurate:{0}".format(elapsed_time1)) + "[sec]"

   	# K-Means法
   	start2 = time.time() #K-means法の時間計測
        
   	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)        
   	K = cluster_size    
   	ret,label,center=cv2.kmeans(HSV,                                    
   	                            K,                                  
   	                            None,                                  
   	                            criteria,                                   
   	                            25,                                 
   	                            cv2.KMEANS_RANDOM_CENTERS)
   	
   	res = center[label.flatten()]
   	#print center   
    
   	#各クラスタのデータ
   	A = Z[label.ravel() == 0]
   	B = Z[label.ravel() == 1]
   	C = Z[label.ravel() == 2]
   	D = Z[label.ravel() == 3]
   	E = Z[label.ravel() == 4]	
    
   	elapsed_time2 = time.time() - start2
   	print ("clustering:{0}".format(elapsed_time2)) + "[sec]"
   	
   	#色相の範囲検索
   	start3 = time.time() #閾値計算の時間計測
   	region = []
   	region.append(threshold(A, center[0]))
   	region.append(threshold(B, center[1]))
   	region.append(threshold(C, center[2]))
   	region.append(threshold(D, center[3]))
   	region.append(threshold(E, center[4]))
   	elapsed_time3 = time.time() - start3
   	print ("threshold:{0}".format(elapsed_time3)) + "[sec]"
	"""

if __name__=="__main__":

    #shelf = shelf[0]
    #spine =spine1[4]

    #画像を入力  
    #image = cv2.imread('/Users/syohei/Google ドライブ/大澤・梅澤研究室/大学院/image/' + shelf[0] + '/' + spine1[4] + '.jpg')
    #image = cv2.imread('/image/samples/' + shelf[0] + '/' + spine1[4] + '.jpg') 
    '''
    for i in range(0, 5):
		print shelf[i],syukusyo[0]
		detection(shelf[i], syukusyo[0])

    
    for i in range(0, 4):   
        print shelf[0],syukusyo[i]    
        detection(shelf[0], syukusyo[i])

    for i in range(0, 4):
        print shelf[1],syukusyo[i]
        detection(shelf[1], syukusyo[i])

    for i in range(0, 4):
        print shelf[2],syukusyo[i]
        detection(shelf[2], syukusyo[i])

    for i in range(0, 4):
        print shelf[3],syukusyo[i]
        detection(shelf[3], syukusyo[i])

    for i in range(0, 4):
        print shelf[4],syukusyo[i]
        detection(shelf[4], syukusyo[i])
    '''
    detection(shelf[0], spine1[4])
    #cv2.imshow('Quantization', img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()