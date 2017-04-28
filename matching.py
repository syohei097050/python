# -*- coding: utf-8 -*-
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

if __name__=="__main__":

  img1 = cv2.imread('/image/opencv.jpg')
  img2 = cv2.imread('/image/IMG_50_0.jpg')
  #img_keypoint = img
  
  SIFT = cv2.xfeatures2d.SIFT_create()
  keypoint1, descript1 = SIFT.detectAndCompute(img1, None)
  keypoint2, descript2 = SIFT.detectAndCompute(img2, None)

  # 比較器作成
  bf = cv2.BFMatcher(cv2.NORM_L2)

  # 画像への特徴点の書き込み
  matches = bf.match(descript1, descript2)
  matches = sorted(matches, key = lambda x:x.distance)

  # 出力画像作成 表示
  h1, w1, c1 = img1.shape[:3]
  h2, w2, c2 = img2.shape[:3]
  height = max([h1,h2])
  width = w1 + w2
  out = np.zeros((height, width, 3), np.uint8)

  cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches[:50],out, flags=0)
  cv2.imwrite("/image/matching.jpg", out)
  #cv2.waitKey(0)
  cv2.destroyAllWindows()