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

def detection(img):
  SIFT = cv2.xfeatures2d.SIFT_create()
  keypoint = SIFT.detect(img, None)
  detected_img = cv2.drawKeypoints(img, keypoint, None)

  return detected_img, len(keypoint)

  cv2.imwrite("/image/matching.jpg", out)


if __name__=="__main__":

  img1 = cv2.imread('/image/color_match/clustering/mask/cluster5/test/' + spine1[1] + '_mask.jpg')
  #img1 = cv2.imread('/image/samples/shelf1/IMG_50_0.jpg')
  src1 = detection(img1)
  cv2.imwrite("/image/detected_" + spine1[1] + ".jpg", src1[0])
  #cv2.imwrite("/image/detected_IMG_50_0.jpg", src1[0])
  

  """
  img2 = cv2.imread('/image/color_match/clustering/mask/cluster5/test/' + spine1[1] + '_mask.jpg')
  img3 = cv2.imread('/image/color_match/clustering/mask/cluster5/test/' + spine1[2] + '_mask.jpg')
  img4 = cv2.imread('/image/color_match/clustering/mask/cluster5/test/' + spine1[3] + '_mask.jpg')
  img5 = cv2.imread('/image/color_match/clustering/mask/cluster5/test/' + spine1[4] + '_mask.jpg')

  src1 = detection(img1)
  src2 = detection(img2)
  src3 = detection(img3)
  src4 = detection(img4)
  src5 = detection(img5)

  cv2.imwrite("/image/detected_" + spine1[0] + ".jpg", src1[0])
  cv2.imwrite("/image/detected_" + spine1[1] + ".jpg", src2[0])
  cv2.imwrite("/image/detected_" + spine1[2] + ".jpg", src3[0])
  cv2.imwrite("/image/detected_" + spine1[3] + ".jpg", src4[0])
  cv2.imwrite("/image/detected_" + spine1[4] + ".jpg", src5[0])
  """

  f1 = open('/image/detected_master.txt','w')
  f1.write(str(src1[1]) + "\n")
  """
  f1.write(src2[1] + "\n")
  f1.write(src3[1] + "\n")
  f1.write(src4[1] + "\n")
  f1.write(src5[1] + "\n")
  """
  f1.close()

  cv2.waitKey(0)