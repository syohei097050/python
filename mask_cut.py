# -*- coding: utf-8 -*-
"""
マスク画像をもとに元画像を切り取る（透明にする）
"""
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

if __name__=="__main__":

	image = cv2.imread('/image/samples/shelf1/IMG_50_0.jpg')
	mask = cv2.imread('/image/color_match/clustering/mask/cluster5/spine1/visual_maskC.jpg',0)

	channels = cv2.split(image)
	channels.append(mask)
	new_image = cv2.merge(channels)

	print new_image

	cv2.imwrite("/image/samples/mask_cut.jpg", new_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()