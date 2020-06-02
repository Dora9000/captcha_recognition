#!/usr/bin/env python
# coding: utf-8

import numpy as np
from math import fabs
from matplotlib import pyplot as plt
import copy
import cv2  ##download opencv-python
import os
from PIL import Image, ImageFilter, ImageDraw 
from collections import deque
from time import time
from random import randrange as rnd


def get_path_to_save(PATH):
    PATH = PATH[:-8]
    return PATH 

	
def open_color(path):
    #img = cv2.imread(GetPathToImage(number, "jpeg"))
    img = cv2.imread(path)
    #plt.imshow(img);
    return img

def image_show(img, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    return fig, ax

def noize_line(img):
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)
    kernel = np.ones((2,2), np.uint8)
    dilation = cv2.dilate(th, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion

def cut_luz(img):
    img[(img > 215) & (img < 255)] = 255
    return img

def no_background(img):
    for i in range(img.shape[0]-1,-1,-1):
        for j in range(img.shape[1]-1,-1,-1):
            img.itemset((i,j,0),max(0, min(254,-img.item(i,j,0) + img.item(0,0,0))))
            img.itemset((i,j,1),max(0, min(254,-img.item(i,j,1) + img.item(0,0,1))))
            img.itemset((i,j,2),max(0, min(254,-img.item(i,j,2) + img.item(0,0,2))))
    img = cv2.bitwise_not(img)
    return img

def dominant_color(img):
    average = img.mean(axis=0).mean(axis=0)
    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 7
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant

def preproseed(img):
    img2 = copy.copy(img)
    average = dominant_color(img2)
    for i in range(img.shape[0]-1,-1,-1):
        for j in range(img.shape[1]-1,-1,-1):
            img.itemset((i,j,0),max(0, min(254,-img.item(i,j,0) + average[0])))
            img.itemset((i,j,1),max(0, min(254,-img.item(i,j,1) + average[1])))
            img.itemset((i,j,2),max(0, min(254,-img.item(i,j,2) + average[2])))
        
    img = cv2.bitwise_not(img)
    img2 = no_background(img2)
    average = dominant_color(img2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img2.item(i,j,0) + img2.item(i,j,1) + img2.item(i,j,2) < average[0] + average[1] + average[2] - 150):
                img2.itemset((i,j,0),0)
                img2.itemset((i,j,1),0)
                img2.itemset((i,j,2),0)
            else:
                if (img2.item(i,j,0) + img2.item(i,j,1) + img2.item(i,j,2) < average[0] + average[1] + average[2] - 50):
                    img2.itemset((i,j,0),170)
                    img2.itemset((i,j,1),170)
                    img2.itemset((i,j,2),170)     
    img = cv2.addWeighted(img,0.4,img2,0.6,0)
    img = cut_luz(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = noize_line(img)
    return img



#DSU part

class DSU_2D:
    d = [[]]
    p = [[]]
    def __init__(self, n, m):
        self.d = [[1 for _ in range(m)] for __ in range(n)]
        self.p = [[(i, j) for j in range(m)] for i in range(n)]
    
    def get(self, x, y):
        if self.p[x][y] == (x, y):
            return (x, y)
        self.p[x][y] = self.get(self.p[x][y][0], self.p[x][y][1])
        return self.p[x][y]
    
    def uni(self, a, b):
        a = self.get(a[0], a[1])
        b = self.get(b[0], b[1])
        if a == b:
            return False
        if self.d[a[0]][a[1]] < self.d[b[0]][b[1]]:
            a, b = b, a
        self.p[b[0]][b[1]] = a
        self.d[a[0]][a[1]] += self.d[b[0]][b[1]]
        return True

    
    
def generate_steps(dlt, flag):
    if flag:
        return [(i, j) for i in range(-dlt, dlt + 1) for j in range(-dlt, dlt + 1)]  
    return [(i, j) for i in range(-dlt, dlt + 1) for j in range(-dlt, dlt + 1) if not (i == 0 and j == 0)]

def distance(dim,a, b, coefs):
    ans = 0
    if dim == 1:
        return abs(a - b) * coefs;
    for i in range(dim):
        ans += abs(a[i] - b[i]) * coefs[i];
    return ans
    
def connect(pixel_map, steps, border, coefs):
    w = len(pixel_map)
    h = len(pixel_map[0])
    dsu = DSU_2D(w, h)
    for x in range(w):
        for y in range(h):
            for (dx, dy) in steps:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if distance(3,pixel_map[x][y], pixel_map[nx][ny], coefs) <= border:
                    dsu.uni((x, y), (nx, ny))
    return dsu



def MainFilter_captcha1(path):
    img = open_color(path)
    image2 = copy.copy(img)
    image2 = preproseed(image2)
    white = 255
    black = 0
    h,w = image2.shape[0], image2.shape[1]
    pixel_map = [[(image2.item(y,x),image2.item(y,x),image2.item(y,x)) for y in range(h)] for x in range(w)]
    
    
    steps = generate_steps(1, True)
    dsu = connect(pixel_map, steps, 0, [1, 1, 1])
    my_mp = {}
    for i in range(w):
        for j in range(h):
            if dsu.get(i,j) not in my_mp:
                my_mp[dsu.get(i,j)] = []
            my_mp[dsu.get(i,j)].append((i,j))
    for key in my_mp:
        if len(my_mp[key]) <= 15:
            for val in my_mp[key]:
                pixel_map[val[0]][val[1]] = (255,255,255)

    steps = generate_steps(1, True)
    dsu = connect(pixel_map, steps, 0, [1, 1, 1])
    my_mp = {}
    for i in range (w):
        for j in range(h):
            if dsu.get(i,j) not in my_mp:
                my_mp[dsu.get(i,j)] = []
            my_mp[dsu.get(i,j)].append((i,j))
    for key in my_mp:
        if len(my_mp[key]) <= 5:
            for val in my_mp[key]:
                pixel_map[val[0]][val[1]] = (0,0,0)

    
    oy = []
    for x in range(w):
        stolb = 0
        for y in range(h):
            if pixel_map[x][y] == (0,0,0):
                stolb += 1
        if stolb <= 0 or x < 2 or x > w - 5:
            stolb = 0
        oy.append(stolb)


    #fig, ax = plt.subplots()
    ox = [i for i in range(w)]
    #ax.plot(ox, oy)
    
    cuts = []
    i = 0
    while i < w:
        check = False
        while i < w and oy[i] <= 0:
            i += 1
        if i < w:
            check = True
            start = i
        while i < w and oy[i] > 0:
            i += 1
        if check:
            cuts.append((start, i))
        start = i
        
    if(len(cuts) < 3 or len(cuts) > 4):
        cuts1 = []
    
        for q in range(len(cuts)):
            i,j = cuts[q]
            if i == 0 and j == 0:
                continue
            if j - i < 15:
                if q == 0:#to right
                    a,b = cuts[q+1]
                    cuts1.append((i, b))
                    cuts[q+1] = (0,0)

                elif q == len(cuts) - 1:#to left
                    a,b = cuts1[len(cuts1) - 1]
                    b = j
                    cuts1[len(cuts1) - 1] = (a,b)
                
                else:
                    il, jl = cuts[q-1]
                    ir, jr = cuts[q+1]
                    left = jl - il
                    right = jr - ir
                    if left <= right:#to left
                        a,b = cuts1[len(cuts1) - 1]
                        b = j
                        cuts1[len(cuts1) - 1] = (a,b)
                    else:#to right
                        a,b = cuts[q+1]
                        cuts1.append((i, b))
                        cuts[q+1] = (0,0)
            elif j - i > 35:
                border = 220
                d = []
                for x in range(j - i):
                    delta = 0
                    cnt = 0
                    for y in range(h):
                        if pixel_map[x+i,y] == (255,255,255):
                            continue
                        if pixel_map[x+i+1,y] == (255,255,255):
                            continue
                        cnt += 1
                        sm = 0
                        for k in range(3):
                            sm += (pixel_map[x+i][y][k] - pixel_map[x+i+1][y][k]) **2
                        if sm >= border:
                            delta += 1
                    if x > 15 and j - i - x > 15:
                        d.append((delta/max(cnt,1), x))
                d.sort()
                if len(d) == 0:
                    x = (j - i) // 2
                else:
                    delta, x = d[len(d) - 1]
                cuts1.append((i, i + x))
                cuts1.append((i + x, j))
            else:
                cuts1.append((i,j))
        cuts = cuts1
    
    cuts.sort()
	
    for x in range(w):
        for y in range(h):
            if pixel_map[x][y] == (255,255,255):
                continue
            else:
                pixel_map[x][y] = (0,0,0)

    
    cuts2 = []
    for (i,j) in cuts:
        mn = h
        mx = 0
        for u in range(j - i):
            for k in range(h):
                if pixel_map[u+i][k] == (255,255,255):
                    continue
                else:
                    mn = min(mn, k)
                    mx = max(mx, k)
        cuts2.append((mn,mx))
    
    #draw cuts _ vertical
    def draw_cut_vert():
        set_cuts = set()
        for i in range(len(cuts)):
            set_cuts.add(cuts[i][0])
            set_cuts.add(cuts[i][1])
        for i in range(w):
            if i in set_cuts:
                for j in range(h):
                    pixel_map[i][j] = (180,180,180)



    #draw cuts _ horizontal
    def draw_cut_hor():
        itt = 0
        for (i,j) in cuts:
            (h1,h2) = cuts2[itt]
            for a in range(j - i):
                pixel_map[a+i][h1] = (180,180,180)
                pixel_map[a+i][h2] = (180,180,180)
            itt += 1
    
    
    
    for x in range(w):
        for y in range(h):
            image2.itemset((y,x),pixel_map[x][y][0])
    
    #image_show(image2)
    itt = 0
    for (i,j) in cuts:
        (h1,h2) = cuts2[itt]
        crop_img = image2[h1:h2, i:j]
        size = (28, 28)
        crop_img = cv2.resize(crop_img, size)
        #image_show(crop_img) 
        cv2.imwrite(get_path_to_save(path) + str(itt) + ".jpeg", crop_img)
        itt+=1
    