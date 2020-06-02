#!/usr/bin/env python
# coding: utf-8


from PIL import Image, ImageDraw

def Classifier(pixel_map):
    
    def Grey_scale(pixel):
        border = 25
        d1 = abs(pixel[0]-pixel[1])**2
        d2 = abs(pixel[0]-pixel[2])**2
        d3 = abs(pixel[1]-pixel[2])**2
        if d1 <= border and d2 <= border and d3 <= border:
            return True
        return False

    w, h = len(pixel_map), len(pixel_map[0])
    
    grey_colors = 0
    for i in range(w):
            for j in range(h):
                if Grey_scale(pixel_map[i][j]):
                    grey_colors += 1

    if grey_colors > 5000:
        return 1
    else:
        return 2
