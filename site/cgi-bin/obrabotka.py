#!/usr/bin/env python3

import cgi, os
import cgitb; cgitb.enable()
from PIL import Image
import io
import os
import sys
import cv2

form = cgi.FieldStorage()
fileitem = form['filename']

CUR_PATH = os.getcwd()
IMAGE_PATH = CUR_PATH  + '/pic/img.'
PATH = '/pic/'
type = 0
if fileitem.filename:
	fn = fileitem.filename
	message = 'Файл "' + fn + '" успешно загружен. '
	image_data = io.BytesIO(fileitem.value)
	image = Image.open(image_data).convert('RGB')
	w, h = image.size
	image_l = image.load()
	
	if w > 700 or h > 600:
		message = 'Файл слишком большой. '
	else:
		answer = "my_result"
		
		os.chdir(CUR_PATH + "/pic/")
		listdir = os.listdir()
		for file in listdir:
			if '.jpeg' in file:
				os.remove(file)
			if '.png' in file:
				os.remove(file)

		sys.path.append(CUR_PATH + "/cgi-bin/my_libs/")
		from my_libs.classifier import *
		pixel_map = [[image_l[x,y] for y in range(h)]for x in range(w)]
		if Classifier(pixel_map) == 1:
			type = 1
			image.save(IMAGE_PATH + "jpeg")
			from my_libs.preprocess_1 import *
			MainFilter_captcha1(IMAGE_PATH + "jpeg")
			from my_libs.recognition import *
			answer = recognition_1()
		else:
			type = 2
			image.save(IMAGE_PATH + "png")
			from my_libs.preprocess_2 import *
			main_filter_captcha_2(IMAGE_PATH + "png")
			from my_libs.recognition import *
			answer = recognition_2()

else:
	message = 'Файл не был загружен. '

images = []
listdir = os.listdir()
for file in listdir:
	if '.jpeg' in file and 'img' not in file:
		images.append(file)
images.sort()
for i in range(len(images)):
	images[i] = r'<img src="http://localhost:8000/pic/' + images[i] + '" >'


print("Content-type: text/html")
print()
print(message)
print("Результат распознавания = ")
print(answer)
print('<br/><br>')
print("Исходное изображение ")
print('<br/><br>')
if type == 1:
	print('<img src="http://localhost:8000//pic//img.jpeg" >')
if type == 2:
	print('<img src="http://localhost:8000//pic//img.png" >')
print('<br/><br>')
print('<br/><br>')

for i in range(len(images)):
	print(images[i])
	print(" Было распознано как ")
	print(answer[i])
	print('<br/><br>')