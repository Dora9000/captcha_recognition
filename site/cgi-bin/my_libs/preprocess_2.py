#!/usr/bin/env python
# coding: utf-8

from PIL import Image
from collections import deque
from random import randrange as rnd
import os

white = (255, 255, 255)
black = (0, 0, 0)


def get_path_to_save(PATH):
    PATH = PATH[:-7]
    return PATH 


#def get_path_to_image(index, s):
#    return 'C:\\Users\\User\\Desktop\\no_names\\  (%d).' % index + s


def generate_steps(dlt, flag):
    if flag:
        return [(i, j) for i in range(-dlt, dlt + 1) for j in range(-dlt, dlt + 1)]
    return [(i, j) for i in range(-dlt, dlt + 1) for j in range(-dlt, dlt + 1) if not (i == 0 and j == 0)]


def connect(mp, steps):
    w = len(mp)
    h = len(mp[0])
    colors = [[-1 for _ in range(h)] for _ in range(w)]

    def connect_bfs(start_x, start_y, color):
        q = deque()
        q.append((start_x, start_y))
        colors[start_x][start_y] = color
        while len(q):
            cx, cy = q.popleft()
            for dx, dy in steps:
                nx = cx + dx
                ny = cy + dy
                if min(nx, ny) < 0 or nx == w or ny == h:
                    continue
                if colors[nx][ny] != -1 or mp[nx][ny] != mp[cx][cy]:
                    continue
                colors[nx][ny] = color
                q.append((nx, ny))

    answer_size = 0
    for x in range(w):
        for y in range(h):
            if colors[x][y] == -1:
                connect_bfs(x, y, answer_size)
                answer_size += 1

    ans = [[] for _ in range(answer_size)]

    for x in range(w):
        for y in range(h):
            ans[colors[x][y]].append((x, y))

    return ans


def clear_corners(mp):
    w = len(mp)
    h = len(mp[0])
    colors = [[-1 for _ in range(h)] for _ in range(w)]
    steps = generate_steps(1, False)
    delta = 3

    def clear_bfs(start_x, start_y, color):
        q = deque()
        q.append((start_x, start_y))
        colors[start_x][start_y] = color
        while len(q):
            cx, cy = q.popleft()
            for dx, dy in steps:
                nx = cx + dx
                ny = cy + dy
                if min(nx, ny) < 0 or nx == w or ny == h:
                    continue
                if colors[nx][ny] != -1 or mp[nx][ny] != mp[cx][cy]:
                    continue
                colors[nx][ny] = color
                q.append((nx, ny))

    for x in range(w):
        for y in range(h):
            if min(x, w - 1 - x, y, h - 1 - y) >= delta:
                continue
            if mp[x][y] != white:
                clear_bfs(x, y, 0)

    for x in range(w):
        for y in range(h):
            if colors[x][y] != -1:
                mp[x][y] = white

    return mp


def k_means(list_of_pixels, k):
    arr = [[rnd(256), rnd(256), rnd(256)] for _ in range(k)]
    n = len(list_of_pixels)

    def get_class(color):
        dist = 10 ** 10
        current_class = -1
        for current_index in range(k):
            cur_dist = (color[0] - arr[current_index][0]) ** 2 \
                       + (color[1] - arr[current_index][1]) ** 2 \
                       + (color[2] - arr[current_index][2]) ** 2
            if cur_dist < dist:
                dist = cur_dist
                current_class = current_index
        return current_class

    classes = [[] for _ in range(k)]

    for i in range(n):
        classes[get_class(list_of_pixels[i][0])].append(i)

    error = 1
    while error:
        error = 0
        dr = [0 for _ in range(k)]
        dg = [0 for _ in range(k)]
        db = [0 for _ in range(k)]
        count = [1 for _ in range(k)]
        for i in range(k):
            class_r, class_g, class_b = arr[i]
            for val in classes[i]:
                count[i] += list_of_pixels[val][1]
                r, g, b = list_of_pixels[val][0]
                dr[i] += (r - class_r) * list_of_pixels[val][1]
                dg[i] += (g - class_g) * list_of_pixels[val][1]
                db[i] += (b - class_b) * list_of_pixels[val][1]

            classes[i].clear()
        for i in range(k):
            error += abs(dr[i]) // count[i] + abs(dg[i]) // count[i] + abs(db[i]) // count[i]
            arr[i][0] += dr[i] // count[i]
            arr[i][1] += dg[i] // count[i]
            arr[i][2] += db[i] // count[i]

        for i in range(n):
            classes[get_class(list_of_pixels[i][0])].append(i)

    answer = dict()
    for j in range(k):
        for index in classes[j]:
            answer[list_of_pixels[index][0]] = (arr[j][0], arr[j][1], arr[j][2])
    return answer


def filter_image(path):
    #name = get_path_to_image(number_of_image, "png")

    try:
        image = Image.open(path).convert('RGB')
    except FileNotFoundError:
        assert False

    w, h = image.size
    image = image.load()

    pixels = [[image[x, y] for y in range(h)] for x in range(w)]

    mp = {}
    for i in range(w):
        for j in range(h):
            cur = pixels[i][j]
            if cur not in mp:
                mp[cur] = 0
            mp[cur] += 1

    k = 10  # 5
    color_mp = k_means(list(mp.items()), k)
    for i in range(w):
        for j in range(h):
            pixels[i][j] = color_mp[pixels[i][j]]

    mp = {}
    for i in range(w):
        for j in range(h):
            cur = pixels[i][j]
            if cur not in mp:
                mp[cur] = 0
            mp[cur] += 1

    presents = 15  # 10 # если больше presents процентов на картинке данного цвета, то будем считать, что это фон
    total_size = w * h
    background_colors = set()
    background_colors.add(white)  # чтобы сразу убрать белые точки

    for color in mp:
        if mp[color] / total_size * 100 >= presents:
            background_colors.add(color)

    for x in range(w):
        for y in range(h):
            if pixels[x][y] in background_colors:
                pixels[x][y] = white

    return pixels


def main_filter_captcha_2(path): #number_of_image
    while True:
        #pixels = clear_corners(filter_image(number_of_image))
        pixels = clear_corners(filter_image(path))
        w = len(pixels)
        h = len(pixels[0])
        black_count = 0
        for x in range(w):
            for y in range(h):
                if pixels[x][y] != white:
                    pixels[x][y] = black
                    black_count += 1

        if 0.05 * w * h < black_count < 0.95 * w * h:
            break

    steps = generate_steps(1, True)
    classes = connect(pixels, steps)
    low_border = 10  # размер связкой области

    for current_class in classes:
        if len(current_class) <= low_border:
            for point in current_class:
                cx, cy = point
                pixels[cx][cy] = white

    # cuts
    oy = []
    for x in range(w):
        column = 0
        for y in range(h):
            if pixels[x][y] == black:
                column += 1
        oy.append(column)

    cuts = []
    low_border = 1
    i = 0
    start = 0
    while i < w:
        check = False
        while i < w and oy[i] < low_border:
            i += 1
        if i < w:
            check = True
            start = i
        while i < w and oy[i] >= low_border:
            i += 1
        if check:
            cuts.append((start, i))
        start = i

    if len(cuts) < 5:
        cuts1 = []
        difference = 5 - len(cuts)
        if difference == 5:
            i = 0
            j = w - 1
            for k in range(6):
                cuts1.append((i + (j - i) * k // 6, i + (j - i) * (k + 1) // 6))

        if difference == 4:
            i = 0
            j = w - 1
            for k in range(5):
                cuts1.append((i + (j - i) * k // 5, i + (j - i) * (k + 1) // 5))
        if difference < 4:
            c = []
            for i, j in cuts:
                if j - i <= 20:
                    cuts1.append((i, j))
                    continue
                else:
                    c.append((i, j))
            for i, j in c:
                if difference == 0:
                    cuts1.append((i, j))
                    continue
                if j - i > 60:
                    coefficient = min(difference + 1, 4)
                elif j - i > 40:
                    coefficient = min(difference + 1, 3)
                else:
                    coefficient = min(difference + 1, 2)
                for k in range(coefficient):
                    cuts1.append((int(i + (j - i) * k / coefficient), int(i + (j - i) * (k + 1) / coefficient)))
                difference = max(0, difference - coefficient)
        cuts = [cuts1[i] for i in range(len(cuts1))]
		

    cuts.sort()
    cuts2 = []
    for (i, j) in cuts:
        mn = h
        mx = 0
        for u in range(j - i):
            for k in range(h):
                if pixels[u + i][k] == white:
                    continue
                else:
                    mn = min(mn, k)
                    mx = max(mx, k)
        cuts2.append((mn, mx))

    #print(cuts2)

    img2 = Image.new('RGB', (w, h))
    mp = img2.load()
    for x in range(w):
        for y in range(h):
            mp[x, y] = pixels[x][y]
    # img2.show()
    itt = 0
    for (mnx, mxx) in cuts:
        (mnh, mxh) = cuts2[itt]
        img3 = img2.crop((mnx, mnh, mxx, mxh))
        size = (28, 28)
        img3 = img3.resize(size)
        # img3.show()
        itt += 1
        img3.save(get_path_to_save(path) + str(itt) + ".jpeg", "JPEG")

    return pixels

