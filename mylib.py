# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections.abc import Iterable
from skimage import io
from sklearn.cluster import MeanShift
from collections import Counter
from PIL import Image

#Aplica uma máscara é obtem a ROI
def interested_region(img, vertices):
    #if len(img.shape) > 2: 
    #    mask_color_ignore = (255,) * img.shape[2]
    #else:
    #    mask_color_ignore = 255
    #    
    #cv2.fillPoly(np.zeros_like(img), vertices, mask_color_ignore)
    #return cv2.bitwise_and(img, np.zeros_like(img))
    mask = np.zeros_like(img)    
    mask = cv2.fillPoly(mask, vertices, 255)
    mask = cv2.bitwise_and(img, mask)
    return mask

# Faz um Canny automático
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def most_frequent(List):
    counted = Counter(List)
    return counted.most_common(1)[0][0]

# Faz a média das linhas da esquerda e da direita (através do coeficiente angular).
def average(image, lines, lane_height, prev_lines, memory, right_mem, left_mem, using_mem=False, min_slope=0.3, pavement=True, max_slope=1):
	left = []
	right = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		if len(list(filter(lambda x: x > image.shape[1] or x < 0, [x1,x2]))) or len(list(filter(lambda y: y > image.shape[0] or y < 0, [y1,y2]))) or ([x1, y1, x2, y2] == [0,0,0,0]) :
			print("Linha fora da imagem")
			break
		parameters = np.polyfit((x1, x2), (y1, y2), 1)
		slope = parameters[0]
		y_intercept = parameters[1]
		# Normalmente um slope positivo = left line  e slope negativo = right line mas neste caso, 
		# o eixo Y da imagem é invertido, logo os slopes são invertidos (OpenCV tem eixo Y invertido).
		if slope < -min_slope and slope > -max_slope:
			left.append((slope, y_intercept))
		elif slope > min_slope and slope < max_slope:
			right.append((slope, y_intercept))
	if len(left):
		ms_left = MeanShift().fit(left)
	if len(right):
		ms_right = MeanShift().fit(right)
	# af_left = AffinityPropagation(random_state=0).fit(left)
	# af_right = AffinityPropagation(random_state=0).fit(right)
	# Pega a média das linhas
	if pavement:
		right_avg = np.average(right, axis=0)
		left_avg = np.average(left, axis=0)
	else:
		if len(right):
			right_counts = np.bincount(ms_right.labels_)
			# right_idxs = np.where(ms_right.labels_ == np.argmax(right_counts))[0]
			right_idx = np.where(ms_right.labels_ == np.argmax(right_counts))[0][0]
			# right_avg = np.average([x for x in right if right.index(x) in right_idxs], axis=0)
			right_avg = right[right_idx]  
		else: 
			right_avg = np.nan
		if len(left):
			left_counts = np.bincount(ms_left.labels_)
			# left_idxs = np.where(ms_left.labels_ == np.argmax(left_counts))[0]
			left_idx = np.where(ms_left.labels_ == np.argmax(left_counts))[0][0]
			left_avg = left[left_idx]  
			# left_avg = np.average([x for x in left if left.index(x) in left_idxs], axis=0)
		else: 
			left_avg = np.nan

	# Se não houver linhas, usar linhas anteriores
	# MEMORIA
	if prev_lines is not None:
		if np.isnan(right_avg).any() and (prev_lines[1] != [0,0,0,0]).all() and right_mem < memory:
			right_mem += 1
			prev_right = prev_lines[1]
			prev_right_parameters = np.polyfit((prev_right[0], prev_right[2]), (prev_right[1], prev_right[3]), 1)
			right_avg = prev_right_parameters
		elif not using_mem:
			right_mem = 0
		if np.isnan(left_avg).any() and (prev_lines[0] != [0,0,0,0]).all() and left_mem < memory:
			left_mem += 1
			prev_left = prev_lines[0]
			prev_left_parameters = np.polyfit((prev_left[0], prev_left[2]), (prev_left[1], prev_left[3]), 1)
			left_avg = prev_left_parameters
		elif not using_mem:
			left_mem = 0

	# Calcula interseção das linhas
	# x_i (b1-b2) / (m2-m1)
	if not np.isnan(right_avg).any() and not np.isnan(left_avg).any():
		x_i = (right_avg[1]-left_avg[1]) / (left_avg[0]-right_avg[0])
		lane_height = (right_avg[0] * x_i + right_avg[1]) / image.shape[0] + 0.02

	# Pega os pontos de início e fim de cada linha
	left_line = make_points(image, left_avg, lane_height)
	right_line = make_points(image, right_avg, lane_height)

	return np.array([left_line, right_line]), right_mem, left_mem

# Cria os pontos das linhas finais
def make_points(image, average, lane_height=0.8): 
	theRet = np.array([0,0,0,0])

	if isinstance(average, Iterable):
		slope, y_int = average 
		y1 = image.shape[0]

		# Define a altura das linhas que será a mesma para ambas
		y2 = int(y1 * (lane_height))  # Constante lane_height bastante dependente da imagem

		#Se y=mx+b   então   x = (y-b) / m
		x1 = int((y1 - y_int) // slope)
		x2 = int((y2 - y_int) // slope)

		theRet = np.array([x1, y1, x2, y2])

	return theRet

# Cria camada com as linhas
def display_lines(image, lines):
	lines_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line
			cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
	return lines_image

# Filtro P/B
# https://stackoverflow.com/questions/55185251/what-is-the-algorithm-behind-photoshops-black-and-white-adjustment-layer
def black_and_white_adjustment(img, weights):
    rw, yw, gw, cw, bw, mw = weights 

    h, w = img.shape[:2]
    min_c = np.min(img, axis=-1).astype(np.float)
    # max_c = np.max(img, axis=-1).astype(np.float)

    # Can try different definitions as explained in the Ligtness section from
    # https://en.wikipedia.org/wiki/HSL_and_HSV
    # like: luminance = (min_c + max_c) / 2 ...
    luminance = min_c 
    diff = img - min_c[:, :, None]

    red_mask = (diff[:, :, 0] == 0)
    green_mask = np.logical_and((diff[:, :, 1] == 0), ~red_mask)
    blue_mask = ~np.logical_or(red_mask, green_mask)

    c = np.min(diff[:, :, 1:], axis=-1)
    m = np.min(diff[:, :, [0, 2]], axis=-1)
    yel = np.min(diff[:, :, :2], axis=-1)

    luminance = luminance + red_mask * (c * cw + (diff[:, :, 1] - c) * gw + (diff[:, :, 2] - c) * bw) \
                + green_mask * (m * mw + (diff[:, :, 0] - m) * rw + (diff[:, :, 2] - m) * bw)  \
                + blue_mask * (yel * yw + (diff[:, :, 0] - yel) * rw + (diff[:, :, 1] - yel) * gw)

    return np.clip(luminance, 0, 255).astype(np.uint8)

def reduce_colors(img, n_colors):
	# arr = img.reshape((-1, 3))
	# kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
	# labels = kmeans.labels_
	# centers = kmeans.cluster_centers_
	# less_colors = centers[labels].reshape(img.shape).astype('uint8')
	# return less_colors
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)
	im_pil = im_pil.convert('P', palette=Image.ADAPTIVE, colors=n_colors)
	im_pil = im_pil.convert('RGB', palette=Image.ADAPTIVE)
	# For reversing the operation:
	im_np = np.asarray(im_pil)
	return cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)