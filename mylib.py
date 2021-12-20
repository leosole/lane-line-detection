# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections.abc import Iterable


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

# Faz a média das linhas da esquerda e da direita (através do coeficiente angular).
def average(image, lines, lane_height, prev_lines, memory, right_mem, left_mem):
	left = []
	right = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		if len(list(filter(lambda x: x > image.shape[1] or x < 0, [x1,x2]))) or len(list(filter(lambda y: y > image.shape[0] or y < 0, [y1,y2]))):
			print("Linha fora da imagem")
			break
		parameters = np.polyfit((x1, x2), (y1, y2), 1)
		slope = parameters[0]
		y_intercept = parameters[1]

		# Normalmente um slope positivo = left line  e slope negativo = right line mas neste caso, 
		# o eixo Y da imagem é invertido, logo os slopes são invertidos (OpenCV tem eixo Y invertido).
		if slope < -0.3:
			left.append((slope, y_intercept))
		elif slope > 0.3:
			right.append((slope, y_intercept))

	# Pega a média das linhas
	right_avg = np.average(right, axis=0)
	left_avg = np.average(left, axis=0)

	# Se não houver linhas, usar linhas anteriores
	# MEMORIA
	if prev_lines is not None:
		if np.isnan(right_avg).any() and (prev_lines[1] != [0,0,0,0]).all() and right_mem < memory:
			right_mem += 1
			prev_right = prev_lines[1]
			prev_right_parameters = np.polyfit((prev_right[0], prev_right[2]), (prev_right[1], prev_right[3]), 1)
			right_avg = prev_right_parameters
		else:
			right_mem = 0
		if np.isnan(left_avg).any() and (prev_lines[0] != [0,0,0,0]).all() and left_mem < memory:
			left_mem += 1
			prev_left = prev_lines[0]
			prev_left_parameters = np.polyfit((prev_left[0], prev_left[2]), (prev_left[1], prev_left[3]), 1)
			left_avg = prev_left_parameters
		else:
			left_mem = 0

	# Calcula interseção das linhas
	# x_i (b1-b2) / (m2-m1)
	if not np.isnan(right_avg).any() and not np.isnan(left_avg).any():
		x_i = (right_avg[1]-left_avg[1]) / (left_avg[0]-right_avg[0])
		lane_height = (right_avg[0] * x_i + right_avg[1]) / image.shape[0] + 0.02

	# Pega os pontos de início e fim de cada linha
	left_line = make_points(image, left_avg, lane_height)
	right_line = make_points(image, right_avg, lane_height)

	return np.array([left_line, right_line])

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
