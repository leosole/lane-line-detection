# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
# import imutils
import glob
from configparser import ConfigParser
import matplotlib.pyplot as plt
import mylib as mylib

#Leitura das configurações
config_object = ConfigParser(allow_no_value=True)
config_object.read("config.ini")
param = config_object["PARAMETERS"]
inputdata = config_object["INPUT"]


#Definições de constantes
DEBUG = bool(int(param['DEBUG'])) 
SOURCE = inputdata['VIDEO'] 
LANE_HEIGHT = float(param['LANE_HEIGHT']) 
SAVE_VIDEO = bool(int(param["SAVE_VIDEO"]))
FRAME_MEMORY = int(param["FRAME_MEMORY"])
FRAME_OVERLAP = int(param["FRAME_OVERLAP"])
CONTRAST_FACTOR = float(param["CONTRAST_FACTOR"])
THRESHOLD = int(param["THRESHOLD"])
PAVEMENT = bool(int(param["PAVEMENT"]))

#Inicializações
debugStart = 0
debugStop = 0
debugCtd = 1

# Abre o vídeo
cap = cv2.VideoCapture(SOURCE)

# Verifica se a captura de vídeo teve sucesso
if (cap.isOpened()== False): 
	print("ERRO: Não foi possível abrir a entrada de vídeo")

# Inicializa o cronômetro para o cálculo de fps
if(DEBUG):
	debugStart = time.time()
if(SAVE_VIDEO):
	output_name = "output_mem" + str(FRAME_MEMORY) + "_overlap" + str(FRAME_OVERLAP) + "_contrast" + str(CONTRAST_FACTOR) + ".mp4"
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_name,fourcc, 20.0, (640*2,360*2))

right_mem = 0
left_mem = 0
prev_frames = list()
# Fica lendo o vídeo
while(cap.isOpened()):
	# Conta os frames para realizar o cálculo de fps (estou aceitando que inicialize com uma unidade a mais)
	if(DEBUG):
		debugCtd = debugCtd + 1

	# Captura frame-por-frame
	ret, frame = cap.read()
	if ret == True:
		#Reduz o tamanho da imagem
		#frameResized = imutils.resize(frame, width=IMAGE_SIZE)
		imageHeight = frame.shape[0]
		imageWidth = frame.shape[1]

		# Dependente de dataset: Aplica uma máscara trapezoidal no objeto de interesse.
		imshape = frame.shape
		# Opção original
		# lower_left = [imshape[1]/9,imshape[0]]
		# lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
		# top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
		# top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
		# Opção baseada no minha visão já com viés
		if PAVEMENT:
			min_slope = 0.3
			lower_left =  [  imshape[1]/9 ,   imshape[0]    ]
			lower_right = [7*imshape[1]/9 ,   imshape[0]    ]
			top_left =    [4*imshape[1]/9 , 7*imshape[0]/10 ]
			top_right =   [4*imshape[1]/9 , 7*imshape[0]/10 ]
		else:
			min_slope = 0.1
			lower_left =  [  	  0  	 , 8*imshape[0]/11 ]
			lower_right = [  imshape[1]  , 8*imshape[0]/11 ]
			top_left =    [3*imshape[1]/9, 5*imshape[0]/11 ]
			top_right =   [6*imshape[1]/9, 5*imshape[0]/11 ]
		vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

		# Pré-processamento
		frameView = frame.copy()

		if PAVEMENT:
			frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		else:
			# frameGray = mylib.black_and_white_adjustment(frame,[-3, -3, 3, 3, 3, -3])
			frameGray = cv2.GaussianBlur(frame, (11, 11), 0)
			# frameGray = cv2.threshold(frameGray,127,255,cv2.THRESH_BINARY)[1]
			# frameGray = cv2.cvtColor(frameGray, cv2.COLOR_BGR2GRAY)

		#  Threshold automático
		mask = np.zeros_like(frameGray) 
		mask = cv2.fillPoly(mask, vertices, 255)
		if PAVEMENT:
			# Aumenta o contraste
			threshold = int(cv2.mean(frameGray, mask)[0] * 2)
			frameGray = (frameGray ** CONTRAST_FACTOR) / (threshold ** (CONTRAST_FACTOR - 1))
			frameGray = np.clip(frameGray, 0, 255)
			frameGray = np.uint8(frameGray)
		else:
			# Reduz número de cores
			effect_vertices = [sum(x) for x in zip(vertices, [[1,-1],[-1,-1],[-1,1],[1,1]])]
			effect_mask = np.zeros_like(frameGray) 
			effect_mask = cv2.fillPoly(effect_mask, effect_vertices, [255,255,255])
			frameGray = cv2.bitwise_and(frameGray, effect_mask)
			frameGray = mylib.reduce_colors(frameGray,3)		

		# Adiciona informação dos frames anteriores
		if not prev_frames:
			prev_frames.append(frameGray)
		else:
			temp = frameGray
			for prev_frame in prev_frames:
				frameGray = cv2.bitwise_or(prev_frame, frameGray)
			prev_frames.append(temp)
		if len(prev_frames) > FRAME_OVERLAP:
			prev_frames.pop(0)

		# Dependente de dataset: as linhas da estrada são sempre amarelas e brancas. O Amarelo é um 
		# transtorno para ser isolado no espaço RBG, então converte para o espaço HSV (Hue Value Saturation)
		# A escolha abaixo de amarelo têm sido muito usada nos tutoriais, mas pode ser modificada. 
		# Para a linha branca, trabalho com os níveis de conza mesmo.
		lower_yellow = np.array([20, 100, 100], dtype = "uint8")
		upper_yellow = np.array([30, 255, 255], dtype="uint8")
		if PAVEMENT:
			mask_yellow = cv2.inRange(frameHSV, lower_yellow, upper_yellow)
			mask_white = cv2.inRange(frameGray, 130, 255)
			mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
			frameYWMask = cv2.bitwise_and(frameGray, mask_yw)
			frameBlur = cv2.GaussianBlur(frameGray, (5, 5), 0)
		else:
			frameYWMask = frameGray
			# frameBlur = cv2.GaussianBlur(frameGray, (15, 15), 0)
			# frameBlur = cv2.threshold(frameBlur,127,255,cv2.THRESH_BINARY)[1]
			frameBlur = cv2.medianBlur(frameGray, 7)
		
		# Tavez devesse tentar pegar a faixa amarela no Canny passando frameYWMask, por hora, não é usado.
		frameCanny = mylib.auto_canny(frameBlur)

		
		frameROI = mylib.interested_region(frameCanny, vertices)

		# Transformada Hough
		# Param 1: gradientes isolados
		# Param 2 e 3: Definindo o tamanhol do bin, 2 é e 3 é theta
		# Param 4: Mínima quantidade de interseções necessárias para um bin ser considerado linha 
		# Param 5: Placeholder array
		# Param 6: Comprimento mínimo da linha
		# Param 7: Máximo de falhas na linha
		if PAVEMENT:
			lines = cv2.HoughLinesP(frameROI, rho=1, theta=np.pi/180, threshold=20, 
				lines=np.array([]), minLineLength=40, maxLineGap=120)
		else:
				lines = cv2.HoughLinesP(frameROI, rho=1, theta=np.pi/180, threshold=10, 
				lines=np.array([]), minLineLength=40, maxLineGap=120)

		if lines is None:
			if 'left_and_right_lines' not in locals() or (right_mem > FRAME_MEMORY and left_mem > FRAME_MEMORY):
				print("ERRO: Nenhuma linha encontrada")
				frameLanes = frameView
			else:
				right_mem += 1
				left_mem += 1
				left_and_right_lines, right_mem, left_mem = mylib.average(frameROI,left_and_right_lines,LANE_HEIGHT, left_and_right_lines, FRAME_MEMORY, right_mem, left_mem, True, min_slope, PAVEMENT)
				black_lines = mylib.display_lines(frameView, left_and_right_lines)
				frameLanes = cv2.addWeighted(frameView, 0.8, black_lines, 1, 1)
		else:
			# Faz a média das linhas da tranformada 
			if 'left_and_right_lines' in locals():
				left_and_right_lines, right_mem, left_mem = mylib.average(frameROI,lines,LANE_HEIGHT, left_and_right_lines, FRAME_MEMORY, right_mem, left_mem, pavement=PAVEMENT)
			else:
				left_and_right_lines, right_mem, left_mem = mylib.average(frameROI,lines,LANE_HEIGHT, None, FRAME_MEMORY, right_mem, left_mem, pavement=PAVEMENT)

			# Prepara para fazer display
			black_lines = mylib.display_lines(frameView, left_and_right_lines)
			frameLanes = cv2.addWeighted(frameView, 0.8, black_lines, 1, 1)
		
		# Faz o display em uma janela
		if(DEBUG or SAVE_VIDEO):
			if PAVEMENT:
				d3_frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
				d3_frameBlur = cv2.cvtColor(frameBlur, cv2.COLOR_GRAY2BGR)			
				d3_frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2BGR)
				d3_frameCanny = cv2.polylines(d3_frameCanny, vertices, True ,(0,0,255),2)
				d3_frameYWMask = cv2.cvtColor(frameYWMask, cv2.COLOR_GRAY2BGR)			
				d3_frameROI = cv2.cvtColor(frameROI, cv2.COLOR_GRAY2BGR)
				# Adiciona info de threshold automático
				cv2.putText(d3_frameBlur, f'threshold: {threshold}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
				numpy_horizontal1 = np.hstack((frameLanes, d3_frameCanny))
				numpy_horizontal2 = np.hstack((d3_frameBlur, d3_frameROI))
			else:
				d3_frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2BGR)
				d3_frameROI = cv2.cvtColor(frameROI, cv2.COLOR_GRAY2BGR)
				numpy_horizontal1 = np.hstack((frameLanes, d3_frameCanny))
				numpy_horizontal2 = np.hstack((frameBlur, d3_frameROI))
			numpy_2x2 = np.vstack((numpy_horizontal1, numpy_horizontal2))
			if(DEBUG):
				cv2.imshow('Frame',numpy_2x2)
			if(SAVE_VIDEO):
				out.write(numpy_2x2)

		# Pressiona tecla Q para sair
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	# Finaliza o loop se não conseguir sucesso ao ler o video de entrada
	else: 
		print("ERRO: Não foi possível ler a entrada de vídeo")
		break

# Finaliza o cronômetro e cálcula o fps
if(DEBUG):
	debugStop = time.time()
	fps = debugCtd/(debugStop-debugStart)
	print("DEBUG: {} fps\n".format(fps))

# Libera o video capture object
cap.release()
# SALVAR O VIDEO
if(SAVE_VIDEO):
	out.release()
	print("Video salvo")

# Fecha todas as janelas
cv2.destroyAllWindows()





