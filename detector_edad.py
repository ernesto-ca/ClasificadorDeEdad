# -*- coding: utf-8 -*-
#-----------------FUENTES-----------------
#https://github.com/techycs18/age-detection-python-opencv
#https://www.thepythoncode.com/article/predict-age-using-opencv
#-----------------------------------------------
# import the necessary packages

import numpy as np
import argparse
import time
import cv2
from funciones_extras import image_resize

def detectarYPredecir(frame, faceNet, ageNet, minConf=0.5):
    #minConf = confianza minima (estimacion)
    # definimos la lista donde se hubicara el rango de las edades que predecira el detector
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    # Lista de los resultados
    results = []

    # agarrar la dimensiones de la foto y construir a traves de esta una gota para el analisis
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))

    # pasando la gota a través de faceNet
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # comenzando ciclo sobre las detenciones encontradas
    for i in range(0, detections.shape[2]):
        # extraer la los valores de confianza de cada detecion
        confidence = detections[0, 0, i, 2]

        # obteniendo la detección con un valor de confianza valuado mayor que el  mínimo
        if confidence > minConf:
            # obteniendo el recuadro para esa detección
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # consiguiendo el roi
            face = frame[startY:endY, startX:endX]

            # asegurarando el roi con el tamaño suficiente
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0, size=(227, 227),
                                             mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Realizando la prediccion de la edad obteniendo el dato con mayor probabilidad
            ageNet.setInput(faceBlob)
            pred = ageNet.forward()
            i = pred[0].argmax()  # retornando el índice del cubo con la mayor probabilidad obtenida
            age = AGE_BUCKETS[i]
            ageConfidence = pred[0][i]

            # se construye un diccionario que cuenta con la ubicación del cuadro delimitador de la cara junto con la predicción de la edad, posteriormente se actualiza nuestra lista de resultados
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    return results


parser = argparse.ArgumentParser(description='Poner minimo de probabilidad.')
parser.add_argument("-p", "--predecir", type=float, default=0.5, help="Minimo de probabilidad para las predicciones.")
args = vars(parser.parse_args())



# El modelo arquitectonico
# descargado de: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_PROTO = 'deploy_age.prototxt'
# Modelo pre-entrenado
# descargado de: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_MODEL = 'age_net.caffemodel'
# descargado de: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "deploy.prototxt"
# descargado de: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"



# Generando el cnn para el reconocimiento facial
faceNet = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


# Generando el cnn para reconocimiento de edad
ageNet = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# Inicializando la transmisión de video
print("Abriendo la camara...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
contadorF = 0
colorRecuadro = (0, 0, 0)
# aqui se recorre la función de detección de edad que creamos
while True:
    # se cambia el tamaño de cada imagen
    _,frame = vs.read()
    frame = image_resize(frame, width=400)

    #llamando a la función de detección de edad para cada imagen
    results = detectarYPredecir(frame, faceNet, ageNet, minConf=args['predecir'])


    for r in results:
        #dibujando el recuadro alrededor de la cara y muestra la edad que se predijo
        text = "{}: {:2f}%".format(r["age"][0], r["age"][1]*100)
        (startX, startY, endX, endY) = r["loc"]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), colorRecuadro, 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        if colorRecuadro == (255, colorRecuadro[1], colorRecuadro[2]):
            colorRecuadro = (255, colorRecuadro[1]+1, 0)
        elif colorRecuadro == (255, 255, colorRecuadro[2]):
            colorRecuadro =  (255,255, colorRecuadro[2]+1 )
        elif colorRecuadro == (255, 255, 255):
            colorRecuadro = (0, 0, 0)
        else:
            colorRecuadro = (colorRecuadro[0]+1, 0, 0)
    # mostrando la imagen de salida
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # si se presionó la tecla `q`, se terminara el ciclo
    if key == ord("q"):
        break
    if key == ord("p"):
        nombreImagen ='fotos/prediccion{0}.jpg'.format(contadorF)
        print(nombreImagen)
        cv2.imwrite(nombreImagen, frame)
        contadorF += 1

# destruimos lo que ya no se necesita
cv2.destroyAllWindows()
vs.release()
print("Hasta luego!")
