# -*- coding: utf-8 -*-
import cv2


# de: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # Inicializando las dimensiones de la imagen a cambiar de tamaño ademas de tomar el tamaño de la imagen.
    dim = None
    (h, w) = image.shape[:2]
    # si tanto el ancho y el alto De la imagen no tienen nada se retorna la imagen original
    if width is None and height is None:
        return image
    # comprobamos si el ancho es nulo
    if width is None:
        # calculando la relación de la altura para construir las dimensiones
        r = height / float(h)
        dim = (int(w * r), height)
    # si no se cumple la condición la altura es nula
    else:
        # se calcula la relación del ancho para construir las dimensiones correspondientes
        r = width / float(w)
        dim = (width, int(h * r))
    # se cambia el tamaño de la imagen
    return cv2.resize(image, dim, interpolation = inter)