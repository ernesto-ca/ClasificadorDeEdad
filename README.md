# Detector de Edad en tiempo real con python y OpenCV

Este programa realiza detección facial y tratará de predecir el rango de edad del rostro identificado,
por medio de **OpenCV** para la creación de las _DNN_ y _CNN_ necesarias en la clasificación de caras y edades.

## Requisitos:

### 1. Crear un entorno virtual para python ya sea, pipenv, venv u otro a elección del programador(a)

En este caso se aplicó con venv de la siguiente forma:

-   Primero instalar pip siguiendo el siguiente enlace [Cómo instalar PIP para Python](https://tecnonucleous.com/2018/01/28/como-instalar-pip-para-python-en-windows-mac-y-linux/)
-   Después instalar el entorno venv (de manera local) con:

    > python -m venv ./

-   Ahora con el entorno virtual instalado para este proyecto, se debe inicializar de la siguiente manera:

    > source ./venv/Scripts/activate

    al presionar "enter" aparecerá la entre paréntesis (venv) en la terminal, esto significa que ya está activado el entorno virtual.

### 2. Instalar las librerías necesarias:

-   Instalar las opencv, numpy una vez activado el entorno virtual :

    > pip install opencv-python numpy

_OpenCV_: Es una biblioteca de código abierto que incluye varios cientos de algoritmos de visión por computadora.

_NumPy_: Es una biblioteca de Python que proporciona un objeto de matriz multidimensional, varios objetos derivados (como matrices y matrices enmascaradas) y una variedad de rutinas para operaciones rápidas en matrices, entre más.

### 3. Comenzar a usar:

Realmente utilizar el detector es muy sencillo, primero es necesario tener el entorno activado (ver paso 1), una vez activado ejecutar de la siguiente manera en la terminal:

    >python detector_edad.py

Esto iniciará la cámara que se vera en un recuadro o ventana, al mismo tiempo que mostrará el recuadro con la predicción en la parte superior.

En caso de que se necesite un minimo de predicción más alto, ingresar en la linea de comandos el argumento:

    >python detector_edad.py --predecir 0.N

donde N es un numero ya sea de 1, 2 o 3 dígitos para clasificar respecto al mínimo de probabilidad introducido, el predefinido es un 0.5 es decir 50% de probabilidad.

### Créditos.

Los correspondientes autores de partes del código se encuentran dentro de los programas, agradecemos totalmente a los autores originales que gracias a sus códigos se pudo lograr un detector de edad "original" y con fines de estudio en la maravillosa rama de la informática **Inteligencia Artificial**.

#### Notas

-   En caso de querer desactivar el entorno virtual solo con escribir **deactivate** en la terminal será suficiente.
-   Para saber más sobre **venv** visita el siguiente [enlace](https://docs.python.org/3/library/venv.html).
-   Para saber más sobre las librerías utilizadas, por favor visita los enlaces: [OpenCV](https://docs.opencv.org/4.x/index.html) y [numpy](https://numpy.org/doc/stable/).
