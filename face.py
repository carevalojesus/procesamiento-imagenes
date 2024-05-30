import cv2
import matplotlib.pyplot as plt

#cargar el clasificador de rostros
cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cargo la imagen
imagen = cv2.imread('caras.jpeg')

#Verificar si la imagen se cargo correctamente
if imagen is None:
    print('No se pudo cargar la imagen')
else:
    print('Imagen cargada correctamente')
    #convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    #Detectar los rostros en la imagen
    caras = cascada_cara.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    #Dibujar los rectangulos sobre los rostros detectados
    for (x, y, w, h) in caras:
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Mostrar la imagen con los rostros detectados
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Rostros detectados')
    plt.axis('off')
    plt.show()