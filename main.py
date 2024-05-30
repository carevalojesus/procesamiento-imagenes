import cv2
import matplotlib.pyplot as plt

#cargar la imagen  desde una archivo
imagen = cv2.imread('logo-fisi@2x.png')

#verificar que la imegan se cargo correctamente
if imagen is None:
    print('No se pudo cargar la imagen')
else:
    print('Imagen cargada correctamente')

    #converit en escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    #Aplicar un desenfoque gaussiano
    imagen_desenfocada = cv2.GaussianBlur(imagen_gris, (5,5),0)

    #Detectar los bordes con el algoritmo de Canny
    bordes = cv2.Canny(imagen_desenfocada, 100, 200)

    #Mostar la imagenes originales y las procesadas

    plt.figure(figsize=(10,7))

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen original')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(imagen_gris, cmap='gray')
    plt.title('Imagen en escala de grises')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(imagen_desenfocada, cmap='gray')
    plt.title('Imagen desenfocada')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(bordes, cmap='gray')
    plt.title('Imagen con bordes')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
