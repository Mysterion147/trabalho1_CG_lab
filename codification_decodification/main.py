import cv2

# Carregar uma imagem
image = cv2.imread('image.jpg')

# Verificar se a imagem foi carregada com sucesso
if image is not None:
    # Redimensionar a imagem para metade do tamanho
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Converter a imagem para escala de cinza
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Aplicar detecção de bordas usando Canny
    edges = cv2.Canny(grayscale_image, 100, 200)

    # Aplicar um efeito de desfoque Gaussiano
    blurred_image = cv2.GaussianBlur(resized_image, (15, 15), 0)

    # Exibir a imagem original, imagem em escala de cinza, bordas e imagem desfocada
    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Grayscale Image', grayscale_image)
    cv2.imshow('Edges', edges)
    cv2.imshow('Blurred Image', blurred_image)

    # Esperar até que uma tecla seja pressionada e, em seguida, fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Erro ao carregar a imagem.")

# Para salvar a imagem processada, você pode usar cv2.imwrite():
# cv2.imwrite('output_image.jpg', blurred_image)
