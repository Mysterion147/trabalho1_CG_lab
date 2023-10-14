import cv2

# Inicialize o objeto do fundo
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Abra o vídeo de entrada
cap = cv2.VideoCapture('video.mp4')

while True:
    # Capture um quadro do vídeo
    ret, frame = cap.read()

    if not ret:
        break

    # Aplicar a subtração de fundo ao quadro
    fg_mask = bg_subtractor.apply(frame)

    # Exibir o quadro original e a máscara de fundo
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:  # Pressione Esc para sair
        break

cap.release()
cv2.destroyAllWindows()
