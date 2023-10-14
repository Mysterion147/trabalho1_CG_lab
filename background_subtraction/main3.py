import cv2
import numpy as np
import time

bg_subtractor = cv2.createBackgroundSubtractorMOG2()
contours = []

cap = cv2.VideoCapture('video.mp4')

# objetos rastreados
tracked_objects = []

# inicializa o timer
start_time = time.time()

while True:
    ret, frame = cap.read()  # quadro a quadro

    if not ret:
        break  # sai do loop quando acabam os quadros a serem processados

    # aplicar a subtração de fundo ao quadro
    fg_mask = bg_subtractor.apply(frame)

    # Tratamento da imagem
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)  # Desfoque Gaussiano
    _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)  # Binarização

    # encontrar contornos na máscara de fundo
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_objects = 0

    # crie uma imagem em branco para desenhar os retângulos
    overlay = np.zeros_like(frame)

    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Filtre contornos pequenos
            num_objects += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # rastreamento de objetos
            if len(tracked_objects) == 0:
                tracked_objects.append((x, y, x + w, y + h))
            else:
                for obj in tracked_objects:
                    obj_x, obj_y, obj_x2, obj_y2 = obj
                    if x > obj_x and y > obj_y and x + w < obj_x2 and y + h < obj_y2:
                        break
                else:
                    tracked_objects.append((x, y, x + w, y + h))

    # exibir o número de objetos em movimento na tela
    cv2.putText(frame, f'Objetos em Movimento: {num_objects}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # exibir o contador de tempo
    current_time = time.time() - start_time
    cv2.putText(frame, f'Tempo: {int(current_time)} s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # mesclar a imagem original com a imagem de sobreposição
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.imshow('Detecção de Movimento', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break  # fecha com ESC

cap.release()
cv2.destroyAllWindows()