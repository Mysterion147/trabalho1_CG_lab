import cv2
import numpy as np

bg_subtractor = cv2.creeateBackgroundSubtractorMOG2()

cap = cv2.VideoCapture('video.mp4')

# variáveis para rastreamento de objetos
tracking = False
track_window = None
roi_hist = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # aplica a subtração de fundo
    fg_mask = bg_subtractor.apply(frame)

    # realiza a limiarização na máscara para obter regiões de destaque
    _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    # encontra contornos nos objetos em movimento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # desenha caixas delimitadoras ao redor dos objetos em movimento
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # ggnora contornos mto pequenos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if not tracking:
                # inicia o rastreamento
                track_window = (x, y, w, h)
                roi = frame[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                tracking = True

    if tracking:
        roi_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # conversao para HSV para montar histograma
        dst = cv2.calcBackProject([roi_hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)) # controle do rastreamento
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # exibe o video com as caixas
    cv2.imshow('tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # espera pelo ESC pra fechar
        break

cap.release()
cv2.destroyAllWindows()
