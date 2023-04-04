import cv2

# Carrega o classificador pré-treinado para detectar pedestres
pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Carrega a imagem ou o vídeo em que você deseja detectar os pedestres
imagem = cv2.imread('caminho_da_imagem.jpg')
cap = cv2.VideoCapture('caminho_do_video.mp4')

# Loop para processar cada frame do vídeo
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta os pedestres na imagem ou no frame do vídeo usando o classificador pré-treinado
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenha um retângulo ao redor de cada pedestre detectado
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Mostra o frame do vídeo com os retângulos desenhados
    cv2.imshow('Pedestrians Detected', frame)

    # Pressione a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos e fecha as janelas
cap.release()
cv2.destroyAllWindows()