import cv2
import mediapipe as mp
import time
import math as math
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__   =  mode
        self.__maxHands__   =  maxHands
        self.__detectionCon__   =   detectionCon
        self.__trackCon__   =   trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition( self, frame, handNo=0, draw=True):
        xList =[]
        yList =[]
        bbox = []
        self.lmsList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                               (0, 255 , 0) , 2)
        return self.lmsList, bbox

    def findFingerUp(self):
         fingers=[]
         if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
              fingers.append(1)
         else:
              fingers.append(0)
         for id in range(1, 5):            
              if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
                   fingers.append(1)
              else:
                   fingers.append(0)
         return fingers

    def findDistance(self, p1, p2, frame, draw= True, r=15, t=3):
        x1 , y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx , cy = (x1+x2)//2 , (y1 + y2)//2
        if draw:
              cv2.line(frame,(x1, y1),(x2,y2) ,(255,0,255), t)
              cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
              cv2.circle(frame,(x2,y2),r, (255,0,0),cv2.FILLED)
              cv2.circle(frame,(cx,cy), r,(0,0.255),cv2.FILLED)
        len= math.hypot(x2-x1,y2-y1)
        return len, frame , [x1, y1, x2, y2, cx, cy]

def resize_proportional_stretch(image, target_size=(64, 64)):
    """
    Redimensiona la imagen manteniendo la proporción y estirando para llenar el espacio.
    Este método mantiene la relación de aspecto en una dirección y estira en la otra.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calcular ratios de escala
    scale_w = target_w / w
    scale_h = target_h / h
    
    # Usar la escala mayor para que la imagen llene todo el espacio objetivo
    scale = max(scale_w, scale_h)
    
    # Redimensionar con la escala calculada
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Crear imagen de salida del tamaño objetivo
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calcular posición para centrar la imagen redimensionada
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    end_x = start_x + new_w
    end_y = start_y + new_h
    
    # Asegurar que no exceda los límites
    if start_x < 0:
        resized = resized[:, -start_x:]
        start_x = 0
    if start_y < 0:
        resized = resized[-start_y:, :]
        start_y = 0
    if end_x > target_w:
        resized = resized[:, :target_w-start_x]
        end_x = target_w
    if end_y > target_h:
        resized = resized[:target_h-start_y, :]
        end_y = target_h
    
    # Colocar la imagen redimensionada en el centro
    result[start_y:end_y, start_x:end_x] = resized
    
    return result

def resize_intelligent_crop(image, target_size=(64, 64)):
    """
    Redimensiona la imagen manteniendo la proporción y recortando el exceso.
    Este método mantiene completamente la relación de aspecto original.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calcular ratios de escala
    scale_w = target_w / w
    scale_h = target_h / h
    
    # Usar la escala menor para que toda la imagen quepa
    scale = min(scale_w, scale_h)
    
    # Redimensionar con la escala calculada
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Crear imagen de salida del tamaño objetivo
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calcular posición para centrar la imagen redimensionada
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    
    # Colocar la imagen redimensionada en el centro
    result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return result

def main():
    ctime=0
    ptime=0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()    # Cargar el modelo entrenado y las etiquetas (29 clases según el dataset)
    model = load_model('asl_alphabet_model.h5')
    
    # Lista de 29 clases del dataset ASL Alphabet en el orden correcto
    # A-Z (0-25), del (26), nothing (27), space (28)
    labels = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']
    print(f"Número de etiquetas: {len(labels)}")  # Para verificar que son 29
    print(f"Forma de salida del modelo: {model.output_shape}")
    print(f"Etiquetas: {labels}")
    print("Usando redimensionamiento 'stretch proporcional' para mejor precisión")
    print("Presiona ESC para salir")
    print("=" * 60)
    
    # Umbral de confianza (puedes ajustarlo)
    confidence_threshold = 70  # Reducido de 80 para ver más predicciones

    while True:
        ret, frame = cap.read()
        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame)

        # Si hay mano detectada y bounding box válido
        if lmsList and bbox and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            # Ajustar límites para evitar errores
            xmin = max(0, xmin - 20)
            ymin = max(0, ymin - 20)
            xmax = min(frame.shape[1], xmax + 20)
            ymax = min(frame.shape[0], ymax + 20)            
            hand_img = frame[ymin:ymax, xmin:xmax]
            
            if hand_img.size > 0 and (ymax-ymin) > 0 and (xmax-xmin) > 0:
                try:                    # Preprocesar la imagen exactamente como en el entrenamiento
                    hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    # Usar redimensionamiento proporcional stretch (mejor para reconocimiento)
                    # Puedes cambiar a resize_intelligent_crop si prefieres mantener aspecto exacto
                    hand_img_resized = resize_proportional_stretch(hand_img_rgb, (64, 64))
                    # hand_img_resized = resize_intelligent_crop(hand_img_rgb, (64, 64))  # Alternativa
                    # El modelo tiene Rescaling(1/255.) integrado, no normalizar aquí
                    hand_img_batch = np.expand_dims(hand_img_resized.astype(np.float32), axis=0)
                    prediction = model.predict(hand_img_batch, verbose=0)
                    # Aplicar softmax para obtener probabilidades normalizadas
                    prediction_probs = tf.nn.softmax(prediction).numpy()
                    predicted_index = np.argmax(prediction_probs)
                    confidence = np.max(prediction_probs) * 100
                      # Mostrar información de debug
                    print(f"Predicción: {predicted_index} ({labels[predicted_index]}), Confianza: {confidence:.2f}%")
                    
                    # Solo mostrar predicción si la confianza es superior al umbral
                    if confidence > confidence_threshold:
                        if predicted_index < len(labels):
                            letter = labels[predicted_index]
                            cv2.putText(frame, f'Seña: {letter} ({confidence:.1f}%)', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        else:
                            cv2.putText(frame, f'Error: Índice {predicted_index} fuera de rango', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.putText(frame, f'Confianza baja: {confidence:.1f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                except Exception as e:
                    print("Error en la predicción:", e)

        ctime = time.time()
        fps =1/(ctime-ptime) if (ctime-ptime) > 0 else 0
        ptime = ctime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()