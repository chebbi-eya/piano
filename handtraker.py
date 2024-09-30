import cv2
import mediapipe as mp
import pygame

# Initialiser Pygame pour jouer des sons
pygame.mixer.init()


sounds = {
    (0, 0): pygame.mixer.Sound('a.mp3'),  
    (0, 1): pygame.mixer.Sound('b.mp3'),  
    (0, 2): pygame.mixer.Sound('c.mp3'),  
    (0, 3): pygame.mixer.Sound('d.mp3'),  
    (0, 4): pygame.mixer.Sound('e.mp3'),  
    (1, 0): pygame.mixer.Sound('f.mp3'),   
    (1, 1): pygame.mixer.Sound('g.mp3'),  
    (1, 2): pygame.mixer.Sound('h.mp3'),   
    (1, 3): pygame.mixer.Sound('i.mp3'),   
    (1, 4): pygame.mixer.Sound('j.mp3')    
}

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]  # Indices

prev_finger_states = {0: [0] * 5, 1: [0] * 5} 


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_index, handLms in enumerate(results.multi_hand_landmarks):
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                if id == 8:
                 cv2.circle(img, (cx, cy), 17, (255, 175, 100), cv2.FILLED)
                 cv2.putText(img, "B", (cx -5 , cy-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if id == 4:
                 cv2.circle(img, (cx, cy), 17, (255, 200, 150), cv2.FILLED)
                 cv2.putText(img, "A", (cx -5 , cy-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if id == 12:
                 cv2.circle(img, (cx, cy), 17, (255, 150, 50), cv2.FILLED)
                 cv2.putText(img, "C", (cx -5 , cy-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if id == 16:
                 cv2.circle(img, (cx, cy), 17, (255, 125, 0), cv2.FILLED)
                 cv2.putText(img, "D", (cx -5 , cy-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if id == 20:
                 cv2.circle(img, (cx, cy), 17, (255, 100, 0), cv2.FILLED)
                 cv2.putText(img, "E", (cx -5 , cy-5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if len(lmList) == 21:
                fingers = []

                # Détecter le pouce (différent pour main gauche et droite)
                if hand_index == 0:  # Main droite
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # Main gauche
                    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Autres doigts
                for tip in range(1, 5):
                    if lmList[tipIds[tip]][2] < lmList[tipIds[tip] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # Vérifier si un doigt vient d'être baissé pour jouer un son
                for i, (prev, curr) in enumerate(zip(prev_finger_states[hand_index], fingers)):
                    if prev == 1 and curr == 0: 
                        sounds[(hand_index, i)].play() 

                # Mettre à jour l'état précédent pour cette main
                prev_finger_states[hand_index] = fingers

    cv2.imshow('hand tracker', img)

    if cv2.waitKey(5) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
