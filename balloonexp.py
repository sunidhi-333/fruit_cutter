import cv2
import time
import random
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Get screen dimensions
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

curr_Frame = 0
prev_Frame = 0
delta_time = 0

next_Time_to_Spawn = 0
Fruit_Size = 30
Spawn_Rate = 1
Score = 0
Lives = 15
Difficulty_level = 1
game_Over = False

slash = np.array([[]], np.int32)
slash_Color = (255, 255, 255)
slash_length = 19

w = h = 0

Fruits = []

def create_balloon():
    x = random.randint(Fruit_Size, screen_width - Fruit_Size)
    y = screen_height + Fruit_Size
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    speed = random.randint(1, 5)
    return {'Curr_position': [x, y], 'Color': color, 'speed': speed}

def draw_balloon(frame, balloon):
    cv2.circle(frame, (int(balloon['Curr_position'][0]), int(balloon['Curr_position'][1])), Fruit_Size, balloon['Color'], -1)
    cv2.line(frame, (int(balloon['Curr_position'][0]), int(balloon['Curr_position'][1] + Fruit_Size)), 
             (int(balloon['Curr_position'][0]), int(balloon['Curr_position'][1] + Fruit_Size + 20)), balloon['Color'], 2)

def Fruit_Movement(Fruits):
    global Lives
    # for fruit in Fruits[:]:
    #     if fruit["Curr_position"][1] < -Fruit_Size:
    #         Lives -= 1
    #         Fruits.remove(fruit)
    for fruit in Fruits:
        if (fruit["Curr_position"][1]) < 20 or (fruit["Curr_position"][0]) > 650:
            Lives = Lives - 1
            # print(Lives)
            print("removed ", fruit)
            Fruits.remove(fruit)


def distance(a, b):
    return int(np.linalg.norm(np.array(a) - np.array(b)))

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("skipping frame")
        continue
    
    h, w, c = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create new balloons
    if random.random() < 0.02:  # Adjust this value to control balloon spawn rate
        Fruits.append(create_balloon())

    # Update and draw balloons
    for balloon in Fruits[:]:
        balloon['Curr_position'][1] -= balloon['speed']
        draw_balloon(img, balloon)

        # Remove balloons that have floated off the screen
        if balloon['Curr_position'][1] < -Fruit_Size:
            Fruits.remove(balloon)
            Lives -= 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:
                    index_pos = [int(lm.x * w), int(lm.y * h)]
                    cv2.circle(img, tuple(index_pos), 18, slash_Color, -1)
                    slash = np.append(slash, index_pos)

                    while len(slash) >= slash_length * 2:  # *2 because each point is two values
                        slash = np.delete(slash, [0, 1])

                    for fruit in Fruits[:]:
                        d = distance(index_pos, fruit["Curr_position"])
                        # cv2.putText(img, str(d), tuple(map(int, fruit["Curr_position"])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 3)
                        if d < Fruit_Size:
                            Score += 100
                            slash_Color = fruit["Color"]
                            Fruits.remove(fruit)

    # if Score > 0 and Score % 1000 == 0:
    #     Difficulty_level = (Score // 1000) + 1
    #     Spawn_Rate = Difficulty_level * 4 / 5
    if Score % 1000 == 0 and Score != 0:
        Difficulty_level = (Score / 1000) + 1
        Difficulty_level = int(Difficulty_level)
        print(Difficulty_level)
        Spawn_Rate = Difficulty_level * 4 / 5

    if Lives <= 0:
        game_Over = True

    slash = slash.reshape((-1, 1, 2))
    cv2.polylines(img, [slash], False, slash_Color, 15, 0)

    curr_Frame = time.time()
    delta_Time = curr_Frame - prev_Frame
    FPS = int(1 / delta_Time)
    cv2.putText(img, f"FPS : {FPS}", (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, f"Score: {Score}", (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, f"Level: {Difficulty_level}", (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, f"Lives remaining : {Lives}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    prev_Frame = curr_Frame

    if not game_Over:
        if time.time() > next_Time_to_Spawn:
            Fruits.append(create_balloon())
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)

        Fruit_Movement(Fruits)
    else:
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("img", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()