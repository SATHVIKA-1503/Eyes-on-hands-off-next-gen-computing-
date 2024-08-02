import cv2
import mediapipe
import pyautogui

face_mesh_landmarks =mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

camer = cv2.VideoCapture(0)
display_w,display_h = pyautogui.size()
while True:
    _,image = camer.read()
    image = cv2.flip(image,1)
    window_h,window_w,_ = image.shape
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks 
    #   print(all_face_landmark_points)
    if all_face_landmark_points:
        one_face_landmark_points = all_face_landmark_points[0].landmark
        for id,landmarks_pont in enumerate(one_face_landmark_points[474:478]):
            x = int(landmarks_pont.x*window_w)
            y = int(landmarks_pont.y*window_h)
            if id==1:
                mouse_x = int(display_w/ window_w * x)
                mouse_y = int(display_h / window_h*y)
                pyautogui.moveTo(mouse_x,mouse_y)

            cv2.circle(image,(x,y),3,(0,0,255))
            #print(landmarks_pont.x,landmarks_pont.y)
        left_eye = [one_face_landmark_points[145],one_face_landmark_points[159]]
        for landmarks_pont in left_eye:
            x = int(landmarks_pont.x*window_w)
            y = int(landmarks_pont.y*window_h)
            cv2.circle(image,(x,y),3,(0,255,255))        
        if(left_eye[0].y - left_eye[1].y<0.01):
            pyautogui.click()
            pyautogui.sleep(2)
            print('mouse clicked')
    cv2.imshow("eye controlled mouse", image)
    key = cv2.waitKey(100)
    if key ==27:
        break
camer.release()
cv2.destroyAllWindows()

