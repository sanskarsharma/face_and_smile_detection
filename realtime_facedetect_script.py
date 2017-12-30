import cv2

# start capturing video from primary camera(0)
video = cv2.VideoCapture(1)     # arg can be 0, 1 2 depending on which camera of device to be used

# counter variable 
total_frames_captured = 0 

cascade_classifier_obj = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # loading the harcascade from file into a object
smile_cascade_obj = cv2.CascadeClassifier("haarcascade_smile.xml")

while True:

    total_frames_captured += 1

    checksuccess, frame = video.read()      # read() returns status alongwith frame read from video

    print("Success in reading : " + str(checksuccess))     # just for logging
    print(frame)

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # converting frame to grayscale

    face_list = cascade_classifier_obj.detectMultiScale(grayframe,
    scaleFactor = 1.2,
    minNeighbors = 5)

    result_frame = frame    # otherwise it gives result_img not declared error

    for x,y,w,h in face_list:
        result_frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 5 )
        roi_face = grayframe[y:y+h, x:x+w]
        roi_face_color = frame[y:y+h, x:x+w]
        smile = smile_cascade_obj.detectMultiScale(roi_face, scaleFactor= 1.8, minNeighbors = 20) # these params are found best for dtetecting smile
        for sx,sy,sw,sh in smile:
            cv2.rectangle(roi_face_color, (sx,sy), (sx+sw,sy+sh), (255,0,0), 3)

    cv2.imshow("capturing video ...", result_frame)       # showing frame in a window

    keypress = cv2.waitKey(1)   # waiting for 1 millisecond before moving ahead 
                                # this runs in while loop and generates frames continuously which we see as video in our window

    if keypress==ord('q'):
        break


video.release()
cv2.destroyAllWindows

print("total_frames_captured = "+ str(total_frames_captured))