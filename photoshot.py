import cv2
import os
import detect_imgs 

# cap = cv2.VideoCapture(0)
class GetImage(object):
    def __init__(self) -> None:
        super().__init__()
        self.image_dir = 'face_bank/imgs'

    def run(self,cap,name):
        done = 0
        _,frame = cap.read()
        img = frame.copy()
        cv2.rectangle(img,(245,165),(395,315),(0,255,0),2)
        cv2.imshow('img show', img)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite(self.image_dir + '/%s.jpg'%name,frame)
            detect_imgs.run(self.image_dir + '/%s.jpg'%name, name)
            done = 1
        return done

# if __name__ == "__main__":
#     name = input('Enter your name: ')
#     photoshot(name)
