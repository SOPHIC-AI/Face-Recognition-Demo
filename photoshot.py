import cv2
import os
import detect_imgs 

class GetImage(object):
    def __init__(self) -> None:
        super().__init__()
        self.image_dir = 'face_bank/imgs'
        self.i = 0

    def run(self,cap,name):
        if not os.path.exists(os.path.join(self.image_dir,name)):
            os.mkdir(os.path.join(self.image_dir,name))
        done = 0
        _,frame = cap.read()
        img = frame.copy()
        cv2.putText(img,'%d'%self.i,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
        cv2.rectangle(img,(245,165),(395,315),(0,255,0),2)
        cv2.imshow('img show', img)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            self.i += 1
            cv2.imwrite(self.image_dir + '/%s'%name + '/%d.jpg'%self.i,frame)
        if self.i == 5:
            detect_imgs.run(self.image_dir + '/%s'%name, name)
            done = 1
        return done