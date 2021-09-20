import os

import time
import torch
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
from mobilefacenet import MobileFaceNet
# onnx runtime
import onnxruntime as ort


class FaceRecognition(object):

    def __init__(self):
        super().__init__()
        self.onnx_path = "models/onnx/version-RFB-320.onnx"
        self.predictor = onnx.load(self.onnx_path)
        onnx.checker.check_model(self.predictor)
        onnx.helper.printable_graph(self.predictor.graph)
        self.predictor = backend.prepare(self.predictor, device="CPU")  # default CPU
        self.ids=[]
        self.unknowns = []
        self.threshold = 0.7
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.checkpoint = 'backbone.pth'
        self.device = 'cpu'
        self.model = MobileFaceNet()
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

    def feature_normalization(self,embedding_features):
        
        normalized_features = embedding_features/np.linalg.norm(embedding_features)
        
        return normalized_features


    def feature_comparason(self,feature_1, feature_2):

        cosine = np.dot(feature_1, feature_2)
        cosine = np.clip(cosine, -1.0, 1.0)

        return cosine

    def preprocess_img(self,img):

        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)

        return img

    def detect_preprocess(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 240))
        image_mean = np.array([127, 127, 127])
        img = (img - image_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return img

    def predict(self,width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def recognition(self,face,boxes,ids =[], unknowns=[]):
        face = self.preprocess_img(face)
        face_dir = "face_bank/features"
        list_face = os.listdir(face_dir)
        with torch.no_grad():
            face = face.to(self.device)
            embedding_features = self.model(face)
            embedding_features = embedding_features.squeeze()
            embedding_features = embedding_features.data.cpu().numpy()
            normalized_features = self.feature_normalization(embedding_features)
            count = 0
            if list_face == []:
                unknowns.append([int(boxes[0]),int(boxes[1])])
            else:
                for face in list_face:
                    face_path = os.path.join(face_dir,face)
                    normalized_features_anchor = np.loadtxt(face_path).reshape(512,)
                    score = self.feature_comparason(normalized_features_anchor,normalized_features)
                    if score > 0.4:
                        ids.append([face,score,int(boxes[0]),int(boxes[1]),int(boxes[2]),int(boxes[3])])
                        break
                    count += 1
                    if count == len(list_face):
                        unknowns.append([int(boxes[0]),int(boxes[1])])
        return ids, unknowns

    def run(self,cap):
        ids=[]
        unknowns=[]
        _, orig_image = cap.read()
        img = self.detect_preprocess(orig_image)
        time_time = time.time()
        confidences, boxes = self.ort_session.run(None, {self.input_name: img})
        boxes, _, _ = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, self.threshold)
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
            faces = orig_image[box[1]:box[3],box[0]:box[2]]
            ids, unknowns = self.recognition(faces,box,ids=ids,unknowns=unknowns)
                    
        for id in ids:
            score = int(id[1] * 100)
            cv2.rectangle(orig_image, (id[2], id[3]), (id[4], id[5]), (0, 255, 0), 4)
            cv2.putText(orig_image,str(id[0][:-4])+' '+str(score)+'%',(int(id[2]),int(id[3])-15),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        for unknown in unknowns:
            cv2.putText(orig_image,'unknown',(unknown[0],unknown[1]-15),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        print("fps:{}".format(1/(time.time() - time_time)))
        return orig_image

        # cap.release()
        # cv2.destroyAllWindows()
        # print("sum:{}".format(sum))

        # return orig_image

    
if __name__ == "__main__":
    recognition = FaceRecognition()
    recognition.run()
