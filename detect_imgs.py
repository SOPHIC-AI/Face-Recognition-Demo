"""
This code is used to batch detect images in a folder.
"""
import torch
import numpy as np
from mobilefacenet import MobileFaceNet
import cv2
import os
from vision.ssd.config.fd_config import define_img_size

define_img_size(320)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

device = torch.device("cpu")

mbf = MobileFaceNet()
mbf.load_state_dict(torch.load('backbone.pth', map_location=device))
mbf.eval()
mbf = mbf.to(device)

def feature_normalization(embedding_features):
    
    normalized_features = embedding_features/np.linalg.norm(embedding_features)
    
    return normalized_features


def feature_comparason(feature_1, feature_2):

    cosine = np.dot(feature_1, feature_2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return cosine

def preprocess_img(img):

    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return img

result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
test_device = "cpu"

class_names = [name.strip() for name in open(label_path).readlines()]

model_path = "models/pretrained/version-RFB-320.pth"
# model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500, device=test_device)
face_bank = 'face_bank'
net.load(model_path)


def run(img_dir,name):    
    list_img = os.listdir(img_dir)
    features = []
    for img in list_img:
        img_path = os.path.join(img_dir,img)    
        orig_image = cv2.imread(img_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, _, _ = predictor.predict(image, 1500 / 2, 0.7)
        box = boxes[0, :]
        faces = orig_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        cv2.imwrite(img_path,faces)
        faces = preprocess_img(faces)
        with torch.no_grad():
            faces = faces.to(device)
            embedding_features = mbf(faces)
            embedding_features = embedding_features.squeeze()
            embedding_features = embedding_features.data.cpu().numpy()
            normalized_features = feature_normalization(embedding_features)
            features.append(normalized_features)
    # import ipdb; ipdb.set_trace()
    mean_features = np.mean(features,axis=0)
    with open('face_bank/features'+'/%s.txt'%name,'w') as f:
        np.savetxt(f, mean_features)