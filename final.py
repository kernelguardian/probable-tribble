import torch
import cv2
import posenet
import time
import pandas as pd
import numpy as np
from torch import nn

if torch.cuda.is_available():
    train_on_gpu = True
else:
    train_on_gpu = False

class NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32,16)
        self.fc4 = torch.nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x
    
input_dim = 51
output_dim  = 3
batch_size = 64
c_model = NN(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(c_model.parameters(), lr=0.001)

if train_on_gpu:
    c_model.load_state_dict(torch.load('output/posenet.pt'))
else:
    c_model.load_state_dict(torch.load('output/posenet.pt', map_location=torch.device('cpu')))
c_model.eval()
if train_on_gpu:
    c_model = c_model.cuda()

org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
font = cv2.FONT_HERSHEY_SIMPLEX  
thickness = 2


model = posenet.load_model(101)
if train_on_gpu:
    model = model.cuda()
output_stride = model.output_stride

cap = cv2.VideoCapture(0)

start = time.time()
frame_count = 0
rows = [200, 200]
while True:
    arr = []
    input_image, display_image, output_scale = posenet.read_cap(
        cap, scale_factor=0.7125, output_stride=output_stride)

    with torch.no_grad():
        if train_on_gpu:
            input_image = torch.Tensor(input_image).cuda()
        else:
            input_image = torch.Tensor(input_image)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=1,
            min_pose_score=0.15)
    keypoint_coords *= output_scale
    
    coordinates = keypoint_coords.squeeze(0)
    scores = keypoint_scores.squeeze(0)
    row = []
    for i in range(0, 17):
        row.append(scores[i])
        row.append(coordinates[i][1])
        row.append(coordinates[i][0])
    row = torch.FloatTensor(row)
    row = row.reshape(1,51)
    
    if train_on_gpu:
        row = row.cuda()
        
    _, pred = torch.max(c_model(row), 1)
    if pred.item() == 0:
        aa = 'standing'
    elif pred.item() == 1:
        aa = 'falling'
    elif pred.item() == 2:
        aa = 'sitting'
#     print(pred.item(),"\n")
    
    
    
    overlay_image = posenet.draw_skel_and_kp(
        display_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.15, min_part_score=0.1)
    
    overlay_image = cv2.putText(overlay_image, aa, org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow('posenet', overlay_image)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Average FPS: ', frame_count / (time.time() - start))

# {'standing':0, 'falling':1, 'sitting':2}