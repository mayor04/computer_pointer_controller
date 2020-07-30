'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import time
import cv2
import logging as log
import numpy as np
import math
from openvino.inference_engine import IENetwork, IECore

class Gaze_Estimator:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self):
        self.plugin = None
        self.net = None
        self.input_name = None
        self.output_name = None
        self.exec_net = None
        self.request = None
        
    def load_model(self,  model_path, device='CPU', extensions=None):
        start = time.time()
        if not os.path.isfile(model_path):
            log.error("Wrong model xml path specified"+model_path)
            exit(1)
            
        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        
        if extensions and "CPU" in device:
            self.plugin.add_extension(extensions, device)
            
        self.net = IENetwork(model=model_xml, weights=model_bin) 
        self.exec_net = self.plugin.load_network(self.net, device)
        
#         iter_i = iter(self.net.inputs)
#         print(next(iter_i))
#         print(next(iter_i))
#         print(next(iter_i))
        
        self.output_name = next(iter(self.net.outputs))
        
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")

        unsupported_layers = []
        for l in self.net.layers.keys(): 
            if l not in supported_layers:
                unsupported_layers.append(l)
            
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check your extensions")
            exit(1)
        
        end = time.time()
#         print('Gaze Estimation model load time',start-end)
        return    

    def preprocess(self, frame):
        image = np.copy(frame)

        image = cv2.resize(image, (60, 60))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        return image
    
    def predict(self, image, left_eye, right_eye, angles, eye_points):
        pre_left = self.preprocess(left_eye)
        pre_right = self.preprocess(right_eye)
        
        dicts = {'head_pose_angles':angles, 'left_eye_image':pre_left, 'right_eye_image':pre_right}
        
        start = time.time()
        self.request = self.exec_net.start_async(request_id=0, 
            inputs=dicts)
        
        status = self.request.wait()
        if(status == 0):
            end = time.time()
#             print('Gaze Estimation model inference time',start-end)
            mx,my = self.preprocess_output(image, eye_points, angles[0])
       
        return mx,my
        
    def check_model(self):
        raise NotImplementedError
        
    def preprocess_output(self, frame, eye_points, roll):
        outputs = self.request.outputs[self.output_name][0]
        
        lx = int(eye_points['lx'])
        ly = int(eye_points['ly'])
        rx = int(eye_points['rx'])
        ry = int(eye_points['ry'])
        
        gazeX = int(outputs[0]*200)
        gazeY = int(-outputs[1]*200)
        
        self.draw_points(frame,lx,ly,rx,ry,gazeX,gazeY)
        
        gaze_vector = outputs / cv2.norm(outputs)
        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)
        
        mx = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        my = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue
        
        return mx,my
    
    def draw_points(self,frame,lx,ly,rx,ry,gazeX,gazeY):
        cv2.arrowedLine(frame,(lx,ly),(lx+gazeX,ly+gazeY), (5, 5, 255), 2)
        cv2.arrowedLine(frame,(rx,ry),(rx+gazeX,ry+gazeY), (5, 5, 255), 2)

    
    