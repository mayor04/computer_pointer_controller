'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class Facial_Landmarks:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, draw):
        self.plugin = None
        self.net = None
        self.input_name = None
        self.output_name = None
        self.exec_net = None
        self.request = None
        self.draw = draw

    def load_model(self,  model_path, device='CPU', extensions=None):
        start = time.time()
        logger = log.getLogger()
        if not os.path.isfile(model_path):
            logger.error("Wrong model xml path specified"+model_path)
            exit(1)
            
        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        
        if extensions and "CPU" in device:
            self.plugin.add_extension(extensions, device)
            
        self.net = IENetwork(model=model_xml, weights=model_bin) 
        self.exec_net = self.plugin.load_network(self.net, device)
        
#         iter_i = iter(self.net.outputs)
#         print(next(iter_i))
#         print(next(iter_i))
#         print(next(iter_i))
        
        self.input_name = next(iter(self.net.inputs))
        self.output_name = next(iter(self.net.outputs))
        
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")

        unsupported_layers = []
        for l in self.net.layers.keys(): 
            if l not in supported_layers:
                unsupported_layers.append(l)
            
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check your extensions")
            exit(1)
        
        end = time.time()
        print('Face landmark model load time',start-end)
        
        return    

    def preprocess(self, frame):
        image = np.copy(frame)
        shape = self.net.inputs[self.input_name].shape

        image = cv2.resize(image, (shape[3], shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        return image
    
    def predict(self, image, crop_face, points):
        frame = self.preprocess(crop_face)
        
        start = time.time()
        self.request = self.exec_net.start_async(request_id=0, 
            inputs={self.input_name: frame})
        
        status = self.request.wait()
        if(status == 0):
            end = time.time()
            print('Facial landmark model inference time',start-end)
            
            h,w,c = crop_face.shape
            left_eye,right_eye,eye_points = self.preprocess_output(image, points, h, w)
       
        return left_eye,right_eye,eye_points
            
        
    def check_model(self):
        raise NotImplementedError
        
    def preprocess_output(self, frame, points, h, w):
        outputs = self.request.outputs[self.output_name][0]
        print(outputs)
        
        left_x = (outputs[0][0][0]*w)+points[0]
        lx_min = int(left_x-20)
        lx_max = int(left_x+20)
        
        left_y = (outputs[1][0][0]*h)+points[1]
        ly_min = int(left_y-10)
        ly_max = int(left_y+10)
        
        right_x = (outputs[2][0][0]*w)+points[0]
        rx_min = int(right_x-20)
        rx_max = int(right_x+20)
        
        right_y = (outputs[3][0][0]*h)+points[1]
        ry_min = int(right_y-10)
        ry_max = int(right_y+10)
        
        eye_points = {'lx':left_x,'ly':left_y,'rx':right_x,'ry':right_y}
        
        cut = np.copy(frame)
        left_eye = cut[ly_min:ly_max,lx_min:lx_max]
        right_eye = cut[ry_min:ry_max,rx_min:rx_max]
        
        print(lx_min,lx_max,points)
        self.draw_points(frame,(lx_min, ly_min),(lx_max, ly_max),
                        (rx_min, ry_min),(rx_max, ry_max))

        return left_eye,right_eye,eye_points

    def draw_points(self,frame,leftA,leftB,rightA,rightB):
        if self.draw == 't':
            cv2.rectangle(frame, leftA, leftB, (40, 100, 230), 4)
            cv2.rectangle(frame, rightA, rightB, (40, 100, 230), 4)
    
    