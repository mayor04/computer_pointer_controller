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

class Pose_Estimator:
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
        print('Head Pose model load time',start-end)
        return    

    def preprocess(self, frame):
        image = np.copy(frame)
        shape = self.net.inputs[self.input_name].shape

        image = cv2.resize(image, (shape[3], shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        return image
    
    def predict(self, image, crop_face):
        frame = self.preprocess(crop_face)
        
        start = time.time()
        self.request = self.exec_net.start_async(request_id=0, 
            inputs={self.input_name: frame})
        
        status = self.request.wait()
        if(status == 0):
            end = time.time()
            print('Head Pose model inference time',start-end)
            angles = self.preprocess_output(image)
       
        return angles
            
        
    def check_model(self):
        raise NotImplementedError
        
    def preprocess_output(self, frame):
        outputs = self.request.outputs
#         print(outputs)
        
        angles = []      
        angles.append(outputs['angle_y_fc'][0][0])
        angles.append(outputs['angle_p_fc'][0][0])
        angles.append(outputs['angle_r_fc'][0][0])
        
        text = "yaw:{:.2f}  pitch:{:.2f}  roll {:.2f}".format(angles[0],angles[1],angles[2])
        self.draw_points(frame, text)
        
#         angles = np.array(angles)
#         angles.reshape(1,3)
        return angles

    def draw_points(self,frame, text):
        if self.draw == 't':
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 2, 0), 1)
    