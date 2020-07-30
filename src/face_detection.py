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

class Face_Detector:
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
        
        self.input_name = next(iter(self.net.inputs))
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
#         print('Face detection model load time',start-end)
        return    

    def preprocess(self, frame):
        image = np.copy(frame)
        shape = self.net.inputs[self.input_name].shape

        image = cv2.resize(image, (shape[3], shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        return image
    
    def predict(self, image, thres):
        frame = self.preprocess(image)
        
        start = time.time()
        self.request = self.exec_net.start_async(request_id=0, 
            inputs={self.input_name: frame})
        
        status = self.request.wait()
        if(status == 0):
            end = time.time()
#             print('Face detection model inference time',start-end)
            crop_face,face_count,points = self.preprocess_output(image, thres)
       
        return crop_face,face_count,points
            
        
    def check_model(self):
        raise NotImplementedError
        
    def preprocess_output(self, frame, thres):
        outputs = self.request.outputs[self.output_name]
        h,w,c = frame.shape
        
        faces = 0
        crop_face = 0
        points = []
    
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= thres and box[1] == 1:
                
                if(faces > 1):
                    log.error("found more than one face this"
                          "may affect performance of the model")
                    return crop_face,faces
                
                xmin = int(box[3] * w)
                ymin = int(box[4] * h)
                xmax = int(box[5] * w)
                ymax = int(box[6] * h)
                faces+=1

                points.append(xmin)
                points.append(ymin)
                
                crop_face = frame[ymin:ymax,xmin:xmax]
                self.draw_points(frame,(xmin, ymin),(xmax, ymax))
                
       
        return crop_face,faces,points

    def draw_points(self,frame, f, s):
        cv2.rectangle(frame, f, s, (130, 20, 25), 4)
    
    