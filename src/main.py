import os
import sys
import cv2
import logging as log
import numpy as np
from argparse import ArgumentParser
from face_detection import Face_Detector
from facial_landmark import Facial_Landmarks
from head_pose import Pose_Estimator
from gaze_estimation import Gaze_Estimator
# from mouse_controller import MouseController
from input_feeder import InputFeeder

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    default = {
        'mfd': 'intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml',
        'mpe': 'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml',
        'mfl': 'intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
        'mge': 'intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml',
        'i':'demo.mp4'
    }
    
    parser.add_argument("-m_fd", "--model_fd", required=False,default = default['mfd'],type=str,
                        help="Path to a face detection model file.")
    
    parser.add_argument("-m_pe", "--model_pe", required=False, default = default['mpe'], type=str,
                        help="Path to a head pose estimation model file.")
    
    parser.add_argument("-m_fl", "--model_fl", required=False, default = default['mfl'], type=str,
                        help="Path to a facial landmark detection file.")
    
    parser.add_argument("-m_ge", "--model_ge", required=False,default = default['mge'], type=str,
                        help="Path to a gaze estimation model file.")
    
    parser.add_argument("-i", "--input", required=False, type=str, default = default['i'],
                        help="Path to image or video file or type 'cam'")
    
    parser.add_argument("-sv", "--save", required=False, type=str,default='t',
                        help="This saves the video file to current directory"
                              "input 'f' not to save")
    
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-dl", "--draw_lines", type=str, default='t',
                        help="Boolean value for drawing bounding boxes and lines"
                        "(true by default) input 'f' not to show")
    parser.add_argument("-pt_fd", "--thres_fd", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def run_controller(args):
    logger = log.getLogger()
    feeder = None
    
    if args.input == "cam":
        feeder = InputFeeder("cam")
        
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'): 
        if not os.path.isfile(args.input):
            logger.error("Unable to find specified video file")
            exit(1)
        print('image')    
        feeder = InputFeeder("image",args.input,args.save)
        
    else:
        if not os.path.isfile(args.input):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder("video",args.input,args.save)
        
    feeder.load_data()
    
#     mc = MouseController('medium','fast')
    
    model_face = Face_Detector(args.draw_lines)
    model_face.load_model(args.model_fd,args.device,CPU_EXTENSION)
    
    model_pose = Pose_Estimator(args.draw_lines)
    model_pose.load_model(args.model_pe,args.device,CPU_EXTENSION)
    
    model_landmark = Facial_Landmarks(args.draw_lines)
    model_landmark.load_model(args.model_fl,args.device,CPU_EXTENSION)
    
    model_gaze = Gaze_Estimator(args.draw_lines)
    model_gaze.load_model(args.model_ge,args.device,CPU_EXTENSION)
    
    frame_count = 0
    for b,frame in feeder.next_batch():
        frame_count+=1
        preview = np.copy(frame)
        crop_face,face_count,points = model_face.predict(preview,args.thres_fd)
        
        key_pressed = cv2.waitKey(30)
        if(face_count == 0):
            if(b or key_pressed == 27):
                break;
                
            print('no face is detected')
            feeder.save_file(preview)
            continue

        angles = model_pose.predict(preview,crop_face)
        left_eye,right_eye,eye_points = model_landmark.predict(preview,crop_face,points)

        mx,my = model_gaze.predict(preview,left_eye,right_eye,angles,eye_points)
        feeder.save_file(preview)
        
        if key_pressed == 27:
            break
        
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
#             mc.move(mx,my)
            
    feeder.close()
    cv2.destroyAllWindows()
    
    
def run_test(args):
    mc = MouseController('medium','fast')
    
    model_face = Face_Detector()
    model_face.load_model(args.model_fd,args.device,CPU_EXTENSION)
    
    model_pose = Pose_Estimator()
    model_pose.load_model(args.model_pe,args.device,CPU_EXTENSION)
    
    model_landmark = Facial_Landmarks()
    model_landmark.load_model(args.model_fl,args.device,CPU_EXTENSION)
    
    model_gaze = Gaze_Estimator()
    model_gaze.load_model(args.model_ge,args.device,CPU_EXTENSION)
    
    frame = cv2.imread(args.input)
    crop_face,face_count,points = model_face.predict(frame,args.thres_fd)
    
    if(face_count == 0):
        print('no face is detected')
        
    angles = model_pose.predict(frame,crop_face)
    left_eye,right_eye,eye_points = model_landmark.predict(frame,crop_face,points)
    
    mx,my = model_gaze.predict(frame,left_eye,right_eye,angles,eye_points)
    cv2.imwrite('images/ne.jpg',frame)
    
    mc.move(mx,my)
    
def main():
    args = build_argparser().parse_args()
    run_controller(args)



if __name__ == '__main__':
    main()
    
    
'''
-m_fd intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml

-m_pe intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml

-m_fl intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml

-m_ge intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml

python main.py -i faceA.jpg -m_fd intel/
face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml -m_pe intel/head-p
ose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -m_fl intel/landmarks-regr
ession-retail-0009/FP16/landmarks-regression-retail-0009.xml -m_ge intel/gaze-estimation-ad
as-0002/FP16/gaze-estimation-adas-0002.xml
'''    