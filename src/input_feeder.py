"""
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
"""
import cv2
from numpy import ndarray


class InputFeeder:
    def __init__(self, input_type, input_file=None, save=True):
        self.save = save
        
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.file = input_file
               
    def load_data(self):
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.file)
            self.init_save()
            
        elif self.input_type == 'cam':
            self.save = False
            self.cap = cv2.VideoCapture(0)
            
        else:
            self.cap = cv2.imread(self.file)

    def init_save(self):
        width = int(self.cap.get(3))
        height = int(self.cap.get(4))
        
        if self.save:
            self.out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    def save_file(self,frame): 
        if self.save:
            print('saving',self.input_type)
            if self.input_type == 'image':
                cv2.imwrite('new.jpg',frame)
                return
            
            self.out.write(frame)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        
        if self.input_type == 'image':
            yield True,self.cap
        else:
            while True:
                flag, frame = self.cap.read()
                if not flag:
                    return

                yield 0,frame    
        return        
           

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type == 'image':
            self.cap.release()
