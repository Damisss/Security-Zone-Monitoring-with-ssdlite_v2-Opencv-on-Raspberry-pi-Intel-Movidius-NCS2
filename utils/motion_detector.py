import numpy as np 
import imutils 
import cv2

class MotionDetector ():
  '''
  this class detect motion using background substraction
  '''
  def __init__(self, accumWeight=.5):
    # accumulate weight factor
    self.accumWeight = accumWeight
    # initialize background model
    self.bg = None
    
  def update (self, image):
    try:
      #check if background is none (that means the update function has not been called yet) then store 
      if self.bg is None:
        self.bg = image.copy().astype('float')
        return 
      
      # update the background by accumulating the weighted average.
      cv2.accumulateWeighted = (image, self.bg, self.accumWeight)
    except Exception as e:
      raise e
    
  def detect (self, image, minThresh= 25):
    try:
      # compute the absolute difference between background model and and the image 
      delta = cv2.absdiff(self.bg.astype('uint8'), image)
      # perform image thresholding
      T, thresh = cv2.threshold(delta, minThresh, 255, cv2.THRESH_BINARY)
      
      # perform some image preprocessing
      thresh = cv2.erode(thresh, None, iterations=2)
      thresh = cv2.dilate(thresh, None, iterations=2)
      # find countours 
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      
      if len(cnts) == 0:
        return 
     
      return True
         
    except Exception as e:
      raise e
  