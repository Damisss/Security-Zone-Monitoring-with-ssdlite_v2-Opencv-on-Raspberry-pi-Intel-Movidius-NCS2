#Usage
# python detect_realtime.py --topo models/frozen_inference_graph.xml --weights models/frozen_inference_graph.bin 
import cv2
from utils.nn import NetworkLoader
from utils.motion_detector import MotionDetector
from utils.video_writer import VideoWriter
from utils import config, helper
import datetime
from imutils.video import FPS
import argparse
import numpy as np
from multiprocessing import Process, Value, Queue

ap = argparse.ArgumentParser()
#parse arguments
ap.add_argument('-i', '--input', help='Path to input video')
ap.add_argument('-o', '--output', default='output/result.avi',  help='Path to output video')
ap.add_argument('-t', '--topo', required=True, help='Path to intermediate xml file (this file contains the topography of the network)')
ap.add_argument('-w', '--weights', required=True, help='Path to intermediate bin file (this file contains models\' weights ')
ap.add_argument('-c', '--confidence', default=.5, type=float, help='Minimum proba to filter weak detections')
args = vars(ap.parse_args())

class Inference ():
  @staticmethod
  def start():
    try:
      LABELS = config.LABELS
      COLORS = (0, 0, 255)
      #np.random.uniform(0, 255, size=(90, 3))
      UNAUTHORIZED_ZONE = config.UNAUTHORIZED_ZONE
      motionDetector = MotionDetector()
      total = 0
      videoWriterProcess = None
      H, W = None, None
      
      # load the model from disk
      net, excNet = NetworkLoader.load(args['topo'], args['weights'])
      # define input and output blobs
      inputBlob = next(iter(net.inputs))
      outputBlob = next(iter(net.outputs))
      # assign default batch size to 1
      net.batch_size = 1
      #grab number of blob number of chanel, height an width of input 
      n, c, h, w = net.inputs[inputBlob].shape
      
      currentRequestId, nextRequestId = 0, 1
      
      if not args.get('input', False):
        cap = cv2.VideoCapture(1)
      else:
        cap = cv2.VideoCapture(args['input'])
        
      ret, frame = cap.read()
      
      startTime = datetime.datetime.now()
      numFrame = 0
      while cap.isOpened():
        ret, nextFrame = cap.read()
        # stop while loop if is there is no frame available 
        if not ret:
          break
        #stop while loop if q is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          break
        
        if H is None or W is None:
          H, W = nextFrame.shape[:2]
        
        # transform image image to gray scale for motion detection data preprocessing.
        gray = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        if args['output'] and videoWriterProcess is None:
          # initialize multiprocess value
          writeVideo = Value('i', 1)
          #initialize framequeue
          frameQueue = Queue()
          #start process 
          videoWriterProcess = Process(target=VideoWriter.write, args=(args['output'], writeVideo, frameQueue, W, H))
          videoWriterProcess.start()
          
        timeStamp = datetime.datetime.now()
        cv2.putText(frame, timeStamp.strftime('%A %d %B %Y %I:%M:%S%p'), (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, .5, COLORS, 2)
        
        if total > 30:
          motion = motionDetector.detect(gray)
          
          if motion is not None:
            # data preprocessing
            inframe = cv2.resize(nextFrame, (300, 300))
             # change image data shape from HWC to CHW
            inframe = inframe.transpose((2, 0, 1))
            # reshape image
            inframe = inframe.reshape((n, c, h, w))
            # start async 
            excNet.start_async(request_id=nextRequestId, inputs={inputBlob:inframe})
            if excNet.requests[currentRequestId].wait(-1) == 0:
              results = excNet.requests[currentRequestId].outputs[outputBlob]
              # 
              for obj in results[0][0]:
                score = obj[2]
                indx = int(obj[1])
                if indx in LABELS.keys() and LABELS[indx] in ['person']:
                  if  score  >= args['confidence']:
                    #bounding box coordinate
                    xmin = int(obj[3] * W)
                    ymin = int(obj[4] * H)
                    xmax = int(obj[5] * W)
                    ymax = int(obj[6] * H)
                    
                    bbox = (xmin, ymin,xmax, ymax)
                    # draw object bbox only if object bbox overlap with unauthorized zone bbox.
                    for zone in UNAUTHORIZED_ZONE:
                      # do nothing if there is no overlap
                      if not helper.overlap(bbox, zone):
                        continue
                      
                      #else:
                      font = cv2.FONT_HERSHEY_SIMPLEX
                      label = '{} {:.2f}%'.format(LABELS[indx], score*100)
                      y = ymin - 15 if ymin - 15 > 15 else ymin - 15
                      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLORS, 2)
                      cv2.putText(frame, label, (xmin, y), font, .5, COLORS, 2)
                      
                      cv2.putText(frame, 'ALARM: Unauthorized Zone', (20, 20), font, .55, COLORS, 2)
                      
                      #break
        # draw unauthorized zone bbox on frame                
        helper.drawUnauthorizedZone(frame, UNAUTHORIZED_ZONE)
        # update motion detector
        motionDetector.update(gray) 
        # count number of frame for effecient cascade background substration 
        total += 1
        # compute averge frame per second and write it on frame
        numFrame += 1
        elaps = (datetime.datetime.now() - startTime).total_seconds()
        fps = numFrame/elaps
        cv2.putText(frame, 'Average FPS: {:.2f}'.format(fps), (W-200, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,  (255, 0, 0), 2)  
        # put frame on queue
        if  videoWriterProcess is not None:
          frameQueue.put(frame)
        # show the image  
        cv2.imshow('Unauthorized Zone', frame)
        currentRequestId, nextRequestId = nextRequestId, currentRequestId
        frame = nextFrame
        H, W = frame.shape[:2]
      # stop writting process
      if videoWriterProcess is not None:
        writeVideo.value = 0
        videoWriterProcess.join()
        
      cap.release()
      cv2.destroyAllWindows()
    except Exception as e:
      raise e
    
if __name__ == '__main__':
  Inference.start()