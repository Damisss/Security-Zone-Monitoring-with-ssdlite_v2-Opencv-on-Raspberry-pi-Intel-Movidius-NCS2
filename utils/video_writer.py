import cv2

class VideoWriter ():
  @staticmethod
  def write (outputPath, writeVideo, frameQueue, W, H):
    try:
      # initiate fourcc
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      # XVID
      writer = cv2.VideoWriter(outputPath, fourcc, 20, (W, H))
      
      while writeVideo.value or not frameQueue.empty():
        #if frame queue is not empty then get frame and write it. Otherwise do not nothing.
        if not frameQueue.empty():
          frame = frameQueue.get()
          writer.write(frame)
          
      writer.release()
    except Exception as e:
      raise e