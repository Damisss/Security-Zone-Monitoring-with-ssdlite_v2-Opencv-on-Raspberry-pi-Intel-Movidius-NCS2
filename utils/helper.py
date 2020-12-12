import cv2

def drawUnauthorizedZone(frame, zones):
  try:
    for zone in zones:
      cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)
  except Exception as e:
    raise e
  
def overlap (rectA, rectB):
  try:
    if rectA[0] > rectB[2] or rectA[2] < rectB[0]:
      return False
    
    if rectA[1] > rectB[3] or rectA[3] < rectB[1]:
      return False
    
    return True
  except Exception as e:
    raise e