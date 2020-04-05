import cv2
import numpy as np
import matplotlib.pyplot as plt



#Edge Detection 색이 바귈때의 gradient가 바뀌는 것을 감지를 이용한다.
def Canny(image):
  
  #Step1 conver to gray image(1 channel) from RGB image(3 channel)
  gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  #Step2 Reduce Noise noise의 위치만 줄이기 어려우므로 noise를 주변의 색과 비슷하게 만들면 된다.
  blur = cv2.GaussianBlur(gray,(5,5), 0)
  #Step3 gradient calculate
  canny = cv2.Canny(blur,50,150)
  
  return canny

def average_slope_intercept(image,lines):
  left_fit = []
  right_fit = []
  
  for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      parameters = np.polyfit((x1,x2),(y1,y2),1)
      slope = parameters[0]
      intercept = parameters[1]
      if slope < 0:
          left_fit.append((slope,intercept))
      else:
          right_fit.append((slope,intercept))
      
  left_fit_average = np.average(left_fit, axis=0)
  right_fit_average = np.average(right_fit,axis=0)
  left_line = make_coordinages(image,left_fit_average)
  right_line = make_coordinages(image,right_fit_average)
  return(np.array([left_line,right_line]))



def make_coordinages(image,line_parameters):
  slope, intercept = line_parameters
  print(image.shape)
  y1 = image.shape[0]
  y2 = int(y1*(3/5))
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  return np.array([x1,y1,x2,y2])


def display_lines(image,lines):
  line_image = np.zeros_like(image)
  if lines is not None:
      for x1,y1,x2,y2 in lines:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
  return line_image

#도로위의 차선만 표시하게 하기 위한 함수
def region_of_interset(image):
  height = image.shape[0]  #image row
  polygons= np.array([ 
      [(200,height),(1100,height),(550,250)]
      ])
  mask = np.zeros_like(image)
  #이미지의 polygons부분만 255로 채우고 나머지는 0 으로 채운다. 즉 검정 부분은 binary상에서 0 하얀 부분은 1로 표시하기 위해서다.
  cv2.fillPoly(mask, polygons, 255)     
  #bit연산을 통해 둘다 1 인곳만 하얗게 표시한다. image = canny 이고 mask는 도로 위기 때문에 차선만 표시됌
  masked_image = cv2.bitwise_and(image,mask) 

  return masked_image


# # Read Image(array)
# image = cv2.imread('test_image.jpg')

# #변경을해도 원래의 image는 바뀌지 않는다.
# lane_image = np.copy(image)
# canny = Canny(lane_image)

# #mask = region_of_interset(canny)
# cropped_image = region_of_interset(canny)

# #1degree to radians = 파이/180, threshold 몇개의 점을 지나는지에 대한 점에 최소값
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) 
# averaged_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image,averaged_lines)

# #기존의 사진과 검출한 사진을 합쳐서 보여줌
# combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
# #Show Image

# cv2.imshow('result',combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
  _, frame = cap.read()
  canny = Canny(frame)

  #mask = region_of_interset(canny)
  cropped_image = region_of_interset(canny)

  #1degree to radians = 파이/180, threshold 몇개의 점을 지나는지에 대한 점에 최소값
  lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) 
  averaged_lines = average_slope_intercept(frame,lines)
  line_image = display_lines(frame,averaged_lines)

  #기존의 사진과 검출한 사진을 합쳐서 보여줌
  combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
  #Show Image

  cv2.imshow('result',combo_image)
  if cv2.waitKey(1) & 0xFF== ord(''):
    break

cap.release()
cv2.destroyAllWindows()




