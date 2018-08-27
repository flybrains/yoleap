import cv2
import numpy as np
from darkflow.net.build import TFNet
from boxing import gen_boxes
import matplotlib.pyplot as plt

threshold = 0.2

options = {"model": "cfg/fly_yolo.cfg", 
           "pbLoad": "built_graph/fly_yolo.pb",
           "metaLoad": "built_graph/fly_yolo.meta", 
           "threshold": 0.2, 
           "gpu": 0.7}

tfnet = TFNet(options)

# im = cv2.imread("C:/Users/Patrick/Desktop/darkflow/sample_img/frame60.jpg")

# results = tfnet.return_predict(im)

# _, ax = plt.subplots(figsize=(20, 10))
# ax.imshow(gen_boxes(im, results))

# plt.show()

cap = cv2.VideoCapture('./sample_video/sample1_pair.MP4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./sample_video/output.avi',fourcc, 20.0, (416, 416))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        frame = np.asarray(frame)[0:1100,350:1450]
        #frame1 = cv2.resize(frame, (416,416))

        #gray1 = np.expand_dims(gray,2)

        #print(gray.shape)

        results = tfnet.return_predict(frame)

        new_frame = gen_boxes(frame, results)

        # Display the resulting frame
        out.write(new_frame)
        cv2.imshow('frame',new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()