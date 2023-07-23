import numpy as np
import cv2
import blobs

frame_index=0
cap = cv2.VideoCapture(r'blob_track\og_vid.mp4')
_, firstframe = cap.read()
first_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)
__blobs = []
while True:
    _, frame = cap.read()
    frame_index += 1
    #uncommenet below three line if you want to update bg every 1 sec/30frames
    # if frame_index % 30 == 0:
    #     print('Update BG')
    #     first_gray = gray_frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)

    
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # img_ero = cv2.erode(difference, (3,3), iterations=1)
    # img_dial = cv2.dilate(img_ero, (3,3), iterations=1)
    close_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_dial = cv2.morphologyEx(difference, cv2.MORPH_CLOSE, close_kernal)

    # erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 4))
	# frame = cv2.erode(frame, erosion_kernel, iterations=1)
    erosion_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,4))
    img_dial = cv2.erode(img_dial, erosion_kernal, iterations=1)

    cont, hirer = cv2.findContours(img_dial, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    array_area = []
    # print(str(len(cont)))
    # print(cont)
    for cn in cont:
        blob_area = cv2.contourArea(cn)
        if (blob_area < 5):
            continue
        array_area.append(blob_area)

        rect = cv2.minAreaRect(cn)
        box = cv2.boxPoints(rect)
        boxfit = np.int0(box)

        x, y, w, h = cv2.boundingRect(cn)

        # density = blob_area / bbox.area()
        # blob = Blob(blob_area, bbox, density, [0, 0], i, frame_index, boxfit)
        # _blobs.append(blob)

        #Bounding BOX
        left = x
        right = x + w
        top = y
        bottom = y + h
        center_x = x + w / 2
        center_y = y + h / 2
        width = w
        height = h
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

        bbox = area = (right - left) * (bottom - top)
        density = blob_area / bbox

        #implementing BLOB
        #Blob(blob_area, bbox, density, [0, 0], i, self.frame_index, boxfit)
        b_area = blob_area
        bounding_box = bbox
        __velocity = [0,0]
        blob_index = i
        __track_id = 0
        __age = 0

        __blobs.append([__track_id,frame_index,left, top] )
        
        i += 1
    # cv2.drawContours(frame, cont, -1, (0,255,0), 3)
    # cv2.imshow('ff', img_ero)
    # first_gray = gray_frame
    cv2.imshow('frm', img_dial)
    cv2.imshow('diff', frame)
    # cv2.imwrite(conframe, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    print(__blobs)

cap.release()
cv2.destroyAllWindows()
print(cont)