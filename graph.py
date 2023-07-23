import numpy as np
import cv2
import blobs
import os
import time
import matplotlib.pyplot as plt


def show_video(path_video):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(path_video)

    # Check if camera opened
    if (cap.isOpened()== False): 
        print("Error while opening video data ")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break
        time.sleep(0.05)
    
    # When everything done, release 
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def high_pass(im,value):
  for i in range(len(im)):
    for j in range(len(im[i])):
      if(im[i][j]<value):
        im[i][j]=0
      else:
        im[i][j]=255
  return(im)


def get_background_video(loc):
  cap = cv2.VideoCapture(loc)
  i=0
  rec, background = cap.read()
  r, g, b = background[:,:,0], background[:,:,1], background[:,:,2]
  background = 0.2989 * r + 0.5870 * g + 0.1140 * b
  #frames=[]
  while True:
   i=i+1
   for j in range(1):
      rec, frame = cap.read()
   rec, frame = cap.read()
   if not(rec):
     break
   r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
   frame = 0.2989 * r + 0.5870 * g + 0.1140 * b
   background=np.max([background,frame], axis=0)
   #frames.append(frame)
  #background=np.median(frames, axis=0)


  print(background.max(),background.min())
  cv2.imwrite('./background_max.png',background)
  cv2.imshow('background', background)
  cv2.waitKey(1)
  cap.release()
  return(background)


def save_background(background,loc):
  f = open(loc, "w")
  s=""
  for i in range(len(background)):
    s=s+"/"
    for j in range(len(background[i])):
      s=s+" "+str(background[i][j])
  f.write(s)
  f.close()


def get_background_file(loc):
  f = open(loc, "r")
  s=f.read()
  f.close()
  s=s.split("/")[1:]
  for i in range(len(s)):
    s[i]=s[i].split(" ")[1:]
    for j in range(len(s[i])):
      s[i][j]=float(s[i][j])
  s1=np.asarray(s, dtype = np.uint8)
  #cv2.imshow('background', s1)
  #cv2.waitKey(1)
  return(s)


def mask(a,m):
	for i in range(len(a)):
		for j in range(len(a[i])):
			if(m[i][j]==255):
				a[i][j]=0
	return(a)
if __name__ == '__main__':
    start_time = time.time()
    background_loc="./background.txt"
    video_loc='./X265-Crf15-1.mp4'
    name_video = './out_video.avi'
    #uncomment to get background from video it will take 20 minutes
    #background=get_background_video(video_loc)
    #save_background(background,background_loc)

    background=get_background_file(background_loc)
    if not os.path.exists('./graph'):
        os.makedirs('./graph')
  
    cap = cv2.VideoCapture('./X265-Crf15-1.mp4')
    rec, img_raw = cap.read()


    #
    #
    #coment [5:108,10:112,:] for full video
    img_raw=img_raw[5:200,10:212,:]
    # comment line 131 and un-comment line 129 to get full frame video.
    #
    #


    factor = 5
    height, width = img_raw.shape[:2]
    #print(height,width)
    name_video = './out_video.avi'
    out = cv2.VideoWriter(name_video,cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (width * factor, height * factor))
    out_binary = cv2.VideoWriter("./binary_output.avi",cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (width * factor, height * factor))
    frame_index=-1
    prior_blobs=[]
    prior_blobs_arr = np.zeros_like(img_raw[:,:,0], dtype = np.uint16)
    max_track_id = 0
    track_id=[]
    cost=[]

    #
    # Un-comment line 151 to get full length video or you can pick frame
    #while(True):
    while(frame_index<500):
        frame_index = frame_index+1
        #print(frame_index)
        rec, frame = cap.read()
        if not(rec):
          break
        r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        gray1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray=np.absolute(gray1-background)
        
        # coment [5:108,10:112] lines for full video
        #
        gray1=gray1[5:200,10:212]
        gray=gray[5:200,10:212]
        # Comment line 164 and 165 in able to see the full 
        #

        gray=np.asarray(gray, dtype = np.uint8)
        
        gray=cv2.GaussianBlur(gray, (5,5), 0)
        gray=cv2.GaussianBlur(gray, (5,5), 0)
        #img=high_pass(np.copy(gray),20)
        _,img=cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        #img=np.asarray(img, dtype = np.uint8)
        #img1=np.asarray(img1, dtype = np.uint8)
        img_raw=gray1
        #print(img_raw.shape)
        img_raw=np.asarray(img_raw, dtype = np.uint8)
        blobs_frame=blobs.BlobsFrame(img, img_raw, frame_index, min_area=5)
        current_blobs_arr=blobs_frame.blob_arr
        current_blobs=blobs_frame.get_blobs()

        graph=blobs.Graph(img_raw,current_blobs,current_blobs_arr,prior_blobs,prior_blobs_arr,max_track_id,track_id)
        graph.optimize()
        max_track_id,track_id=graph.max_track_id,graph.track_id
        cost_val=graph.cost()
        print("Frame: {} Current blobs count: {} cost value: {}".format(frame_index,len(current_blobs),cost_val))
        cost.append(cost_val)
        img1=graph.get_bbox_image(factor)
        #print(img1)
        out_frame=img1	
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img= cv2.resize(img, (width * factor, height * factor))
        cv2.imshow('out_binary',img)
        cv2.waitKey(1)
        cv2.imshow('out_frame',out_frame)
        #if(frame_index==1):
         # cv2.waitKey(0)
        cv2.waitKey(1)
        out.write(out_frame)
        out_binary.write(img)
        prior_blobs=current_blobs
        prior_blobs_arr=current_blobs_arr

    plt.figure()
    plt.plot(cost)
    plt.savefig('./cost.png')
    plt.show()
    plt.pause(0.0001)
    print("--- %s seconds ---" % (time.time() - start_time))

    out.release()
    out_binary.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)