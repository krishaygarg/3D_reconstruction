from remotezip import RemoteZip #we import the library to treat links as zip files
from pathlib import Path #we import this for paths

import cv2 #this is what reads videos


def extract(IN,OUT):

    if not list(Path(OUT).rglob("rgb_video.mp4")):    
    #if the video already exists in the folder called out, then dont start this
    #we search in every folder and file inside of OUT with the rglob function and we force the result into a list
    #an empty list is treated as false

        with RemoteZip(IN) as z: 
        #this is the common method for opening files in python, it deals with errors and stuff and closes stuff
        #btw reading a link here means just taking the last bits, bc they store the info in the last bits, of course not reading 20GB of video
            vids = [i for i in z.infolist() if i.filename.endswith("rgb_video.mp4")] #FIRST, we look for videos
            shoes = [i for i in vids if any(k in i.filename.lower() for k in ("sneaker", "shoe", "boot"))] or vids #SECOND ,we look for videos with shoes, or else all vids
            shoes.sort(key=lambda i: i.file_size) #we sort the list by file size
            pick = shoes[len(shoes) // 2]  # we pick the median one, this is really arbitrary, however worked for me
            print("Picked:", pick.filename, f"({pick.file_size/1e6:.1f} MB)") #print the name and its size
            z.extract(pick, OUT) #extract that video in OUT, which is the uco3dshoe folder

def extract_frames(n_frames, maxs, video, IMAGES):

    cap = cv2.VideoCapture(str(video)) #we open the video for reading
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #we count the number of frames in the video
    keep = {round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)} #the index of the frames we want to keep, in a set

    idx = 0 #index for current frame
    saved = 0 #number of frames saved
    while True:
        ok, frame = cap.read() #we read a frame, every iteration it reads the next frame
        if not ok: #if out of bounds, then break
            break
        if idx in keep: #if the frame id is one of the ones we want to keep
            h, w = frame.shape[:2] #frame is a numpy array, shape just returns the dimensions
            s = maxs / max(h, w) #scale factor
            if s < 1: #if our frame is bigger than 1600
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) #we resize it using INTER AREA interpolation so doesnt lose detail much
            cv2.imwrite(str(IMAGES / f"{saved:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]) # we store the image with the saved number as name
            saved += 1  
        idx += 1 
    cap.release() #closes the video
    print("Frames saved:", saved)

    return saved
