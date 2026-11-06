
from pathlib import Path #we import this for paths
from video import extract, extract_frames #the function from the video.py file that extracts the video and the frames
from colmap import colmap_reconstruction #the function we going to use for point cloud and reconstruction

BASE = "https://huggingface.co/datasets/facebook/uco3d/resolve/main/" #link that has all the videos
RGB_ZIP = BASE + "part_rgb_videos_0232.zip" #the specific folder that has all the videos
OUT = "uco3d_shoe" #this is the folder where we are going to spit out the video

WS = Path("ws") #workspace, where colmap will put everythin, this is just the path for now
IMAGES = WS / "images" #this is we are going to put all the frames we extract from the video
SPARSE = WS / "sparse" #this is where colmap stores the info about the camera, the positions in each frame and of course, the point cloud
UNDIST = WS / "undistorted" #this is the final file we send for training, the photos without any distorsion + sparse info with camera without distorsion values
IMAGES.mkdir(parents=True, exist_ok=True) #we create images and sparse folder
SPARSE.mkdir(parents=True, exist_ok=True)

db = WS / "database.db"
if db.exists():
    db.unlink()  # start fresh each run, otherwise features get duplicated

N_FRAMES, MAX_SIDE = 200, 1600 #number of frames we want and the width resolution we want for each frame (1600 is the convention for gaussian splatting)

extract(RGB_ZIP,OUT)#extract from arg 1, insert into arg 2
video = next(Path(OUT).rglob("rgb_video.mp4")) #we store the path for our video

saved = extract_frames(N_FRAMES, MAX_SIDE, video , IMAGES)

colmap_reconstruction(db,IMAGES,SPARSE,UNDIST,saved)