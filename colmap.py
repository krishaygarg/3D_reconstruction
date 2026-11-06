import pycolmap #this is colmap for python, to create the point cloud from the video

def colmap_reconstruction(db,IMAGES,SPARSE,UNDIST,saved):

    pycolmap.extract_features( #colmap tries to find keypoints for every image and that info goes to the database
        str(db), str(IMAGES),
        camera_mode=pycolmap.CameraMode.SINGLE,  # all frames come from the same phone, so they all share every intrinsic
        camera_model="OPENCV",
    )
    pycolmap.match_exhaustive(str(db))  # slow step, ~10-20 min on CPU, this compares descriptors to find corresponding keypoints
    recs = pycolmap.incremental_mapping(str(db), str(IMAGES), str(SPARSE)) #this is where triangulation and bundle adjustment happens

    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images()) #keeps the reconstruction with the most registered images

    pycolmap.undistort_images(str(UNDIST), str(SPARSE / str(best_id)), str(IMAGES)) #create the undistorted images
    (UNDIST / "sparse" / "0").mkdir(parents=True, exist_ok=True) #save them in a folder
    for f in (UNDIST / "sparse").glob("*.bin"):
        f.rename(UNDIST / "sparse" / "0" / f.name)
