# 3D Reconstruction – Video → COLMAP

Requires Python and these libraries:

pip install -r requirements.txt

- remotezip – downloads only the needed video from the dataset zip
- opencv-python – reads the video and saves the frames
- pycolmap – runs COLMAP to get camera poses and the point cloud

Run `python main.py`, zip `ws/undistorted`, and train with `train_gsplat_shoe.ipynb` on Colab (GPU runtime).