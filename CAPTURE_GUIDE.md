# Group 1: Live Capture Session Field Guide

This guide details best practices for capturing physical video for 3D reconstruction (NeRF, Gaussian Splatting, Photogrammetry / COLMAP).

---

## 1. Object & Environment Setup
- **Object Choice**: Select an object with rich, non-repeating surface textures (e.g., shoe, patterned backpack, potted succulent, textured figurine).
- **Surface Avoidance**: Avoid transparent glass, highly polished mirrors, or uniform matte white surfaces (these lack feature points for Structure-from-Motion).
- **Lighting**:
  - Use even, diffuse 360° illumination (avoid a single harsh lamp creating moving shadows as you orbit).
  - Ambient overcast daylight or diffuse multi-point indoor LED lighting works best.
- **Clearance**: Place the object on a pedestal, stool, or table with at least 1.5–2 meters of walking clearance around it.

---

## 2. Camera / Smartphone Settings
- **Resolution & Frame Rate**: 
  - Set to **4K (3840×2160) at 60 FPS**.
  - 60 FPS ensures faster shutter speeds and smaller inter-frame displacement, drastically minimizing motion blur.
- **AE/AF Lock (Critical!)**:
  - Press and hold on the object in your camera viewfinder to **lock Auto-Focus and Auto-Exposure (AE/AF Lock)**.
  - If focus or exposure "pumps" during the 360° orbit, camera alignment (SfM) will struggle to converge.
- **Color Profile & Stabilization**:
  - Keep standard color / SDR.
  - Standard optical / digital stabilization enabled.

---

## 3. Orbit Execution Protocol
1. Stand at a steady distance (e.g. 1.0–1.5 meters) with camera at object eye level.
2. Press Record and **hold still for 1.5–2 seconds** before starting the movement.
3. Orbit around the object in a smooth, continuous circle:
   - Walk with bent knees ("ninja walk") to absorb footsteps.
   - Maintain a constant radius and elevation.
   - Complete the full 360° circle in **15 to 20 seconds**.
4. Once completing the loop, **hold still for another 1.5–2 seconds** before pressing Stop.
5. Immediately transfer the raw video (`.mp4` or `.mov`) to the team workspace.

---

## 4. Run Automated Processing
```bash
python3 process_video.py -i my_capture_raw.mov -o extracted_frames -a group2_handoff.zip
```
The script will automatically discard the start/stop shake, sample 80–130 sharp frames, normalize resolution to 1280px longest edge, and produce the `group2_handoff.zip` deliverable!
