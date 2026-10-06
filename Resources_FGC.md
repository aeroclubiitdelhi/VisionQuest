# AeroClub Tech FGC: Learning Resources

AeroClub, Tech FGC · Oct 3, 2026 · @yasasvi

## About this list

These are the topics worth learning before the AeroClub event. You do not need prior experience with drones or computer vision: basic Python is enough to start. Go through the setup and the first two sections first.

## Setup

Have a working Python environment on at least one teammate's laptop. Everything here runs on Windows, macOS or Linux.

1. Install Python 3.9 or newer from [python.org](https://www.python.org/downloads/). On Windows, tick "Add Python to PATH" during install.
2. Install an editor. [VS Code](https://code.visualstudio.com/) with its Python extension works well.
3. Optional but recommended: learn to make a virtual environment ([venv guide](https://docs.python.org/3/library/venv.html)).
4. Run `pip install numpy opencv-contrib-python` ([pip help](https://pip.pypa.io/en/stable/installation/)).
5. Check it works: `python -c "import cv2, numpy; print(cv2.__version__)"` should print a version number.

## Python and NumPy

NumPy is how you do maths on arrays of numbers quickly in Python.

| Resource | What to take from it |
| --- | --- |
| [The Python Tutorial](https://docs.python.org/3/tutorial/) | Sections 3 to 9: lists, dicts, functions, classes |
| [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | Creating arrays, indexing, shapes, basic maths |
| [NumPy linear algebra (numpy.linalg)](https://numpy.org/doc/stable/reference/routines.linalg.html) | Matrix multiplication (`@`), inverse, norms |

## OpenCV and ArUco markers

OpenCV is the standard image processing library. ArUco markers are square black and white codes that robots and drones use to recognise known points.

| Resource | What to take from it |
| --- | --- |
| [OpenCV-Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) | "Gui Features" and "Core Operations": reading, showing and drawing on images |
| [Detection of ArUco markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) | What a marker dictionary is, and how detection returns IDs and corners |
| [ArUco module reference](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html) | Look up function arguments when you need them |

Older blog posts use ArUco functions that were renamed in OpenCV 4.7, so prefer the official docs.

## Cameras and images

These pages explain how a camera turns the 3D world into a 2D image.

| Resource | What to take from it |
| --- | --- |
| [Pinhole camera model (Wikipedia)](https://en.wikipedia.org/wiki/Pinhole_camera_model) | How a 3D point becomes a pixel |
| [Camera calibration (OpenCV tutorial)](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) | What the camera matrix and lens distortion are |
| [Camera calibration and 3D reconstruction (OpenCV reference)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | The camera and distortion equations OpenCV uses |
| [Homogeneous coordinates (Wikipedia)](https://en.wikipedia.org/wiki/Homogeneous_coordinates) | Why pixels are often written as (u, v, 1) |

## 3D geometry

Vectors, rotations and coordinate frames at a first-year level.

| Resource | What to take from it |
| --- | --- |
| [Essence of Linear Algebra (3Blue1Brown)](https://www.3blue1brown.com/topics/linear-algebra) | Chapters 1 to 5 and 13: vectors, matrices as transformations, change of basis |
| [Rotation matrix (Wikipedia)](https://en.wikipedia.org/wiki/Rotation_matrix) | Rotations about x, y and z, and combining them |
| [Euler angles (Wikipedia)](https://en.wikipedia.org/wiki/Euler_angles) | What roll, pitch and yaw mean, and why the order matters |
| [Local tangent plane coordinates (Wikipedia)](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates) | The East-North-Up (ENU) frame used in robotics |

## Before the event

- [ ] Python, NumPy and OpenCV installed and the check command works
- [ ] Detected ArUco markers in an image with OpenCV
- [ ] Comfortable multiplying matrices and vectors in NumPy
- [ ] Know what roll, pitch and yaw mean
- [ ] Laptop charger ready
