""" ===================================================================================================================
Head pose estimation code for a modified iTracker where estimated head pose values are added as a new input.

Author: hysts (github), 2018/19.

Modified: Thomas Gibson (tjg1g19@soton.ac.uk), 2022.

Website: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-
         interaction/appearance-based-gaze-estimation-in-the-wild

Original Paper: Appearance-based Gaze Estimation in the Wild (MPIIGaze)
X. Zhang, Y. Sugano, M. Fritz and A. Bulling,
Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June, p.4511-4520, (2015).

Date: 05/04/2022
====================================================================================================================="""

import cv2
import dlib
import numpy as np
from scipy.spatial.transform import Rotation

LANDMARKS: np.ndarray = np.array([
    [-0.07141807, -0.02827123, 0.08114384],
    [-0.07067417, -0.00961522, 0.08035654],
    [-0.06844646, 0.00895837, 0.08046731],
    [-0.06474301, 0.02708319, 0.08045689],
    [-0.05778475, 0.04384917, 0.07802191],
    [-0.04673809, 0.05812865, 0.07192291],
    [-0.03293922, 0.06962711, 0.06106274],
    [-0.01744018, 0.07850638, 0.04752971],
    [0., 0.08105961, 0.0425195],
    [0.01744018, 0.07850638, 0.04752971],
    [0.03293922, 0.06962711, 0.06106274],
    [0.04673809, 0.05812865, 0.07192291],
    [0.05778475, 0.04384917, 0.07802191],
    [0.06474301, 0.02708319, 0.08045689],
    [0.06844646, 0.00895837, 0.08046731],
    [0.07067417, -0.00961522, 0.08035654],
    [0.07141807, -0.02827123, 0.08114384],
    [-0.05977758, -0.0447858, 0.04562813],
    [-0.05055506, -0.05334294, 0.03834846],
    [-0.0375633, -0.05609241, 0.03158344],
    [-0.02423648, -0.05463779, 0.02510117],
    [-0.01168798, -0.04986641, 0.02050337],
    [0.01168798, -0.04986641, 0.02050337],
    [0.02423648, -0.05463779, 0.02510117],
    [0.0375633, -0.05609241, 0.03158344],
    [0.05055506, -0.05334294, 0.03834846],
    [0.05977758, -0.0447858, 0.04562813],
    [0., -0.03515768, 0.02038099],
    [0., -0.02350421, 0.01366667],
    [0., -0.01196914, 0.00658284],
    [0., 0., 0.],
    [-0.01479319, 0.00949072, 0.01708772],
    [-0.00762319, 0.01179908, 0.01419133],
    [0., 0.01381676, 0.01205559],
    [0.00762319, 0.01179908, 0.01419133],
    [0.01479319, 0.00949072, 0.01708772],
    [-0.045, -0.032415, 0.03976718],
    [-0.0370546, -0.0371723, 0.03579593],
    [-0.0275166, -0.03714814, 0.03425518],
    [-0.01919724, -0.03101962, 0.03359268],
    [-0.02813814, -0.0294397, 0.03345652],
    [-0.03763013, -0.02948442, 0.03497732],
    [0.01919724, -0.03101962, 0.03359268],
    [0.0275166, -0.03714814, 0.03425518],
    [0.0370546, -0.0371723, 0.03579593],
    [0.045, -0.032415, 0.03976718],
    [0.03763013, -0.02948442, 0.03497732],
    [0.02813814, -0.0294397, 0.03345652],
    [-0.02847002, 0.03331642, 0.03667993],
    [-0.01796181, 0.02843251, 0.02335485],
    [-0.00742947, 0.0258057, 0.01630812],
    [0., 0.0275555, 0.01538404],
    [0.00742947, 0.0258057, 0.01630812],
    [0.01796181, 0.02843251, 0.02335485],
    [0.02847002, 0.03331642, 0.03667993],
    [0.0183606, 0.0423393, 0.02523355],
    [0.00808323, 0.04614537, 0.01820142],
    [0., 0.04688623, 0.01716318],
    [-0.00808323, 0.04614537, 0.01820142],
    [-0.0183606, 0.0423393, 0.02523355],
    [-0.02409981, 0.03367606, 0.03421466],
    [-0.00756874, 0.03192644, 0.01851247],
    [0., 0.03263345, 0.01732347],
    [0.00756874, 0.03192644, 0.01851247],
    [0.02409981, 0.03367606, 0.03421466],
    [0.00771924, 0.03711846, 0.01940396],
    [0., 0.03791103, 0.0180805],
    [-0.00771924, 0.03711846, 0.01940396],
],
    dtype=np.float)

REYE_INDICES: np.ndarray = np.array([36, 39])
LEYE_INDICES: np.ndarray = np.array([42, 45])
MOUTH_INDICES: np.ndarray = np.array([48, 54])

CHIN_INDEX: int = 8
NOSE_INDEX: int = 30


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def detect_facial_landmarks(image: np.ndarray, detector: dlib.fhog_object_detector, predictor: dlib.shape_predictor):
    # Face detection
    bboxes = detector(image, 0)
    bbox = bboxes[0]
    # Landmark detection
    predictions = predictor(image, bbox)
    landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()], dtype=np.float)  # Format Landmarks

    return landmarks

def generate_face_bbox(landmarks: np.ndarray):
    faceWidth = (landmarks[16][0] - landmarks[0][0])
    bBoxScale = 1.5
    faceWidth = bBoxScale * faceWidth
    faceCornerTX = landmarks[30][0] - (faceWidth / 2)
    faceCornerTY = landmarks[30][1] - (faceWidth / 2)
    faceCornerBX = landmarks[30][0] + (faceWidth / 2)
    faceCornerBY = landmarks[30][1] + (faceWidth / 2)

    faceBbox = [int(faceCornerTX), int(faceCornerTY), faceWidth, faceWidth]

    return faceBbox


def annotate_facial_landmarks(image: np.ndarray, landmarks: np.ndarray):
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
               (0, 0, 0)]

    for i in range(0, 17):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (255, 0, 0), -1)

    for i in range(17, 27):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (0, 255, 0), -1)

    for i in range(27, 36):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (0, 0, 255), -1)

    for i in range(36, 42):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (255, 255, 0), -1)

    for i in range(42, 48):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (0, 255, 255), -1)

    for i in range(48, 68):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (255, 0, 255), -1)

    return image


def estimate_head_pose(landmarks: np.ndarray):
    """3D face model for Multi-PIE 68 points mark-up.

    In the camera coordinate system, the X axis points to the right from
    camera, the Y axis points down, and the Z axis points forward.

    The face model is facing the camera. Here, the Z axis is
    perpendicular to the plane passing through the three midpoints of
    the eyes and mouth, the X axis is parallel to the line passing
    through the midpoints of both eyes, and the origin is at the tip of
    the nose.

    The units of the coordinate system are meters and the distance
    between outer eye corners of the model is set to 90mm.

    The model coordinate system is defined as the camera coordinate
    system rotated 180 degrees around the Y axis.
    """

    # Find rotation matrix (head pose) of face model
    camera_matrix = np.array([960., 0., 30, 0., 960., 18., 0., 0., 1.]).reshape(3, 3)
    dist_coefficients = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)

    rvec = np.zeros(3, dtype=np.float)
    tvec = np.array([0, 0, 1], dtype=np.float)

    _, rvec, tvec = cv2.solvePnP(LANDMARKS, landmarks, camera_matrix, dist_coefficients, rvec, tvec,
                                 useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    # Calculate model after rotation and translation
    rot = Rotation.from_rotvec(rvec)
    head_position = tvec
    rotationMat = rot.as_matrix().T
    model3d = LANDMARKS @ rotationMat
    model3d = model3d + head_position
    center = model3d[np.concatenate([REYE_INDICES, LEYE_INDICES, MOUTH_INDICES])].mean(axis=0)

    # Compute normalising rotation
    center = center.ravel()
    z_axis = normalize_vector(center)
    head_rot = rot.as_matrix()
    head_x_axis = head_rot[:, 0]
    y_axis = normalize_vector(np.cross(z_axis, head_x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    axis_mat = np.vstack([x_axis, y_axis, z_axis])
    normalizing_rot = Rotation.from_matrix(axis_mat)

    # Normalise head pose using rotation computed above
    normalized_head_rot = rot * normalizing_rot
    normalized_mat1 = normalized_head_rot.as_matrix()
    euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
    normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    return normalized_head_rot2d
