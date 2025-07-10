import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import math
from torch.nn.functional import conv2d

def cartesian_to_polar(cartisian_vector_field):
    dx = cartisian_vector_field[..., 0]
    dy = cartisian_vector_field[..., 1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan(dy / dx)
    np.nan_to_num(theta)
    return np.stack([r, theta], axis = -1)


cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")

ret, frame1 = cap.read()
cv2.namedWindow("Optical Flow Visualizer")

flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
# Gaussian kernal initialization (for convolving in future)
kernal = cv2.getGaussianKernel(ksize=3, sigma=1)
gaussian_kernal = kernal @ kernal.T
kernel_tensor = torch.from_numpy(gaussian_kernal).unsqueeze(0).unsqueeze(0).float()

while ret:
    ret, frame2 = cap.read()
    visulizer_frame = frame2.copy() # So we don't write vectors on this frame when it becomes frame 1
    if not ret:
        break

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, flow,
                                        0.5, 4, 25, 5, 7, 1.5,
                                        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    # Convolve flow vectors into compressed image
    # Convert to torch tensors and add batch/channel dimensions
    x_tensor = torch.from_numpy(flow[..., 0]).unsqueeze(0).unsqueeze(0).float()
    y_tensor = torch.from_numpy(flow[..., 1]).unsqueeze(0).unsqueeze(0).float()
    # Stride represents the factor by which the convolution compresses the image
    flow_convolved = np.stack([conv2d(x_tensor, kernel_tensor, stride=8).squeeze(0).squeeze(0),
                               conv2d(y_tensor, kernel_tensor, stride=8).squeeze(0).squeeze(0)], axis = -1)
    
    # Visulize Vectors
    frameH, frameW, _ = flow_convolved.shape
    vector_points = np.stack(np.meshgrid(np.arange(frameW), np.arange(frameH), indexing="xy"), axis = -1).reshape(-1, 2)

    vector_endings = flow_convolved.reshape(-1, 2)
    for vector_point, vector_endings in zip(vector_points, vector_endings):
        starting_point = list(map(int, vector_point * 8))
        ending_point = list(map(int, vector_point * 8 + vector_endings))
        cv2.arrowedLine(visulizer_frame, starting_point, ending_point, (0,225,0), 1)


    """
    flow_polar_convolved = cartesian_to_polar(flow_convolved) # Convert convoluted to polar form
    
    # Cluster flow
    normalizer = StandardScaler()
    # Generate Feature Map
    H, W, _ = flow_polar_convolved.shape  # Convolved dimensions
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    positions = np.stack([ii / H, jj / W], axis=-1).reshape(-1, 2)
    vector_features = flow_polar_convolved.reshape(-1, 2)
    features = normalizer.fit_transform(np.concatenate([positions, vector_features], axis = 1))

    # TODO: Consider using the random state from last time
    n_clusters = 5
    cluster = KMeans(n_clusters = n_clusters, random_state = 0).fit(features)
    
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n_clusters)]
    """
    
    cv2.imshow("Optical Flow Visualizer", visulizer_frame)
    cv2.waitKey(1)
    frame1 = frame2
