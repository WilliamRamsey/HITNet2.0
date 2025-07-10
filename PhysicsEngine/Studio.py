import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn.functional import conv2d

def cartesian_to_polar(cartisian_vector_field):
    dx = cartisian_vector_field[..., 0]
    dy = cartisian_vector_field[..., 1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan(dy / dx)
    np.nan_to_num(theta)
    return np.stack([r, theta], axis = -1)

def generate_flow_features(flow, ):
    """
    
    """

# Video Reading
cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")
ret, frame1 = cap.read()

# Template and Frame Point Structure
flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
global frame_points
frame_points = [] # [(x, y)]
template_points = [] # [(x,y)]


# CV2 window management
def frame_click_handler(event, x, y, _, __):
    global frame_points
    global frame_select_image
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame_select_image, (x, y), 2, (0,0,225), 5)
        frame_points.append((x, y))

cv2.namedWindow("Optical Flow Visualizer")
cv2.namedWindow("Template Point Selection")
cv2.namedWindow("Frame Point Selection")
cv2.setMouseCallback("Frame Point Selection", frame_click_handler)

# Initialize field tracking points
def reselect_image_points(frame):
    global frame_points
    frame_points.clear()
    global frame_select_image
    frame_select_image = frame.copy()
    while len(frame_points) < 4:
        cv2.imshow("Frame Point Selection", frame_select_image)
        cv2.waitKey(1)
    cv2.destroyWindow("Frame Point Selection")

reselect_image_points(frame1)

# Gaussian kernal initialization (for convolving in future)
kernal = cv2.getGaussianKernel(ksize=3, sigma=1)
gaussian_kernal = kernal @ kernal.T
kernel_tensor = torch.from_numpy(gaussian_kernal).unsqueeze(0).unsqueeze(0).float()

while ret:
    # Load new frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Apply gray transform
    # TODO: Consider Applying Dynamic Canny Edge Detection
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    """
    Flow and Clustering
    """
    # Find the flow between frame1 and frame 2
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0) #row,col,(x,y)

    # Convolve flow vectors into compressed image
    # Convert to torch tensors and add batch/channel dimensions
    x_tensor = torch.from_numpy(flow[..., 0]).unsqueeze(0).unsqueeze(0).float()
    y_tensor = torch.from_numpy(flow[..., 1]).unsqueeze(0).unsqueeze(0).float()
    # Stride represents the factor by which the convolution compresses the image
    flow_convolved = np.stack([conv2d(x_tensor, kernel_tensor, stride=8).squeeze(0).squeeze(0),
                               conv2d(y_tensor, kernel_tensor, stride=8).squeeze(0).squeeze(0)], axis = -1)
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
    cluster = KMeans(n_clusters = 5, random_state = 0).fit(features)

    # Find the clusters each tracked point is in.
    # We just use the convoluted normalized value. 
    frame_point_clusters = [cluster.labels_.reshape(H, W)[point[0] // 8, point[1] // 8] for point in frame_points]
    if all([cluster_id == frame_point_clusters[0] for cluster_id in frame_point_clusters]):
        flow_cluster_vectors = flow_convolved[cluster.labels_ == frame_point_clusters[0]]
        avg_dx = sum(flow_convolved[...,0]) / len(flow_convolved[...,0])
        avg_dy = sum(flow_convolved[...,1]) / len(flow_convolved[...,1])
        for i in range(len(frame_points)):
            frame_points[i] = (frame_points[i][0] + avg_dx, frame_points[i][1] + avg_dy)
    else:
        print("reselecting points")
        reselect_image_points(frame2)
    

    # TODO: HOMOGRAPHY TEMPLATE SELECTION GOES HERE
    # TODO: Find in frame helmet velo by averaging the optical flow vectors originating (or prob ending since flow = dx from last to this) in mask.

    # Display output
    visulizer_frame = frame2.copy() # So we don't write vectors on this frame when it becomes frame 1
    
    for point in frame_points:
        cv2.circle(visulizer_frame, (int(point[0]), int(point[1])), 3, (0, 225, 225), 5)
    
    cv2.imshow("Optical Flow Visualizer", visulizer_frame)
    cv2.waitKey(0)
    frame1 = frame2

# Create a smooth homography transform
# Apply transform to masks and generate real accelerations
