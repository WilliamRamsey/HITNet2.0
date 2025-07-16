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

class BackgroundTracker:
    @staticmethod
    def cartesian_to_polar(cartisian_vector_field):
        """
        Returns a polar vector field
        Return:
            (H, W, 2) -> (r, theta)
        """
        dx = cartisian_vector_field[..., 0]
        dy = cartisian_vector_field[..., 1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan(dy / dx)
        np.nan_to_num(theta)
        return np.stack([r, theta], axis = -1)

    def __init__(self, frame1, tracking_points: list, stride: int = 8):
        self.tracking_points = tracking_points
        self.frame_H, self.frame_W = frame1.shape
        self.stride = stride
        self.flow = np.zeros((self.frame_H, self.frame_W, 2), dtype=np.float32)
    
    def generate_flow(self, frame1, frame2):
        """
        (H, W, 2) -> (dx, dy)
        Generates a vector field representing the flow at each point.
        """
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        self.flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, self.flow,
                                            0.5, 4, 25, 5, 7, 1.5,
                                            flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        return self.flow

    def convolve_flow(self):
        """
        Convolves a flow field in each dimension.
        Uses a gaussian blur sample at each specified stride

        The blur kernal size is not related to the stride
        This means that there can be unsampled and double sampled areas within a flow field.

        Return:
        (H / stride, W / stride, 2) -> (dx, dy)
        """
        kernal = cv2.getGaussianKernel(ksize=3, sigma=1)
        gaussian_kernal = kernal @ kernal.T
        kernel_tensor = torch.from_numpy(gaussian_kernal).unsqueeze(0).unsqueeze(0).float()
        
        # Convolve flow vectors into compressed image
        # Convert to torch tensors and add batch/channel dimensions
        x_tensor = torch.from_numpy(self.flow[..., 0]).unsqueeze(0).unsqueeze(0).float()
        y_tensor = torch.from_numpy(self.flow[..., 1]).unsqueeze(0).unsqueeze(0).float()
        # Stride represents the factor by which the convolution compresses the image
        return np.stack([conv2d(x_tensor, kernel_tensor, stride = self.stride).squeeze(0).squeeze(0),
                         conv2d(y_tensor, kernel_tensor, stride = self.stride).squeeze(0).squeeze(0)],
                        axis = -1)

    def generate_flow_features(self):
        """
        
        """

    def classify_tracking_points(self):
        ...

    def track_points(self, frame1, frame2):
        # TODO: How do we deal with tracking points in different clusters?
        # TODO: Where do we handle tracking point reselection?
        """
        Given the tracking points coorispond to point1
        Returns their new position in frame2
        

        Generate flow (H, W, 2) -> (dx, dy)
        Convolve flow (H/step, W/step, 2)
        Convert convolved to polar (H/step, W/step, 2) -> r, theta
        Generate flow feature map from polar ((H/step * W/step), 4) -> (Y / (H/step), X / (W/step), r, theta / pi) with minmax norm radius
        Run K means cluster on flow feature map
        """
        # Generate flow
        # Convolve flow
        # Generate flow feature map
        # Run KMeans cluster on flow
        # Classify points based on K-Means

        ...
    
    def visualize_flow(self, convolved_flow, display_img):
        """
        Plots the given convolved flow on the image.
        """
        convolved_H, convolved_W, _ = convolved_flow.shape
        vector_origins = np.stack(np.meshgrid(np.arange(convolved_W),
                                              np.arange(convolved_H),
                                              indexing="xy"), axis = -1).reshape(-1, 2)
        vector_endings = convolved_flow.reshape(-1, 2)

        for vector_origin, vector_ending in zip(vector_origins, vector_endings):
            starting_point = list(map(int, vector_origin * 8))
            ending_point = list(map(int, vector_origin * 8 + vector_ending))
            cv2.arrowedLine(display_img, starting_point, ending_point, (0,225,0), 1)
        return display_img


cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")

ret, frame1 = cap.read()
cv2.namedWindow("Optical Flow Visualizer")

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
