import numpy as np
from scipy.linalg import pinv

def sif(x, z, u, P, A, B, C, Q, R,delta):
    """
    Sliding Innovation Filter (SIF) Function

    Parameters:
    x : np.ndarray
        Current state estimate (n,)
    z : np.ndarray
        Current measurement (m,)
    u : float
        Control input
    P : np.ndarray
        Current error covariance (n x n)
    A, B, C : np.ndarray
        System matrices
    Q, R : np.ndarray
        Process and measurement noise covariances
    delta: np.ndarray
        Sliding layer widths 
            For instance: (delta = np.array([0.05, 1, 0.5]))

    Returns:
    x_updated : np.ndarray
        Updated state estimate (n,)
    P_updated : np.ndarray
        Updated error covariance (n x n)
    """
    n = x.shape[0]
    m = z.shape[0]
    sat = np.zeros(m)  # Saturation terms

    # Prediction stage
    x_pred = A @ x + B.flatten() * u  # Predict state
    P_pred = A @ P @ A.T + Q  # Predict error covariance
    innov = z - C @ x_pred  # Innovation

    # Update stage
    for i in range(m):
        abs_innov = np.abs(innov[i])
        if abs_innov / delta[i] >= 1:
            sat[i] = 1
        else:
            sat[i] = abs_innov / delta[i]

    K = pinv(C) @ np.diag(sat)  # SIF gain
    x_updated = x_pred + K @ innov  # Update state estimate
    P_updated = (np.eye(n) - K @ C) @ P_pred @ (np.eye(n) - K @ C).T + K @ R @ K.T  # Update error covariance

    x_updated = x_updated.flatten()

    return x_updated, P_updated

def kf_filter(x, z, u, P, A, B, C, Q, R):
    """
    Kalman Filter (KF) Function

    Parameters:
    x : np.ndarray
        Current state estimate (n,)
    z : np.ndarray
        Current measurement (m,)
    u : float
        Control input
    P : np.ndarray
        Current error covariance (n x n)
    A, B, C : np.ndarray
        System matrices
    Q, R : np.ndarray
        Process and measurement noise covariances

    Returns:
    x_updated : np.ndarray
        Updated state estimate (n,)
    P_updated : np.ndarray
        Updated error covariance (n x n)
    """
    # Prediction
    x_pred = A @ x + B.flatten() * u
    P_pred = A @ P @ A.T + Q

    # Innovation
    innov = z - C @ x_pred
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ pinv(S)

    # Update
    x_updated = x_pred + K @ innov
    P_updated = (np.eye(A.shape[0]) - K @ C) @ P_pred

    # Flatten the updated state to ensure it's a 1D array
    x_updated = x_updated.flatten()

    return x_updated, P_updated
