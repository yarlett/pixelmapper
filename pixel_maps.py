import numpy as np


def get_locally_random_pixelmap(width, height, sigma):
    """
    Returns a pixelmap in which pixels derive their values from randomly sampled spatially proximal pixels.
    """
    X = np.arange(width).reshape(1, width)
    X = np.tile(X, (height, 1))
    Y = np.arange(height).reshape(height, 1)
    Y = np.tile(Y, (1, width))
    # Add Gaussian noise to the pixel locations.
    X += np.round(sigma * np.random.randn(*X.shape)).astype(np.int64)
    X = np.clip(X, 0, width - 1)
    Y += np.round(sigma * np.random.randn(*Y.shape)).astype(np.int64)
    Y = np.clip(Y, 0, height - 1)
    # Enumerate and return the pixelmap.
    out = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out.append((i, j, Y[i, j], X[i, j]))
    out = np.array(out, dtype=np.int64)
    return out


def get_kaleidoscope_pixelmap(width, height, K):
    """
    Returns a pixel map that implements kaleidoscopic reflection of an image.
    """
    # Initialization.
    theta_wedge = 360. / K
    # Get X vectors relative to the image center.
    xvals = np.arange(width)
    X = xvals.astype(np.float64)
    X -= xvals.mean()
    X = X.reshape(1, width)
    X = np.tile(X, (height, 1))
    # Get Y vectors relative to the image center.
    yvals = np.arange(height)[::-1]
    Y = yvals.astype(np.float64)
    Y -= Y.mean()
    Y = Y.reshape(height, 1)
    Y = np.tile(Y, (1, width))
    # Get the angle and length of each vector from the image center.
    A = np.degrees(np.arctan2(Y, X))
    A[A < 0.] += 360.
    L = np.sqrt(X * X + Y * Y + 1e-6)
    # Set the amount to rotate each pixel.
    R = A - np.round(A / theta_wedge) * theta_wedge
    R = np.radians(R)
    # For each target pixel determine its source x value.
    SX = L * np.cos(R)
    SX = np.round(SX)
    SX += xvals.mean()
    SX = SX.astype(np.int64)
    SX = np.clip(SX, 0, width - 1)
    # For each target pixel determine its source y value.
    SY = L * np.sin(R)
    SY = np.round(SY)
    SY += yvals.mean()
    SY = SY.astype(np.int64)
    SY = np.clip(SY, 0, height - 1)
    # Enumerate and return the pixelmap.
    out = []
    for i in range(SX.shape[0]):
        for j in range(SX.shape[1]):
            out.append((i, j, SY[i, j], SX[i, j]))
    out = np.array(out, dtype=np.int64)
    return out
