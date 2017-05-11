import numpy as np

"""
Structure From Motion (SFM) Utilities
"""


def find_point_zplane(u, v, cam_data, z=0):
    """
    Converts a 2D point to a 3D point using a known z value

    cam_data must contain:
    -mtx: The camera matrix of intrinsic parameters
    -rotation_matrix_inv: The inverse of the extrinsic rotation vectors in the form 3x3
    -tvec: The extrinsic translation vectors in the form 1x3

    :param u: The x coordinate of the 2D point
    :param v: The y coordinate of the 2D point
    :param cam_data: The camera data which holds data such as mtx, rotation_matrix_inv and tvec
    :param z: The z coordinate of the point in 3D. This is 0 by default
    :type cam_data: CamData
    :type z: int
    """

    # TODO: change back-projection function to use camera and cached matrices

    """
    Theory:
    a 2D point p (u, v, 1).T can be represented as a 3D point P (X, Y, Z).T as follows:
    p = K(R * P + t) / s
    where K is camera matrix, R is rotation matrix, t is translation matrix and s is a constant
    If Z is constant, we can find s:
    s = ( P + (R^-1)t ) / ( (R^-1)(K^-1)p )
    thus,
    s = ( Z + ((R^-1)t)[2, 0] ) / ( (R^-1)(K^-1)p[2,0] )
    we can then find P using s;
    P = (R^-1) * (s * (K^-1) * p - t)
    """

    assert cam_data.mtx.shape == (3, 3), "camera intrinsic parameters must be a 3x3 matrix: {}".format(
        cam_data.mtx.shape)
    assert cam_data.rotation_matrix_inv.shape == (3, 3), \
        "camera inverse rotation matrix must be a 3x3 matrix: {}".format(cam_data.rotation_matrix_inv.shape)
    assert cam_data.tvec.shape == (3, 1), "camera translation vectors must be a 3x1 matrix: {}".format(
        cam_data.tvec.shape)

    # Convert matrices to useful formats. Uses notation common for 3d reconstruction such as K, R and t
    k = cam_data.mtx
    p = np.matrix(np.array([u, v, 1]).reshape(3, 1))
    k_inv = np.matrix(np.linalg.inv(k))
    r_inv = cam_data.rotation_matrix_inv
    t = np.matrix(cam_data.tvec)

    temp_mtx = r_inv * k_inv * p
    temp_mtx2 = r_inv * t
    s = (z + temp_mtx2[2, 0]) / temp_mtx[2, 0]
    pt = r_inv * (s * k_inv * p - t)
    return np.array(pt).ravel()


def calc_projection_error(projected, original):
    """
    Calculates the projection error of a set of points and their corresponding projected points
    :param projected: The projected points
    :param original: The original image points
    :return: projection erorr
    :rtype: float
    """
    # TODO: Add assertions to calc_projection_error
    diffs = projected.reshape(-1, 2) - original.reshape(-1, 2)
    error = 0
    for i in xrange(len(diffs)):
        diff = diffs[i]
        error += np.linalg.norm(diff)
    error /= len(diffs)
    return error


def triangulate(all_markers, cameras):
    """
    Triangulates points using Linear-Eigen Singular Value Decomposition (SVD).
    See Page 7-8 of 'Triangulation ~ Richard I. Hartley and Peter Sturm' for more
    :param all_markers:
    :param cameras:
    :return:
    """
    # TODO: Add check to make sure all markers are the same length
    # Triangulate each marker
    if len(all_markers) > 0:
        zipped = zip(all_markers, cameras)
        A = {}
        for markers, cam in zipped:
            proj_mtx = cam.data.proj_mtx  # proj_mtx is 3 rows x 4 columns
            # Let proj_mtx = P, then pi is the i-th row of proj_mtx
            p1 = proj_mtx[0, :]
            p2 = proj_mtx[1, :]
            p3 = proj_mtx[2, :]

            for i in xrange(len(markers)):
                if i not in A:
                    A[i] = []
                u, v = markers[i].pos
                A[i].append(u * p3 - p1)
                A[i].append(v * p3 - p2)
                A[i].append(u * p2 - v * p1)

        points3d = []
        for key, a in A.iteritems():
            # Calculate best point
            a = np.array(a)
            u, d, vt = np.linalg.svd(a)
            X = vt[-1, :3] / vt[-1, 3]  # normalize
            points3d.append(X)

        return np.array(points3d)
    else:
        return False
