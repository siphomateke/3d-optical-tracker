# region Image processing
BLUR_KSIZE = (7, 7)
BLUR_SIGMA = 1.4

# CLAHE : Contrast Limited Adaptive Histogram Equalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE_RATIO = 100  # average image size is divided by this

CANNY_LOWER = 50
CANNY_UPPER = 150
# endregion

# region Contours filtering
CONTOUR_MIN_AREA = 10
# ratio of parent to child area
CONTOUR_MIN_CHILD_AREA_RATIO = 1.6
CONTOUR_MAX_CHILD_AREA_RATIO = 2.2

CONTOUR_MAX_ASPECT_RATIO = 3
CONTOUR_MAX_CENTER_DIST_RATIO = 4.0  # (center_dist < (major + minor) / ratio)
CONTOUR_MAJOR2RADIUS_RATIO = 2.1  # ratio of the ellipse major axis to circle radius (major <= radius * ratio)
CONTOUR_MINOR2RADIUS_RATIO = 2.1  # ratio of the ellipse minor axis to circle radius (minor <= radius * ratio)

CONTOUR_ELLIPSE_CONTOUR_DIFF = 0.1
# endregion

# region Markers
MAX_MARKER_DIST = 2 # maximum distance between two markers for them to be considered the same
# endregion
