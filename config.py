import numpy as np
import os

class GlobalConfig:
    # whether to show debug information
    DEBUG = False
    INFO = False
    HIDDEN = True

    # dataset path
    DATASET_EN_PATH = os.path.join("datasets", "en")
    DATASET_ZH_PATH = os.path.join("datasets", "zh")
    DATASET_EN_TOP_PATH = "en"
    DATASET_ZH_TOP_PATH = "zh"

    # classifier model path
    SVM_EN_PATH = os.path.join("models", "svm_en.dat")
    SVM_ZH_PATH = os.path.join("models", "svm_zh.dat")


class GUIConfig:
    ORIGINAL_SHOW_MAX_HEIGHT = 600
    ORIGINAL_SHOW_MAX_WIDTH = 600
    RESULT_SHOW_MAX_HEIGHT = 300
    RESULT_SHOW_MAX_WIDTH = 300
    VISUALIZATION_SHOW_MAX_HEIGHT = 150
    VISUALIZATION_SHOW_MAX_WIDTH = 150


class ClassificationConfig:
    # classification image size
    IMAGE_SIZE = 20

    # deskew hyperparameters
    DESKEW_EPSILON = 1e-2

    # HOG hyperparameters
    BIN_NUM = 32

    # Hellinger hyperparameters
    HELLINGER_EPSILON = 1e-7

    # SVM hyperparameters
    EN_SVM_C = 1.0
    EN_SVM_GAMMA = 0.5
    ZH_SVM_C = 5.0
    ZH_SVM_GAMMA = 1.0

    # labels
    LABELS_EN = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
    LABELS_EN_INVERTED = {v: k for k, v in LABELS_EN.items()}
    LABELS_ZH = {'chuan': [0, '川'], 'e': [1, '鄂'], 'gan': [2, '赣'], 'gan2': [3, '甘'], 'gui': [4, '贵'], 'gui2': [5, '桂'], 'hei': [6, '黑'], 'hu': [7, '沪'], 'ji': [8, '冀'], 'ji2': [9, '吉'], 'jin': [10, '津'], 'jin2': [11, '晋'], 'jing': [12, '京'], 'liao': [13, '辽'], 'lu': [14, '鲁'], 'meng': [15, '蒙'], 'min': [16, '闵'], 'ning': [17, '宁'], 'qing': [18, '青'], 'qiong': [19, '琼'], 'shan': [20, '陕'], 'su': [21, '苏'], 'wan': [22, '皖'], 'xiang': [23, '湘'], 'xin': [24, '新'], 'yu': [25, '豫'], 'yu2': [26, '渝'], 'yue': [27, '粤'], 'yun': [28, '云'], 'zang': [29, '藏'], 'zhe': [30, '浙']}
    LABELS_ZH_INVERTED = {v[0]: v[1] for k, v in LABELS_ZH.items()}


class DetectionConfig:
    # common preprocess
    MAX_LENGTH = 1000

    # color_shape preprocess
    COLOR_SHAPE_MEDIAN_BLUR_SIZE = 7

    # shape_color preprocess
    SHAPE_COLOR_MEDIAN_BLUR_SIZE = 3
    SHAPE_COLOR_PADDING_SIZE = 20

    # blue based on RGB color space, auxiliary HSV
    BLUE_LOW_BGR = (100, 0, 0)
    BLUE_UPPER_BGR = (255, 140, 140)
    BLUE_LOW_HSV = (100, 100, 0)
    BLUE_UPPER_HSV = (124, 255, 255)
    # green based on HSV color space, auxiliary BGR
    GREEN_LOW_BGR = (0, 100, 0)
    GREEN_UPPER_BGR = (140, 255, 140)
    GREEN_LOW_HSV = (45, 35, 0)
    GREEN_UPPER_HSV = (90, 255, 255)

    # color bounding box other hyperparameters
    COLOR_ROI_DILATE_KERNEL_SIZE = (15, 3)
    COLOR_BOUNDING_BOX_THRESHOLD = 0.8
    COLOR_BOUNDING_BOX_RECTIFY_THRESHOLD = 0.9

    # blue bounding box hyperparameters
    BLUE_BOUNDING_BOX_RATIO_HIGH = 4.5
    BLUE_BOUNDING_BOX_RATIO_LOW = 2
    BLUE_BOUNDING_BOX_AREA_THRESHOLD = 0.1 * MAX_LENGTH * 0.05 * MAX_LENGTH
    BLUE_BOUNDING_BOX_WHOLE_AREA_RATIO = 0.95

    # green bounding box hyperparameters
    GREEN_BOUNDING_BOX_RATIO_HIGH = 6
    GREEN_BOUNDING_BOX_RATIO_LOW = 3
    GREEN_BOUNDING_BOX_AREA_THRESHOLD = 0.15 * MAX_LENGTH * 0.05 * MAX_LENGTH
    GREEN_BOUNDING_BOX_WHOLE_AREA_RATIO = 0.85
    GREEN_BOUNDING_BOX_PADDING_RATIO = 1/6

    # shape bounding box hyperparameters
    ORB_FEATURES_NUM = 500
    ORB_SCALE_FACTOR = 1.5
    SHAPE_ROI_CHARACTER_FEATURE_RANGE = 10
    SHAPE_ROI_DILATE_KERNEL_SIZE = (20, 5)
    SHAPE_ROI_ERODE_KERNEL_SIZE = (10, 10)
    SHAPE_BOUNDING_BOX_THRESHOLD = 0.5
    SHAPE_BOUNDING_BOX_RATIO_HIGH = 6
    SHAPE_BOUNDING_BOX_RATIO_LOW = 2
    SHAPE_BOUNDING_BOX_AREA_THRESHOLD = 0.15 * MAX_LENGTH * 0.05 * MAX_LENGTH
    SHAPE_BOUNDING_BOX_COUNT_IN_THRESHOLD = 30

    # shape rectification hyperparameters
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    HOUGH_RHO = 1.0
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 30
    LINE_THETA_OFFSET = np.pi / 12
    BLUE_LONG = 440
    BLUE_SHORT = 140
    GREEN_LONG = 480
    GREEN_SHORT = 140

    # color rectification hyperparameters
    CLUSTER_K = 3
    CLUSTER_ITERATIONS = 10
    CLUSTER_EPSILON = 1.0
    CLUSTER_ATTEMPTS = 10
    CLUSTER_ERODE_KERNEL_SIZE = (5, 5)
    CLUSTER_DILATE_KERNEL_SIZE = (15, 10)
    CENTER_OFFSET_RATIO = 10
    RECTANGLE_AREA_THRESHOLD = 0.5
    REGULAR_RECTANGLE_AREA_RATIO = 0.7
    FILTER_RECTANGLE_AREA_THRESHOLD = 0.95
    EPSILON1_DIVIDER = 40
    EPSILON2_DIVIDER = 10
    AVERAGE_LONG = (BLUE_LONG + GREEN_LONG) // 2
    AVERAGE_SHORT = (BLUE_SHORT + GREEN_SHORT) // 2

    # character preprocess
    WAVE_X_THRESHOLD_DIVIDER = 2
    WAVE_Y_THRESHOLD_DIVIDER = 10
    WAVE_LOW_GRANULARITY_BORDER = 2
    WAVE_HIGH_GRANULARITY_BORDER = 5
    WAVE_LOW_GRANULARITY_CHARACTER = 15
    WAVE_HIGH_GRANULARITY_CHARACTER = 30
    BLUE_CHARACTERS_NUM = 7
    GREEN_CHARACTERS_NUM = 8
    CHARACTER_PADDING_SIZE = 10

    # filter hyperparameters
    FILTER_OFFSET = MAX_LENGTH * 0.02
