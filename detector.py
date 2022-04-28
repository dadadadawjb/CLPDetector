import os
import time
import numpy as np
import cv2

from classifier import SVMWrapper, classify_preprocess
from utils import wave_analysis, digit2char, char2digit, MyThread
from config import GlobalConfig, DetectionConfig, ClassificationConfig

class ChineseLicensePlateDetector:
    def __init__(self) -> None:
        self.image = None                                   # a single BGR image
        
        # initialize intermediate results
        self.color_shape_feature = None                     # a single BGR image, actually concatenate BGR and HSV images
        self.color_shape_blue_mask = None                   # a single binary/gray image
        self.color_shape_green_mask = None                  # a single binary/gray image
        self.color_shape_blue_RoIs = []                     # a list of BGR images
        self.color_shape_green_RoIs = []                    # a list of BGR images
        self.color_shape_RoI_contours_image = None          # a single BGR image
        self.color_shape_blue_RoIs_features = []            # a list of BGR images, same length with self.color_shape_blue_RoIs
        self.color_shape_green_RoIs_features = []           # a list of BGR images, same length with self.color_shape_green_RoIs
        self.color_shape_blue_RoIs_rectified = []           # a list of BGR images, some elements may be None, same length with self.color_shape_blue_RoIs
        self.color_shape_green_RoIs_rectified = []          # a list of BGR images, some elements may be None, same length with self.color_shape_green_RoIs
        self.color_shape_blue_characters_images = []        # a list of (list of binary/gray images, index)s, while index denote the index of self.color_shape_blue_RoIs
        self.color_shape_green_characters_images = []       # a list of (list of binary/gray images, index)s, while index denote the index of self.color_shape_green_RoIs
        self.color_shape_blue_characters_results = []       # a list of (list of characters, index)s, while index denote the index of self.color_shape_blue_RoIs
        self.color_shape_green_characters_results = []      # a list of (list of characters, index)s, while index denote the index of self.color_shape_green_RoIs
        self.shape_color_feature = None                     # a single BGR image
        self.shape_color_mask = None                        # a single binary/gray image
        self.shape_color_RoIs = []                          # a list of BGR images
        self.shape_color_RoI_contours_image = None          # a single BGR image
        self.shape_color_RoIs_features = []                 # a list of binary/gray images, same length with self.shape_color_RoIs
        self.shape_color_RoIs_rectified = []                # a list of BGR images, some elements may be None, same length with self.shape_color_RoIs
        self.shape_color_blue_characters_images = []        # a list of (list of binary/gray images, index)s, while index denote the index of self.shape_color_RoIs
        self.shape_color_green_characters_images = []       # a list of (list of binary/gray images, index)s, while index denote the index of self.shape_color_RoIs
        self.shape_color_blue_characters_results = []       # a list of (list of characters, index)s, while index denote the index of self.shape_color_RoIs
        self.shape_color_green_characters_results = []      # a list of (list of characters, index)s, while index denote the index of self.shape_color_RoIs
        self.blue_characters_results = []                   # a list of (BGR image, list of characters)s
        self.green_characters_results = []                  # a list of (BGR image, list of characters)s

        # load classifier model
        self.model_en = SVMWrapper(ClassificationConfig.EN_SVM_C, ClassificationConfig.EN_SVM_GAMMA)
        self.model_zh = SVMWrapper(ClassificationConfig.ZH_SVM_C, ClassificationConfig.ZH_SVM_GAMMA)
        if os.path.exists(GlobalConfig.SVM_EN_PATH):
            self.model_en.load(GlobalConfig.SVM_EN_PATH)
        else:
            raise Exception(f"Cannot find classifier model {GlobalConfig.SVM_EN_PATH}, please train the model first")
        if os.path.exists(GlobalConfig.SVM_ZH_PATH):
            self.model_zh.load(GlobalConfig.SVM_ZH_PATH)
        else:
            raise Exception(f"Cannot find classifier model {GlobalConfig.SVM_ZH_PATH}, please train the model first")
    
    def set_image(self, image:np.ndarray) -> None:
        self.image = image
        
        # reset intermediate results
        self.color_shape_feature = None
        self.color_shape_blue_mask = None
        self.color_shape_green_mask = None
        self.color_shape_blue_RoIs = []
        self.color_shape_green_RoIs = []
        self.color_shape_RoI_contours_image = None
        self.color_shape_blue_RoIs_features = []
        self.color_shape_green_RoIs_features = []
        self.color_shape_blue_RoIs_rectified = []
        self.color_shape_green_RoIs_rectified = []
        self.color_shape_blue_characters_images = []
        self.color_shape_green_characters_images = []
        self.color_shape_blue_characters_results = []
        self.color_shape_green_characters_results = []
        self.shape_color_feature = None
        self.shape_color_mask = None
        self.shape_color_RoIs = []
        self.shape_color_RoI_contours_image = None
        self.shape_color_RoIs_features = []
        self.shape_color_RoIs_rectified = []
        self.shape_color_blue_characters_images = []
        self.shape_color_green_characters_images = []
        self.shape_color_blue_characters_results = []
        self.shape_color_green_characters_results = []
        self.blue_characters_results = []
        self.green_characters_results = []
    
    ##########---------- Bounding Box ----------##########
    
    def color_shape_preprocess(self) -> tuple:
        '''preprocess for color_shape_pipeline
        Params:
            self.image: original BGR image
        ----------
        Return:
            image: only resized original image
            image_bgr: preprocessed BGR image
            image_hsv: preprocessed HSV image
        '''
        # resize to fixed max size to finetune the hyperparameters
        hight, width = self.image.shape[:2]
        if width > hight:
            image = cv2.resize(self.image, (DetectionConfig.MAX_LENGTH, int(hight * DetectionConfig.MAX_LENGTH / width)), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(self.image, (int(width * DetectionConfig.MAX_LENGTH / hight), DetectionConfig.MAX_LENGTH), interpolation=cv2.INTER_AREA)
        
        # median blur to remove salt and pepper noise, also smooth the image
        image_bgr = cv2.medianBlur(image, DetectionConfig.COLOR_SHAPE_MEDIAN_BLUR_SIZE)

        # convert to HSV color space
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        return (image, image_bgr, image_hsv)
    
    def color_RoI_proposal(self, image_resized:np.ndarray, image_preprocessed_bgr:np.ndarray, image_preprocessed_hsv:np.ndarray) -> tuple:
        '''propose RoI according to license plate color
        Params:
            image_resized: resized original image
            image_preprocessed_bgr: preprocessed BGR image
            image_preprocessed_hsv: preprocessed HSV image
        ----------
        Return:
            blue_RoIs, green_RoIs: RoIs of blue and green license plates, 
                                    first image, second mask, third whether already rectified, fourth center
                                    when already rectified, the mask is None
        '''
        self.color_shape_feature = np.vstack((image_preprocessed_bgr, image_preprocessed_hsv))
        if GlobalConfig.DEBUG:
            cv2.imshow("color_feature", self.color_shape_feature)
            cv2.waitKey(0)
        
        # find color license plate in BGR color space
        blue_mask_bgr = cv2.inRange(image_preprocessed_bgr, DetectionConfig.BLUE_LOW_BGR, DetectionConfig.BLUE_UPPER_BGR)
        green_mask_bgr = cv2.inRange(image_preprocessed_bgr, DetectionConfig.GREEN_LOW_BGR, DetectionConfig.GREEN_UPPER_BGR)

        # find color license plate in HSV color space
        blue_mask_hsv = cv2.inRange(image_preprocessed_hsv, DetectionConfig.BLUE_LOW_HSV, DetectionConfig.BLUE_UPPER_HSV)
        green_mask_hsv = cv2.inRange(image_preprocessed_hsv, DetectionConfig.GREEN_LOW_HSV, DetectionConfig.GREEN_UPPER_HSV)

        # combine the mask
        blue_mask = blue_mask_bgr & blue_mask_hsv
        green_mask = green_mask_bgr & green_mask_hsv
        
        # dilate to fill the holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DetectionConfig.COLOR_ROI_DILATE_KERNEL_SIZE)
        blue_mask = cv2.dilate(blue_mask, kernel)
        green_mask = cv2.dilate(green_mask, kernel)
        self.color_shape_blue_mask = blue_mask
        self.color_shape_green_mask = green_mask
        if GlobalConfig.DEBUG:
            cv2.imshow("blue_mask", blue_mask)
            cv2.imshow("green_mask", green_mask)
            cv2.waitKey(0)

        # refine the mask by forming regular contours
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_bounding_boxes = []
        blue_RoIs = []
        for blue_contour in blue_contours:
            blue_rectangle = cv2.minAreaRect(blue_contour)
            blue_rectangle_center = blue_rectangle[0]
            blue_bounding_box = cv2.boxPoints(blue_rectangle)
            blue_bounding_box_width = int(blue_rectangle[1][0])
            blue_bounding_box_height = int(blue_rectangle[1][1])
            blue_bounding_box_long = max(blue_bounding_box_width, blue_bounding_box_height)
            blue_bounding_box_short = min(blue_bounding_box_width, blue_bounding_box_height)
            blue_bounding_box_ratio = blue_bounding_box_long / blue_bounding_box_short
            blue_contour_area = cv2.contourArea(blue_contour)
            blue_bounding_box_area = cv2.contourArea(blue_bounding_box)
            image_area = image_resized.shape[0] * image_resized.shape[1]

            # regular contour
            condition1 = blue_contour_area > blue_bounding_box_area * DetectionConfig.COLOR_BOUNDING_BOX_THRESHOLD
            # satisfy blue license plate ratio
            condition2 = blue_bounding_box_ratio < DetectionConfig.BLUE_BOUNDING_BOX_RATIO_HIGH and blue_bounding_box_ratio > DetectionConfig.BLUE_BOUNDING_BOX_RATIO_LOW
            # area not too small
            condition3 = blue_bounding_box_area > DetectionConfig.BLUE_BOUNDING_BOX_AREA_THRESHOLD
            # shortcut for already license plate image
            condition4 = blue_bounding_box_area > image_area * DetectionConfig.BLUE_BOUNDING_BOX_WHOLE_AREA_RATIO
            if condition4:
                blue_bounding_boxes.append(np.int0(blue_bounding_box))
                blue_RoIs.append((image_resized, None, True, blue_rectangle_center))
                self.color_shape_blue_RoIs.append(image_resized)
                if GlobalConfig.DEBUG:
                    cv2.imshow("blue_RoI", image_resized)
                    cv2.waitKey(0)
            elif condition1 and condition2 and condition3:
                blue_bounding_boxes.append(np.int0(blue_bounding_box))

                # mask the contour
                blue_contour_mask = np.zeros(image_resized.shape[:2], dtype=np.uint8)
                blue_contour_mask = cv2.drawContours(blue_contour_mask, [blue_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

                # crop the bounding box
                src_pts = blue_bounding_box.astype("float32")
                dst_pts = np.array([[0, blue_bounding_box_height-1],
                                    [0, 0],
                                    [blue_bounding_box_width-1, 0],
                                    [blue_bounding_box_width-1, blue_bounding_box_height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image_resized, M, (blue_bounding_box_width, blue_bounding_box_height))
                warped_mask = cv2.warpPerspective(blue_contour_mask, M, (blue_bounding_box_width, blue_bounding_box_height))
                if blue_bounding_box_width < blue_bounding_box_height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    warped_mask = cv2.rotate(warped_mask, cv2.ROTATE_90_CLOCKWISE)
                rectifed = blue_contour_area > blue_bounding_box_area * DetectionConfig.COLOR_BOUNDING_BOX_RECTIFY_THRESHOLD
                blue_RoIs.append((warped, warped_mask, rectifed, blue_rectangle_center))
                self.color_shape_blue_RoIs.append(warped)
                if GlobalConfig.DEBUG:
                    cv2.imshow("blue_RoI", warped)
                    cv2.waitKey(0)
            else:
                continue
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_bounding_boxes = []
        green_RoIs = []
        for green_contour in green_contours:
            green_rectangle = cv2.minAreaRect(green_contour)
            green_rectangle_center = green_rectangle[0]
            # deal with green license plates green incomplete
            green_rectangle = [[green_rectangle[0][0], green_rectangle[0][1]], [green_rectangle[1][0], green_rectangle[1][1]], green_rectangle[2]]
            if green_rectangle[1][0] < green_rectangle[1][1]:
                green_rectangle[1][0] = (1 + DetectionConfig.GREEN_BOUNDING_BOX_PADDING_RATIO) * green_rectangle[1][0]
            else:
                green_rectangle[1][1] = (1 + DetectionConfig.GREEN_BOUNDING_BOX_PADDING_RATIO) * green_rectangle[1][1]
            green_bounding_box = cv2.boxPoints(green_rectangle)
            green_bounding_box_width = int(green_rectangle[1][0])
            green_bounding_box_height = int(green_rectangle[1][1])
            green_bounding_box_long = max(green_bounding_box_width, green_bounding_box_height)
            green_bounding_box_short = min(green_bounding_box_width, green_bounding_box_height)
            green_bounding_box_ratio = green_bounding_box_long / green_bounding_box_short
            green_contour_area = cv2.contourArea(green_contour)
            green_bounding_box_area = cv2.contourArea(green_bounding_box)
            image_area = image_resized.shape[0] * image_resized.shape[1]

            # regular contour
            condition1 = green_contour_area > green_bounding_box_area * DetectionConfig.COLOR_BOUNDING_BOX_THRESHOLD / (1 + DetectionConfig.GREEN_BOUNDING_BOX_PADDING_RATIO)
            # satisfy green license plate ratio
            condition2 = green_bounding_box_ratio < DetectionConfig.GREEN_BOUNDING_BOX_RATIO_HIGH and green_bounding_box_ratio > DetectionConfig.GREEN_BOUNDING_BOX_RATIO_LOW
            # area not too small
            condition3 = green_bounding_box_area > DetectionConfig.GREEN_BOUNDING_BOX_AREA_THRESHOLD
            # shortcut for already license plate image
            condition4 = green_bounding_box_area > image_area * DetectionConfig.GREEN_BOUNDING_BOX_WHOLE_AREA_RATIO
            if condition4:
                green_bounding_boxes.append(np.int0(green_bounding_box))
                green_RoIs.append((image_resized, None, True, green_rectangle_center))
                self.color_shape_green_RoIs.append(image_resized)
                if GlobalConfig.DEBUG:
                    cv2.imshow("green_RoI", image_resized)
                    cv2.waitKey(0)
            elif condition1 and condition2 and condition3:
                green_bounding_boxes.append(np.int0(green_bounding_box))

                # mask the contour
                green_contour_mask = np.zeros(image_resized.shape[:2], dtype=np.uint8)
                green_contour_mask = cv2.drawContours(green_contour_mask, [green_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

                # crop the bounding box
                src_pts = green_bounding_box.astype("float32")
                dst_pts = np.array([[0, green_bounding_box_height-1],
                                    [0, 0],
                                    [green_bounding_box_width-1, 0],
                                    [green_bounding_box_width-1, green_bounding_box_height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image_resized, M, (green_bounding_box_width, green_bounding_box_height))
                warped_mask = cv2.warpPerspective(green_contour_mask, M, (green_bounding_box_width, green_bounding_box_height))
                if green_bounding_box_width < green_bounding_box_height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    warped_mask = cv2.rotate(warped_mask, cv2.ROTATE_90_CLOCKWISE)
                rectifed = green_contour_area > green_bounding_box_area * DetectionConfig.COLOR_BOUNDING_BOX_RECTIFY_THRESHOLD / (1 + DetectionConfig.GREEN_BOUNDING_BOX_PADDING_RATIO)
                green_RoIs.append((warped, warped_mask, rectifed, green_rectangle_center))
                self.color_shape_green_RoIs.append(warped)
                if GlobalConfig.DEBUG:
                    cv2.imshow("green_RoI", warped)
                    cv2.waitKey(0)
            else:
                continue
        
        if GlobalConfig.DEBUG:
            print("color RoIs proposal num:", len(blue_RoIs), len(green_RoIs))
        color_RoI_contours = image_resized.copy()
        cv2.drawContours(color_RoI_contours, blue_contours, contourIdx=-1, color=(0,0,255), thickness=3)
        cv2.drawContours(color_RoI_contours, green_contours, contourIdx=-1, color=(0,0,255), thickness=3)
        cv2.drawContours(color_RoI_contours, [blue_bounding_box for blue_bounding_box in blue_bounding_boxes], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.drawContours(color_RoI_contours, [green_bounding_box for green_bounding_box in green_bounding_boxes], contourIdx=-1, color=(0,255,0), thickness=3)
        self.color_shape_RoI_contours_image = color_RoI_contours
        if GlobalConfig.DEBUG:
            cv2.imshow('color_RoI_contours', color_RoI_contours)
            cv2.waitKey(0)

        return (blue_RoIs, green_RoIs)
    
    def shape_rectification(self, blue_RoIs:list, green_RoIs:list) -> tuple:
        '''rectify bounding boxes from color RoI proposal by shape
        Params:
            blue_RoIs: list of RoIs based on blue
            green_RoIs: list of RoIs based on green
        ----------
        Return:
            blue_plates, green_plates: list of blue and green license plates, 
                            first image, second center, third index
        '''
        blue_plates = []
        for i, blue_RoIs_item in enumerate(blue_RoIs):
            blue_RoI, blue_RoI_mask, rectified, blue_center = blue_RoIs_item
            height, width = blue_RoI.shape[:2]
            if not rectified:
                # find the distort edge
                edges = cv2.Canny(blue_RoI_mask, DetectionConfig.CANNY_THRESHOLD1, DetectionConfig.CANNY_THRESHOLD2, apertureSize=3)
                lines = cv2.HoughLines(edges, DetectionConfig.HOUGH_RHO, DetectionConfig.HOUGH_THETA, DetectionConfig.HOUGH_THRESHOLD)
                flag = False
                for line in lines:
                    rho, theta = line[0]
                    # rectify according to distort edge
                    if theta - 0 > DetectionConfig.LINE_THETA_OFFSET and np.pi/2 - theta > DetectionConfig.LINE_THETA_OFFSET:
                        blue_RoI_edge = blue_RoI.copy()
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(blue_RoI_edge, (x1,y1), (x2,y2), (0,0,255), 2)
                        self.color_shape_blue_RoIs_features.append(blue_RoI_edge)
                        if GlobalConfig.DEBUG:
                            cv2.imshow("blue_RoI_feature", blue_RoI_edge)
                            cv2.waitKey(0)
                        
                        offset = np.tan(theta) * height
                        src_pts = np.array([[width, 0], [offset, 0], [0, height], [width-offset, height]], dtype="float32")
                        dst_pts = np.array([[DetectionConfig.BLUE_LONG, 0], [0, 0], [0, DetectionConfig.BLUE_SHORT], [DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT]], dtype="float32")
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        blue_plate = cv2.warpPerspective(blue_RoI, M, (DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT))
                        blue_plates.append((blue_plate, blue_center, i))
                        flag = True
                        self.color_shape_blue_RoIs_rectified.append(blue_plate)
                        if GlobalConfig.DEBUG:
                            cv2.imshow('blue_RoI_rectified', blue_plate)
                            cv2.waitKey(0)
                        break
                    elif np.pi - theta > DetectionConfig.LINE_THETA_OFFSET and theta - np.pi/2 > DetectionConfig.LINE_THETA_OFFSET:
                        blue_RoI_edge = blue_RoI.copy()
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(blue_RoI_edge, (x1,y1), (x2,y2), (0,0,255), 2)
                        self.color_shape_blue_RoIs_features.append(blue_RoI_edge)
                        if GlobalConfig.DEBUG:
                            cv2.imshow("blue_RoI_feature", blue_RoI_edge)
                            cv2.waitKey(0)

                        offset = -np.tan(theta) * height
                        src_pts = np.array([[0, 0], [offset, height], [width, height], [width-offset, 0]], dtype="float32")
                        dst_pts = np.array([[0, 0], [0, DetectionConfig.BLUE_SHORT], [DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT], [DetectionConfig.BLUE_LONG, 0]], dtype="float32")
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        blue_plate = cv2.warpPerspective(blue_RoI, M, (DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT))
                        blue_plates.append((blue_plate, blue_center, i))
                        flag = True
                        self.color_shape_blue_RoIs_rectified.append(blue_plate)
                        if GlobalConfig.DEBUG:
                            cv2.imshow('blue_RoI_rectified', blue_plate)
                            cv2.waitKey(0)
                        break
                    else:
                        # not distort edge, skip
                        continue
                if not flag:
                    self.color_shape_blue_RoIs_features.append(blue_RoI)
                    self.color_shape_blue_RoIs_rectified.append(None)
                    if GlobalConfig.DEBUG:
                        print(f"{i}th blue RoI rectification failed")
            else:
                src_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype="float32")
                dst_pts = np.array([[0, 0], [0, DetectionConfig.BLUE_SHORT], [DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT], [DetectionConfig.BLUE_LONG, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                blue_plate = cv2.warpPerspective(blue_RoI, M, (DetectionConfig.BLUE_LONG, DetectionConfig.BLUE_SHORT))
                blue_plates.append((blue_plate, blue_center, i))
                self.color_shape_blue_RoIs_features.append(blue_RoI)
                self.color_shape_blue_RoIs_rectified.append(blue_plate)
                if GlobalConfig.DEBUG:
                    cv2.imshow('blue_RoI_rectified', blue_plate)
                    cv2.waitKey(0)

        green_plates = []
        for i, green_RoIs_item in enumerate(green_RoIs):
            green_RoI, green_RoI_mask, rectified, green_center = green_RoIs_item
            height, width = green_RoI.shape[:2]
            if not rectified:
                # find the distort edge
                edges = cv2.Canny(green_RoI_mask, DetectionConfig.CANNY_THRESHOLD1, DetectionConfig.CANNY_THRESHOLD2, apertureSize=3)
                lines = cv2.HoughLines(edges, DetectionConfig.HOUGH_RHO, DetectionConfig.HOUGH_THETA, DetectionConfig.HOUGH_THRESHOLD)
                flag = False
                for line in lines:
                    rho, theta = line[0]
                    # rectify according to distort edge
                    if theta - 0 > DetectionConfig.LINE_THETA_OFFSET and np.pi/2 - theta > DetectionConfig.LINE_THETA_OFFSET:
                        green_RoI_edge = green_RoI.copy()
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(green_RoI_edge, (x1,y1), (x2,y2), (0,0,255), 2)
                        self.color_shape_green_RoIs_features.append(green_RoI_edge)
                        if GlobalConfig.DEBUG:
                            cv2.imshow("green_RoI_feature", green_RoI_edge)
                            cv2.waitKey(0)
                        
                        offset = np.tan(theta) * height
                        src_pts = np.array([[width, 0], [offset, 0], [0, height], [width-offset, height]], dtype="float32")
                        dst_pts = np.array([[DetectionConfig.GREEN_LONG, 0], [0, 0], [0, DetectionConfig.GREEN_SHORT], [DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT]], dtype="float32")
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        green_plate = cv2.warpPerspective(green_RoI, M, (DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT))
                        green_plates.append((green_plate, green_center, i))
                        flag = True
                        self.color_shape_green_RoIs_rectified.append(green_plate)
                        if GlobalConfig.DEBUG:
                            cv2.imshow('green_RoI_rectified', green_plate)
                            cv2.waitKey(0)
                        break
                    elif np.pi - theta > DetectionConfig.LINE_THETA_OFFSET and theta - np.pi/2 > DetectionConfig.LINE_THETA_OFFSET:
                        green_RoI_edge = green_RoI.copy()
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(green_RoI_edge, (x1,y1), (x2,y2), (0,0,255), 2)
                        self.color_shape_green_RoIs_features.append(green_RoI_edge)
                        if GlobalConfig.DEBUG:
                            cv2.imshow("green_RoI_feature", green_RoI_edge)
                            cv2.waitKey(0)
                        
                        offset = -np.tan(theta) * height
                        src_pts = np.array([[0, 0], [offset, height], [width, height], [width-offset, 0]], dtype="float32")
                        dst_pts = np.array([[0, 0], [0, DetectionConfig.GREEN_SHORT], [DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT], [DetectionConfig.GREEN_LONG, 0]], dtype="float32")
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        green_plate = cv2.warpPerspective(green_RoI, M, (DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT))
                        green_plates.append((green_plate, green_center, i))
                        flag = True
                        self.color_shape_green_RoIs_rectified.append(green_plate)
                        if GlobalConfig.DEBUG:
                            cv2.imshow('green_RoI_rectified', green_plate)
                            cv2.waitKey(0)
                        break
                    else:
                        # not distort edge, skip
                        continue
                if not flag:
                    self.color_shape_green_RoIs_features.append(green_RoI)
                    self.color_shape_green_RoIs_rectified.append(None)
                    if GlobalConfig.DEBUG:
                        print(f"{i}th green RoI rectification failed")
            else:
                src_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype="float32")
                dst_pts = np.array([[0, 0], [0, DetectionConfig.GREEN_SHORT], [DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT], [DetectionConfig.GREEN_LONG, 0]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                green_plate = cv2.warpPerspective(green_RoI, M, (DetectionConfig.GREEN_LONG, DetectionConfig.GREEN_SHORT))
                green_plates.append((green_plate, green_center, i))
                self.color_shape_green_RoIs_features.append(green_RoI)
                self.color_shape_green_RoIs_rectified.append(green_plate)
                if GlobalConfig.DEBUG:
                    cv2.imshow('green_RoI_rectified', green_plate)
                    cv2.waitKey(0)

        if GlobalConfig.DEBUG:
            print("shape rectification num:", len(blue_plates), len(green_plates))

        return (blue_plates, green_plates)
    
    def shape_color_preprocess(self) -> tuple:
        '''preprocess for shape_color_pipeline
        Params:
            self.image: original BGR image
        ----------
        Return:
            image: only resized original image
            image_gray: preprocessed gray image
        '''
        # resize to fixed max size to finetune the hyperparameters
        hight, width = self.image.shape[:2]
        if width > hight:
            image = cv2.resize(self.image, (DetectionConfig.MAX_LENGTH, int(hight * DetectionConfig.MAX_LENGTH / width)), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(self.image, (int(width * DetectionConfig.MAX_LENGTH / hight), DetectionConfig.MAX_LENGTH), interpolation=cv2.INTER_AREA)
        
        # median blur to remove salt and pepper noise, not too smooth the image since we need the detailed shape
        image_blur = cv2.medianBlur(image, DetectionConfig.SHAPE_COLOR_MEDIAN_BLUR_SIZE)

        # padding to avoid border effect
        image_padding = cv2.copyMakeBorder(image_blur, DetectionConfig.SHAPE_COLOR_PADDING_SIZE, DetectionConfig.SHAPE_COLOR_PADDING_SIZE, DetectionConfig.SHAPE_COLOR_PADDING_SIZE, DetectionConfig.SHAPE_COLOR_PADDING_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # convert to gray image
        image_gray = cv2.cvtColor(image_padding, cv2.COLOR_BGR2GRAY)

        return (image, image_gray)

    def shape_RoI_proposal(self, image_resized:np.ndarray, image_preprocessed_gray:np.ndarray) -> list:
        '''propose RoI according to license plate shape
        Params:
            image_resized: resized original image
            image_preprocessed_gray: preprocessed gray image
        ----------
        Return:
            shape_RoIs: RoIs of license plates, 
                        first image, second center
        ----------
        Note:
            TODO: possible improvement: scale-invariant
        '''
        # local shape feature extraction
        orb = cv2.ORB_create(nfeatures=DetectionConfig.ORB_FEATURES_NUM, scaleFactor=DetectionConfig.ORB_SCALE_FACTOR)
        keypoints = orb.detect(image_preprocessed_gray, None)
        self.shape_color_feature = cv2.drawKeypoints(image_preprocessed_gray, keypoints, None)
        if GlobalConfig.DEBUG:
            cv2.imshow('shape_feature', self.shape_color_feature)
            cv2.waitKey(0)
        
        # create threshold image
        shape_mask = np.zeros(image_preprocessed_gray.shape, np.uint8)
        for keypoint in keypoints:
            x, y = keypoint.pt
            cv2.rectangle(shape_mask, (int(x) - DetectionConfig.SHAPE_ROI_CHARACTER_FEATURE_RANGE, int(y) - DetectionConfig.SHAPE_ROI_CHARACTER_FEATURE_RANGE), (int(x) + DetectionConfig.SHAPE_ROI_CHARACTER_FEATURE_RANGE, int(y) + DetectionConfig.SHAPE_ROI_CHARACTER_FEATURE_RANGE), 255, -1)

        # dilate and erode to fill the gaps between characters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DetectionConfig.SHAPE_ROI_DILATE_KERNEL_SIZE)
        shape_mask = cv2.dilate(shape_mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DetectionConfig.SHAPE_ROI_ERODE_KERNEL_SIZE)
        shape_mask = cv2.erode(shape_mask, kernel)
        self.shape_color_mask = shape_mask
        if GlobalConfig.DEBUG:
            cv2.imshow('shape_mask', shape_mask)
            cv2.waitKey(0)

        # find contours
        shape_contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_contours = [shape_contour - np.array([[DetectionConfig.SHAPE_COLOR_PADDING_SIZE, DetectionConfig.SHAPE_COLOR_PADDING_SIZE]]) for shape_contour in shape_contours]         # notice the padding
        shape_bounding_boxes = []
        shape_RoIs = []
        for shape_contour in shape_contours:
            shape_rectangle = cv2.minAreaRect(shape_contour)
            shape_rectangle_center = shape_rectangle[0]
            shape_bounding_box = cv2.boxPoints(shape_rectangle)
            shape_bounding_box_width = int(shape_rectangle[1][0])
            shape_bounding_box_height = int(shape_rectangle[1][1])
            shape_bounding_box_long = max(shape_bounding_box_width, shape_bounding_box_height)
            shape_bounding_box_short = min(shape_bounding_box_width, shape_bounding_box_height)
            shape_bounding_box_ratio = shape_bounding_box_long / shape_bounding_box_short
            shape_contour_area = cv2.contourArea(shape_contour)
            shape_bounding_box_area = cv2.contourArea(shape_bounding_box)
            count_in = 0
            for keypoint in keypoints:
                if cv2.pointPolygonTest(shape_contour, keypoint.pt, False) >= 0:
                    count_in += 1
            
            # regular contour
            condition1 = shape_contour_area > shape_bounding_box_area * DetectionConfig.SHAPE_BOUNDING_BOX_THRESHOLD
            # satisfy license plate ratio
            condition2 = shape_bounding_box_ratio < DetectionConfig.SHAPE_BOUNDING_BOX_RATIO_HIGH and shape_bounding_box_ratio > DetectionConfig.SHAPE_BOUNDING_BOX_RATIO_LOW
            # area not too small
            condition3 = shape_bounding_box_area > DetectionConfig.SHAPE_BOUNDING_BOX_AREA_THRESHOLD
            # texture rich enough
            condition4 = count_in > DetectionConfig.SHAPE_BOUNDING_BOX_COUNT_IN_THRESHOLD
            if condition1 and condition2 and condition3 and condition4:
                shape_bounding_boxes.append(np.int0(shape_bounding_box))

                # crop the bounding box
                src_pts = shape_bounding_box.astype("float32")
                dst_pts = np.array([[0, shape_bounding_box_height-1], 
                                    [0, 0], 
                                    [shape_bounding_box_width-1, 0], 
                                    [shape_bounding_box_width-1, shape_bounding_box_height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image_resized, M, (shape_bounding_box_width, shape_bounding_box_height))
                if shape_bounding_box_width < shape_bounding_box_height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                shape_RoIs.append((warped, shape_rectangle_center))
                self.shape_color_RoIs.append(warped)
                if GlobalConfig.DEBUG:
                    cv2.imshow("shape_RoI", warped)
                    cv2.waitKey(0)
        
        if GlobalConfig.DEBUG:
            print("shape RoIs proposal num:", len(shape_RoIs))
        shape_RoI_contours = image_resized.copy()
        cv2.drawContours(shape_RoI_contours, shape_contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
        cv2.drawContours(shape_RoI_contours, [shape_bounding_box for shape_bounding_box in shape_bounding_boxes], contourIdx=-1, color=(0, 255, 0), thickness=3)
        self.shape_color_RoI_contours_image = shape_RoI_contours
        if GlobalConfig.DEBUG:
            cv2.imshow('shape_RoI_contours', shape_RoI_contours)
            cv2.waitKey(0)

        return shape_RoIs
    
    def color_rectification(self, shape_RoIs:list) -> list:
        '''rectify bounding boxes from shape RoI proposal by color
        Params:
            shape_RoIs: list of RoIs based on shape
        ----------
        Return:
            shape_plates: list of license plates
                            first image, second center, third index
        '''
        shape_plates = []
        for i, shape_RoIs_item in enumerate(shape_RoIs):
            shape_RoI, shape_center = shape_RoIs_item
            height, width = shape_RoI.shape[:2]
            shape_RoI_blur = cv2.medianBlur(shape_RoI, DetectionConfig.SHAPE_COLOR_MEDIAN_BLUR_SIZE)

            # clustering
            clustering_data = np.float32(shape_RoI_blur.reshape((-1, 3)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, DetectionConfig.CLUSTER_ITERATIONS, DetectionConfig.CLUSTER_EPSILON)
            _compactness, labels, _centers = cv2.kmeans(clustering_data, DetectionConfig.CLUSTER_K, None, criteria, DetectionConfig.CLUSTER_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
            A = np.zeros(clustering_data.shape[0], dtype="uint8")
            A[labels.ravel() == 0] = 255
            A = A.reshape(shape_RoI.shape[:2])
            B = np.zeros(clustering_data.shape[0], dtype="uint8")
            B[labels.ravel() == 1] = 255
            B = B.reshape(shape_RoI.shape[:2])
            C = np.zeros(clustering_data.shape[0], dtype="uint8")
            C[labels.ravel() == 2] = 255
            C = C.reshape(shape_RoI.shape[:2])
            
            # erode and dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DetectionConfig.CLUSTER_ERODE_KERNEL_SIZE)
            A = cv2.erode(A, kernel)
            B = cv2.erode(B, kernel)
            C = cv2.erode(C, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DetectionConfig.CLUSTER_DILATE_KERNEL_SIZE)
            A = cv2.dilate(A, kernel)
            B = cv2.dilate(B, kernel)
            C = cv2.dilate(C, kernel)
            shape_RoI_clustering = np.vstack((A, B, C))
            self.shape_color_RoIs_features.append(shape_RoI_clustering)
            if GlobalConfig.DEBUG:
                cv2.imshow("shape_RoI_feature", shape_RoI_clustering)
                cv2.waitKey(0)
            
            # select the right region
            A_contours, _ = cv2.findContours(A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            B_contours, _ = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            C_contours, _ = cv2.findContours(C, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(A_contours) + list(B_contours) + list(C_contours)
            correct_contour = None
            correct_rectangle = None
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rectangle_center_X = x + w/2
                rectangle_center_Y = y + h/2
                rectangle_area = w * h
                contour_area = cv2.contourArea(contour)
                
                # license plate should lie around the center of ROI
                condition1 = (abs(rectangle_center_X - width/2) < width/DetectionConfig.CENTER_OFFSET_RATIO) and (abs(rectangle_center_Y - height/2) < height/DetectionConfig.CENTER_OFFSET_RATIO)
                # area large enough
                condition2 = rectangle_area > width * height * DetectionConfig.RECTANGLE_AREA_THRESHOLD
                # regular rectangle
                condition3 = contour_area > rectangle_area * DetectionConfig.REGULAR_RECTANGLE_AREA_RATIO
                # filter background rectangle
                condition4 = rectangle_area < width * height * DetectionConfig.FILTER_RECTANGLE_AREA_THRESHOLD
                if condition1 and condition2 and condition3 and condition4:
                    correct_contour = contour
                    correct_rectangle = (x, y, w, h)
                    break
            if correct_contour is None:
                self.shape_color_RoIs_rectified.append(None)
                if GlobalConfig.DEBUG:
                    print(f"{i}th shape RoI rectification failed")
                continue
            
            # rectify according to selected region
            epsilon1 = height / DetectionConfig.EPSILON1_DIVIDER
            epsilon2 = height / DetectionConfig.EPSILON2_DIVIDER
            up_y_range = [correct_rectangle[1]+epsilon1, correct_rectangle[1]+epsilon1+epsilon2]
            up_left_point = [width, height]
            up_right_point = [0, 0]
            down_y_range = [correct_rectangle[1]+correct_rectangle[3]-epsilon1-epsilon2, correct_rectangle[1]+correct_rectangle[3]-epsilon1]
            down_left_point = [width, height]
            down_right_point = [0, 0]
            for correct_contour_point in correct_contour:
                correct_contour_point = correct_contour_point[0]
                if correct_contour_point[1] >= up_y_range[0] and correct_contour_point[1] <= up_y_range[1] and correct_contour_point[0] < up_left_point[0]:
                    up_left_point = correct_contour_point
                if correct_contour_point[1] >= up_y_range[0] and correct_contour_point[1] <= up_y_range[1] and correct_contour_point[0] > up_right_point[0]:
                    up_right_point = correct_contour_point
                if correct_contour_point[1] >= down_y_range[0] and correct_contour_point[1] <= down_y_range[1] and correct_contour_point[0] < down_left_point[0]:
                    down_left_point = correct_contour_point
                if correct_contour_point[1] >= down_y_range[0] and correct_contour_point[1] <= down_y_range[1] and correct_contour_point[0] > down_right_point[0]:
                    down_right_point = correct_contour_point
            
            src_pts = np.array([up_left_point, down_left_point, down_right_point, up_right_point], dtype="float32")
            dst_pts = np.array([[0, 0], [0, DetectionConfig.AVERAGE_SHORT], [DetectionConfig.AVERAGE_LONG, DetectionConfig.AVERAGE_SHORT], [DetectionConfig.AVERAGE_LONG, 0]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            shape_plate = cv2.warpPerspective(shape_RoI, M, (DetectionConfig.AVERAGE_LONG, DetectionConfig.AVERAGE_SHORT))
            shape_plates.append((shape_plate, shape_center, i))
            self.shape_color_RoIs_rectified.append(shape_plate)
            if GlobalConfig.DEBUG:
                cv2.imshow('shape_RoI_rectified', shape_plate)
                cv2.waitKey(0)
        
        if GlobalConfig.DEBUG:
            print("color rectification num:", len(shape_plates))
            
        return shape_plates

    ##########---------- Characters ----------##########

    def color_shape_characters_preprocess(self, blue_plates:list, green_plates:list) -> tuple:
        '''preprocess for characters detection to split each character's region from plates by color_shape_pipeline
        Params:
            blue_plates: list of blue license plates detected by color_shape_pipeline
            green_plates: list of green license plates detected by color_shape_pipeline
        ----------
        Return:
            blue_characters_list, green_characters_list: list of blue and green characters
                                                            first images, second center, third index
        '''
        blue_characters_list = []
        for blue_plate, blue_center, index in blue_plates:
            # binarization to show character shapes better, characters white
            blue_plate = cv2.cvtColor(blue_plate, cv2.COLOR_BGR2GRAY)
            _, blue_plate = cv2.threshold(blue_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # remove up and down borders according to wave peaks
            x_histogram = np.sum(blue_plate, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / DetectionConfig.WAVE_X_THRESHOLD_DIVIDER
            x_wave_peaks = wave_analysis(x_histogram, x_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_BORDER, DetectionConfig.WAVE_HIGH_GRANULARITY_BORDER)
            if len(x_wave_peaks) == 0:
                if GlobalConfig.DEBUG:
                    print(f"{index}th blue plate x wave analysis failed")
                continue
            characters_range = max(x_wave_peaks, key=lambda x: x[1] - x[0])     # max range of wave peak
            blue_plate = blue_plate[characters_range[0]:characters_range[1]]
            
            # split characters according to wave peaks
            y_histogram = np.sum(blue_plate, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / DetectionConfig.WAVE_Y_THRESHOLD_DIVIDER
            y_wave_peaks = wave_analysis(y_histogram, y_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_CHARACTER, DetectionConfig.WAVE_HIGH_GRANULARITY_CHARACTER)
            if len(y_wave_peaks) != DetectionConfig.BLUE_CHARACTERS_NUM:
                # not 7 wave peaks
                # TODO: cannot deal with `1` since it is also thin
                if GlobalConfig.DEBUG:
                    print(f"{index}th blue plate y wave analysis failed")
                continue
            blue_characters = []
            for y_wave_peak in y_wave_peaks:
                blue_character = blue_plate[:, y_wave_peak[0]:y_wave_peak[1]]
                blue_character = cv2.copyMakeBorder(blue_character, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                blue_characters.append(blue_character)
            blue_characters_list.append((blue_characters, blue_center, index))
            self.color_shape_blue_characters_images.append((blue_characters, index))
            if GlobalConfig.DEBUG:
                for i in range(len(blue_characters)):
                    cv2.imshow('blue_character_' + str(i), blue_characters[i])
                cv2.waitKey(0)
        
        green_characters_list = []
        for green_plate, green_center, index in green_plates:
            # binarization to show character shapes better, characters white
            green_plate = cv2.cvtColor(green_plate, cv2.COLOR_BGR2GRAY)
            green_plate = cv2.bitwise_not(green_plate)              # green plates with black characters
            _, green_plate = cv2.threshold(green_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # remove up and down borders according to wave peaks
            x_histogram = np.sum(green_plate, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / DetectionConfig.WAVE_X_THRESHOLD_DIVIDER
            x_wave_peaks = wave_analysis(x_histogram, x_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_BORDER, DetectionConfig.WAVE_HIGH_GRANULARITY_BORDER)
            if len(x_wave_peaks) == 0:
                if GlobalConfig.DEBUG:
                    print(f"{index}th green plate x wave analysis failed")
                continue
            characters_range = max(x_wave_peaks, key=lambda x: x[1] - x[0])     # max range of wave peak
            green_plate = green_plate[characters_range[0]:characters_range[1]]
            
            # split characters according to wave peaks
            y_histogram = np.sum(green_plate, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / DetectionConfig.WAVE_Y_THRESHOLD_DIVIDER
            y_wave_peaks = wave_analysis(y_histogram, y_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_CHARACTER, DetectionConfig.WAVE_HIGH_GRANULARITY_CHARACTER)
            if len(y_wave_peaks) != DetectionConfig.GREEN_CHARACTERS_NUM:
                # not 8 wave peaks
                # TODO: cannot deal with `1` since it is also thin
                if GlobalConfig.DEBUG:
                    print(f"{index}th green plate y wave analysis failed")
                continue
            green_characters = []
            for y_wave_peak in y_wave_peaks:
                green_character = green_plate[:, y_wave_peak[0]:y_wave_peak[1]]
                green_character = cv2.copyMakeBorder(green_character, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                green_characters.append(green_character)
            green_characters_list.append((green_characters, green_center, index))
            self.color_shape_green_characters_images.append((green_characters, index))
            if GlobalConfig.DEBUG:
                for i in range(len(green_characters)):
                    cv2.imshow('green_character_' + str(i), green_characters[i])
                cv2.waitKey(0)
        
        if GlobalConfig.DEBUG:
            print("color_shape characters split num:", len(blue_characters_list), len(green_characters_list))

        return (blue_characters_list, green_characters_list)
    
    def shape_color_characters_preprocess(self, shape_plates:list) -> tuple:
        '''preprocess for characters detection to split each character's region from plates by shape_color_pipeline
        Params:
            shape_plates: list of license plates detected by shape_color_pipeline
        ----------
        Return:
            blue_shape_characters_list, green_shape_characters_list: list of blue and green characters
                                                                        first images, second center, third index
        ----------
        Note:
            almost same with color_shape_characters_preprocess except first binarization
        '''
        blue_shape_plates = []
        green_shape_plates = []
        for shape_plate, shape_center, index in shape_plates:
            # binarization to show character shapes better, characters white
            shape_plate1 = cv2.cvtColor(shape_plate, cv2.COLOR_BGR2GRAY)
            shape_plate2 = cv2.bitwise_not(shape_plate1)
            _, shape_plate1 = cv2.threshold(shape_plate1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, shape_plate2 = cv2.threshold(shape_plate2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blue_shape_plates.append((shape_plate1, shape_center, index))
            green_shape_plates.append((shape_plate2, shape_center, index))

        blue_shape_characters_list = []
        for blue_shape_plate, blue_shape_center, index in blue_shape_plates:
            # remove up and down borders according to wave peaks
            x_histogram = np.sum(blue_shape_plate, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / DetectionConfig.WAVE_X_THRESHOLD_DIVIDER
            x_wave_peaks = wave_analysis(x_histogram, x_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_BORDER, DetectionConfig.WAVE_HIGH_GRANULARITY_BORDER)
            if len(x_wave_peaks) == 0:
                if GlobalConfig.DEBUG:
                    print(f"{index}th shape plate's blue branch x wave analysis failed")
                continue
            characters_range = max(x_wave_peaks, key=lambda x: x[1] - x[0])     # max range of wave peak
            blue_shape_plate = blue_shape_plate[characters_range[0]:characters_range[1]]
            
            # split characters according to wave peaks
            y_histogram = np.sum(blue_shape_plate, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / DetectionConfig.WAVE_Y_THRESHOLD_DIVIDER
            y_wave_peaks = wave_analysis(y_histogram, y_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_CHARACTER, DetectionConfig.WAVE_HIGH_GRANULARITY_CHARACTER)
            if len(y_wave_peaks) != DetectionConfig.BLUE_CHARACTERS_NUM:
                # not 7 wave peaks
                # TODO: cannot deal with `1` since it is also thin
                if GlobalConfig.DEBUG:
                    print(f"{index}th shape plate's blue branch y wave analysis failed")
                continue
            blue_shape_characters = []
            for y_wave_peak in y_wave_peaks:
                blue_shape_character = blue_shape_plate[:, y_wave_peak[0]:y_wave_peak[1]]
                blue_shape_character = cv2.copyMakeBorder(blue_shape_character, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                blue_shape_characters.append(blue_shape_character)
            blue_shape_characters_list.append((blue_shape_characters, blue_shape_center, index))
            self.shape_color_blue_characters_images.append((blue_shape_characters, index))
            if GlobalConfig.DEBUG:
                for i in range(len(blue_shape_characters)):
                    cv2.imshow('blue_shape_character_' + str(i), blue_shape_characters[i])
                cv2.waitKey(0)
        
        green_shape_characters_list = []
        for green_shape_plate, green_shape_center, index in green_shape_plates:
            # remove up and down borders according to wave peaks
            x_histogram = np.sum(green_shape_plate, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / DetectionConfig.WAVE_X_THRESHOLD_DIVIDER
            x_wave_peaks = wave_analysis(x_histogram, x_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_BORDER, DetectionConfig.WAVE_HIGH_GRANULARITY_BORDER)
            if len(x_wave_peaks) == 0:
                if GlobalConfig.DEBUG:
                    print(f"{index}th shape plate's green branch x wave analysis failed")
                continue
            characters_range = max(x_wave_peaks, key=lambda x: x[1] - x[0])     # max range of wave peak
            green_shape_plate = green_shape_plate[characters_range[0]:characters_range[1]]
            
            # split characters according to wave peaks
            y_histogram = np.sum(green_shape_plate, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / DetectionConfig.WAVE_Y_THRESHOLD_DIVIDER
            y_wave_peaks = wave_analysis(y_histogram, y_threshold, DetectionConfig.WAVE_LOW_GRANULARITY_CHARACTER, DetectionConfig.WAVE_HIGH_GRANULARITY_CHARACTER)
            if len(y_wave_peaks) != DetectionConfig.GREEN_CHARACTERS_NUM:
                # not 8 wave peaks
                # TODO: cannot deal with `1` since it is also thin
                if GlobalConfig.DEBUG:
                    print(f"{index}th shape plate's green branch y wave analysis failed")
                continue
            green_shape_characters = []
            for y_wave_peak in y_wave_peaks:
                green_shape_character = green_shape_plate[:, y_wave_peak[0]:y_wave_peak[1]]
                green_shape_character = cv2.copyMakeBorder(green_shape_character, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, DetectionConfig.CHARACTER_PADDING_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                green_shape_characters.append(green_shape_character)
            green_shape_characters_list.append((green_shape_characters, green_shape_center, index))
            self.shape_color_green_characters_images.append((green_shape_characters, index))
            if GlobalConfig.DEBUG:
                for i in range(len(green_shape_characters)):
                    cv2.imshow('green_shape_character_' + str(i), green_shape_characters[i])
                cv2.waitKey(0)
        
        if GlobalConfig.DEBUG:
            print("shape_color characters split num:", len(blue_shape_characters_list), len(green_shape_characters_list))
        
        return (blue_shape_characters_list, green_shape_characters_list)
    
    def characters_classify(self, blue_characters_list:list, green_characters_list:list, pipeline:str) -> tuple:
        '''classify characters according to SVM models
        Params:
            blue_characters_list: a list of lists of blue characters images
            green_characters_list: a list of lists of green characters images
            pipeline: 'color_shape' or 'shape_color'
        ----------
        Return:
            blue_characters_result, green_characters_result: a list of lists of blue and green characters results
                                                                first characters, second center, third index
        '''
        blue_characters_result = []
        for blue_characters, blue_center, index in blue_characters_list:
            result = []
            case = 0        # 0 for all digits, 1 for 1 en, 2 for 2 en
            for i, blue_character in enumerate(blue_characters):
                image = cv2.resize(blue_character, (ClassificationConfig.IMAGE_SIZE, ClassificationConfig.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                feature = classify_preprocess(image)
                feature = np.float32([feature])
                if i == 0:
                    # Chinese province
                    ret = self.model_zh.predict(feature)
                    charactor = ClassificationConfig.LABELS_ZH_INVERTED[ret[0]]
                elif i == 1:
                    # English code, must be English
                    ret = self.model_en.predict(feature)
                    charactor = ClassificationConfig.LABELS_EN_INVERTED[ret[0]]
                    charactor = digit2char(charactor)
                else:
                    # sequence numbers, conservative transformation
                    ret = self.model_en.predict(feature)
                    charactor = ClassificationConfig.LABELS_EN_INVERTED[ret[0]]
                    if charactor == 'O':
                        charactor = '0'
                    elif charactor == 'I':
                        charactor = '1'
                    else:
                        charactor = charactor
                    if case < 2:
                        if not charactor.isdigit():
                            case += 1
                        else:
                            case = case
                    else:
                        charactor = char2digit(charactor)
                result.append(charactor)
            blue_characters_result.append((result, blue_center, index))
            if pipeline == 'color_shape':
                self.color_shape_blue_characters_results.append((result, index))
            elif pipeline == 'shape_color':
                self.shape_color_blue_characters_results.append((result, index))
            else:
                raise ValueError(f"pipeline should be 'color_shape' or 'shape_color', but got {pipeline}")
        
        green_characters_result = []
        for green_characters, green_center, index in green_characters_list:
            result = []
            case = 0        # 0 for all digits, 1 for 1st en, 2 for 1st 2nd en, 3 for 6th en
            for i, green_character in enumerate(green_characters):
                image = cv2.resize(green_character, (ClassificationConfig.IMAGE_SIZE, ClassificationConfig.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                feature = classify_preprocess(image)
                feature = np.float32([feature])
                if i == 0:
                    # Chinese province
                    ret = self.model_zh.predict(feature)
                    charactor = ClassificationConfig.LABELS_ZH_INVERTED[ret[0]]
                elif i == 1:
                    # English code, must be English
                    ret = self.model_en.predict(feature)
                    charactor = ClassificationConfig.LABELS_EN_INVERTED[ret[0]]
                    charactor = digit2char(charactor)
                else:
                    # sequence numbers, conservative transformation
                    ret = self.model_en.predict(feature)
                    charactor = ClassificationConfig.LABELS_EN_INVERTED[ret[0]]
                    if i == 2:
                        if charactor.isdigit():
                            charactor = charactor
                            case = 0
                        else:
                            if charactor == 'O':
                                charactor = '0'
                                case = 0
                            elif charactor == 'I':
                                charactor = '1'
                                case = 0
                            else:
                                charactor = charactor
                                case = 1
                    elif i == 3:
                        if charactor.isdigit():
                            charactor = charactor
                            case = case
                        else:
                            if charactor == 'O':
                                charactor = '0'
                                case = case
                            elif charactor == 'I':
                                charactor = '1'
                                case = case
                            else:
                                if case == 0:
                                    charactor = char2digit(charactor)
                                    case = 0
                                else:
                                    charactor = charactor
                                    case = 2
                    elif i == 7:
                        if charactor.isdigit():
                            charactor = charactor
                            case = case
                        else:
                            if charactor == 'O':
                                charactor = '0'
                                case = case
                            elif charactor == 'I':
                                charactor = '1'
                                case = case
                            else:
                                if case == 0:
                                    charactor = charactor
                                    case = 3
                                else:
                                    charactor = char2digit(charactor)
                                    case = case
                    else:
                        charactor = char2digit(charactor)
                result.append(charactor)
            green_characters_result.append((result, green_center, index))
            if pipeline == 'color_shape':
                self.color_shape_green_characters_results.append((result, index))
            elif pipeline == 'shape_color':
                self.shape_color_green_characters_results.append((result, index))
            else:
                raise ValueError(f"pipeline should be 'color_shape' or 'shape_color', but got {pipeline}")

        return (blue_characters_result, green_characters_result)
    

    def color_shape_pipeline(self) -> tuple:
        '''pipeline for first color second shape detection
        Params:
            self.image: original BGR image
        ----------
        Return:
            blue_characters_result: a list of lists of blue characters results
            green_characters_result: a list of lists of green characters results
        ----------
        Note:
            step 1: RoI proposal based on color
            step 2: image rectification based on shape
            step 3: characters split based on wave analysis
            step 4: characters classification based on SVM
        '''
        if GlobalConfig.INFO:
            print('[Detector] Start color_shape pipeline...')
        start_time = time.time()

        image_resized, image_preprocessed_bgr, image_preprocessed_hsv = self.color_shape_preprocess()
        blue_RoIs, green_RoIs = self.color_RoI_proposal(image_resized, image_preprocessed_bgr, image_preprocessed_hsv)
        blue_plates, green_plates = self.shape_rectification(blue_RoIs, green_RoIs)
        blue_characters_list, green_characters_list = self.color_shape_characters_preprocess(blue_plates, green_plates)
        blue_characters_result, green_characters_result = self.characters_classify(blue_characters_list, green_characters_list, 'color_shape')
        
        end_time = time.time()
        if GlobalConfig.DEBUG:
            cv2.destroyAllWindows()
        if GlobalConfig.INFO:
            print(f'[Detector] Pipeline color_shape finished in {end_time - start_time} seconds.')
        if GlobalConfig.INFO:
            print('[Detector] Pipeline color_shape blue characters results: ', [blue_characters[0] for blue_characters in blue_characters_result])
            print('[Detector] Pipeline color_shape green characters results: ', [green_characters[0] for green_characters in green_characters_result])
        
        return (blue_characters_result, green_characters_result)

    def shape_color_pipeline(self) -> tuple:
        '''pipeline for first shape second color detection
        Params:
            self.image: original BGR image
        ----------
        Return:
            blue_characters_result: a list of lists of blue characters results
            green_characters_result: a list of lists of green characters results
        ----------
        Note:
            step 1: RoI proposal based on shape
            step 2: image rectification based on color
            step 3: characters split based on wave analysis
            step 4: characters classification based on SVM
        '''
        if GlobalConfig.INFO:
            print('[Detector] Start shape_color pipeline...')
        start_time = time.time()

        image_resized, image_preprocessed_gray = self.shape_color_preprocess()
        shape_RoIs = self.shape_RoI_proposal(image_resized, image_preprocessed_gray)
        shape_plates = self.color_rectification(shape_RoIs)
        blue_characters_list, green_characters_list = self.shape_color_characters_preprocess(shape_plates)
        blue_characters_result, green_characters_result = self.characters_classify(blue_characters_list, green_characters_list, 'shape_color')
        
        end_time = time.time()
        if GlobalConfig.DEBUG:
            cv2.destroyAllWindows()
        if GlobalConfig.INFO:
            print(f'[Detector] Pipeline shape_color finished in {end_time - start_time} seconds.')
        if GlobalConfig.INFO:
            print('[Detector] Pipeline shape_color blue characters results: ', [blue_characters[0] for blue_characters in blue_characters_result])
            print('[Detector] Pipeline shape_color green characters results: ', [green_characters[0] for green_characters in green_characters_result])
        
        return (blue_characters_result, green_characters_result)


    def detect(self) -> None:
        '''detection complete pipelines
        Params:
            self.image: original BGR image
        ----------
        Return:
            self.blue_characters_results: a list of (BGR image, list of characters)s
            self.green_characters_results: a list of (BGR image, list of characters)s
        ----------
        Note:
            TODO: possible improvement: merge results rather than just filter
        '''
        if not GlobalConfig.DEBUG:
            thread_color_shape = MyThread(self.color_shape_pipeline, ())
            thread_shape_color = MyThread(self.shape_color_pipeline, ())
            thread_color_shape.start()
            thread_shape_color.start()
            thread_color_shape.join()
            thread_shape_color.join()
            blue_characters_result_color_shape, green_characters_result_color_shape = thread_color_shape.get_result()
            blue_characters_result_shape_color, green_characters_result_shape_color = thread_shape_color.get_result()
        else:
            blue_characters_result_color_shape, green_characters_result_color_shape = self.color_shape_pipeline()
            blue_characters_result_shape_color, green_characters_result_shape_color = self.shape_color_pipeline()
        
        # filter results
        blue_hashset = []
        for blue_characters_result in blue_characters_result_color_shape:
            center = blue_characters_result[1]
            flag = False
            for blue_hash in blue_hashset:
                if abs(blue_hash[0] - center[0]) < DetectionConfig.FILTER_OFFSET and abs(blue_hash[1] - center[1]) < DetectionConfig.FILTER_OFFSET:
                    flag = True
                    break
            if not flag:
                blue_hashset.append(center)
                self.blue_characters_results.append((self.color_shape_blue_RoIs[blue_characters_result[2]], blue_characters_result[0]))
        for blue_characters_result in blue_characters_result_shape_color:
            center = blue_characters_result[1]
            flag = False
            for blue_hash in blue_hashset:
                if abs(blue_hash[0] - center[0]) < DetectionConfig.FILTER_OFFSET and abs(blue_hash[1] - center[1]) < DetectionConfig.FILTER_OFFSET:
                    flag = True
                    break
            if not flag:
                blue_hashset.append(center)
                self.blue_characters_results.append((self.shape_color_RoIs[blue_characters_result[2]], blue_characters_result[0]))
        
        green_hashset = []
        for green_characters_result in green_characters_result_color_shape:
            center = green_characters_result[1]
            flag = False
            for green_hash in green_hashset:
                if abs(green_hash[0] - center[0]) < DetectionConfig.FILTER_OFFSET and abs(green_hash[1] - center[1]) < DetectionConfig.FILTER_OFFSET:
                    flag = True
                    break
            if not flag:
                green_hashset.append(center)
                self.green_characters_results.append((self.color_shape_green_RoIs[green_characters_result[2]], green_characters_result[0]))
        for green_characters_result in green_characters_result_shape_color:
            center = green_characters_result[1]
            flag = False
            for green_hash in green_hashset:
                if abs(green_hash[0] - center[0]) < DetectionConfig.FILTER_OFFSET and abs(green_hash[1] - center[1]) < DetectionConfig.FILTER_OFFSET:
                    flag = True
                    break
            if not flag:
                green_hashset.append(center)
                self.green_characters_results.append((self.shape_color_RoIs[green_characters_result[2]], green_characters_result[0]))

        if GlobalConfig.INFO:
            if not GlobalConfig.HIDDEN:
                for i, blue_characters_result in enumerate(self.blue_characters_results):
                    cv2.imshow(f'ultimate_blue_RoI_{i}', blue_characters_result[0])
            print('[Detector] Ultimate blue characters result: ', [blue_characters_result[1] for blue_characters_result in self.blue_characters_results])
            if not GlobalConfig.HIDDEN:
                for i, green_characters_result in enumerate(self.green_characters_results):
                    cv2.imshow(f'ultimate_green_RoI_{i}', green_characters_result[0])
            print('[Detector] Ultimate green characters result: ', [green_characters_result[1] for green_characters_result in self.green_characters_results])
            if not GlobalConfig.HIDDEN:
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    my_detector = ChineseLicensePlateDetector()
    image_path = "images/easy/1-1.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/easy/1-2.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/easy/1-3.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/medium/2-1.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/medium/2-2.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/medium/2-3.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/difficult/3-1.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/difficult/3-2.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
    image_path = "images/difficult/3-3.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    my_detector.set_image(image)
    my_detector.detect()
    print()
