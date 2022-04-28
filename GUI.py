import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import ttk

from detector import ChineseLicensePlateDetector
from config import GUIConfig, GlobalConfig

class CLPDetectorApp(ttk.Frame):
    def __init__(self, master:tk.Tk) -> None:
        super(CLPDetectorApp, self).__init__(master)
        # initialize members
        self.master = master
        self.original_image = None
        self.original_frame = None
        self.result_frame = None
        self.button_frame = None
        self.original_show = None
        self.original_imgtk = None
        self.results_show = []
        self.results_imgtk = []
        self.color_shape_feature_imgtk = None
        self.color_shape_blue_mask_imgtk = None
        self.color_shape_green_mask_imgtk = None
        self.color_shape_blue_RoIs_imgtk = []
        self.color_shape_green_RoIs_imgtk = []
        self.color_shape_RoI_contours_imgtk = None
        self.color_shape_blue_RoIs_features_imgtk = []
        self.color_shape_green_RoIs_features_imgtk = []
        self.color_shape_blue_RoIs_rectified_imgtk = []
        self.color_shape_green_RoIs_rectified_imgtk = []
        self.color_shape_blue_characters_images_imgtk = []
        self.color_shape_green_characters_images_imgtk = []
        self.shape_color_feature_imgtk = None
        self.shape_color_mask_imgtk = None
        self.shape_color_RoIs_imgtk = []
        self.shape_color_RoI_contours_imgtk = None
        self.shape_color_RoIs_features_imgtk = []
        self.shape_color_RoIs_rectified_imgtk = []
        self.shape_color_blue_charactors_images_imgtk = []
        self.shape_color_green_charactors_images_imgtk = []
        self.detected = False
        self.detector = ChineseLicensePlateDetector()

        # set up the GUI
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="10", pady="10")
        self.original_frame = ttk.Frame(self)
        self.result_frame = ttk.Frame(self)
        self.button_frame = ttk.Frame(self)
        self.original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.result_frame.pack(side=tk.TOP, expand=True, fill=tk.Y)
        self.button_frame.pack(side=tk.RIGHT, expand=False)
        ttk.Label(self.original_frame, text="original image: ").pack(anchor="nw")
        self.original_show = ttk.Label(self.original_frame)
        self.original_show.pack(anchor="nw")
        ttk.Label(self.result_frame, text="detection results: ").grid(column=0, row=0, sticky=tk.W)
        ttk.Button(self.button_frame, text="Start", command=self.start).pack(anchor="s", pady="5")
        ttk.Button(self.button_frame, text="Visualization", command=self.visualization).pack(anchor="s", pady="5")
        ttk.Button(self.button_frame, text="Reset", command=self.reset).pack(anchor="s", pady="5")
        ttk.Button(self.button_frame, text="Exit", command=self.exit).pack(anchor="s", pady="5")
    
    def _upload_image(self) -> int:
        '''upload image
        Return:
            code: 0 for success, 1 for cancel, -1 for error
        ----------
        Note:
            influence self.original_image
        '''
        image_path = fd.askopenfilename(title="select", filetypes=[("jpg image", "*.jpg"), ("png image", "*.png")])
        if GlobalConfig.DEBUG:
            print("select image path:", image_path)
        if image_path:
            try:
                self.original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                return 0
            except Exception:
                self.original_image = None
                mb.showerror("error", "read in image failed")
                return -1
        else:
            self.original_image = None
            return 1
    
    def _get_imgtk(self, image:np.ndarray, max_width:int, max_height:int, gray:bool=False) -> ImageTk.PhotoImage:
        '''convert normal image to imgtk
        Parameters:
            image: the image to be converted
            max_width: the max width of the image
            max_height: the max height of the image
            gray: whether the image is gray
        Return:
            imgtk: the converted imgtk
        '''
        if not gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        wide = imgtk.width()
        height = imgtk.height()
        if wide > max_width or height > max_height:
            factor = min(max_width / wide, max_height / height)
            wide = int(wide * factor)
            if wide <= 0:
                wide = 1
            height = int(height * factor)
            if height <= 0:
                height = 1
            image = image.resize((wide, height), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=image)
        return imgtk
    
    def _set_original(self) -> None:
        '''set original image to show'''
        self.original_imgtk = self._get_imgtk(self.original_image, GUIConfig.ORIGINAL_SHOW_MAX_WIDTH, GUIConfig.ORIGINAL_SHOW_MAX_HEIGHT)
        self.original_show.configure(image=self.original_imgtk)
    
    def _set_result(self) -> None:
        '''set result to show'''
        r = 1

        for i, blue_result in enumerate(self.detector.blue_characters_results):
            image, characters = blue_result

            blue_result_text = ttk.Label(self.result_frame, text=f"{i}th blue license plate: ")
            self.results_show.append(blue_result_text)
            blue_result_text.grid(column=0, row=r, sticky=tk.W)

            imgtk = self._get_imgtk(image, GUIConfig.RESULT_SHOW_MAX_WIDTH, GUIConfig.RESULT_SHOW_MAX_HEIGHT)
            blue_result_image = ttk.Label(self.result_frame, image=imgtk)
            self.results_imgtk.append(imgtk)
            self.results_show.append(blue_result_image)
            blue_result_image.grid(column=1, row=r, sticky=tk.W)

            blue_result_characters = ttk.Label(self.result_frame, text=str(characters), font=('Times',20))
            self.results_show.append(blue_result_characters)
            blue_result_characters.grid(column=2, row=r, sticky=tk.W)

            r += 1
        
        for i, green_result in enumerate(self.detector.green_characters_results):
            image, characters = green_result

            green_result_text = ttk.Label(self.result_frame, text=f"{i}th green license plate: ")
            self.results_show.append(green_result_text)
            green_result_text.grid(column=0, row=r, sticky=tk.W)

            imgtk = self._get_imgtk(image, GUIConfig.RESULT_SHOW_MAX_WIDTH, GUIConfig.RESULT_SHOW_MAX_HEIGHT)
            green_result_image = ttk.Label(self.result_frame, image=imgtk)
            self.results_imgtk.append(imgtk)
            self.results_show.append(green_result_image)
            green_result_image.grid(column=1, row=r, sticky=tk.W)

            green_result_characters = ttk.Label(self.result_frame, text=str(characters), font=('Times',20))
            self.results_show.append(green_result_characters)
            green_result_characters.grid(column=2, row=r, sticky=tk.W)

            r += 1
        
        if r == 1:
            result_text = ttk.Label(self.result_frame, text="no detected license plates", foreground="red", font=('Arial',15))
            self.results_show.append(result_text)
            result_text.grid(column=0, row=r, sticky=tk.W)
    
    def start(self) -> None:
        # reset the variables
        self.original_image = None
        self.original_show.destroy()
        self.original_show = ttk.Label(self.original_frame)
        self.original_show.pack(anchor="nw")
        self.original_imgtk = None
        for item in self.results_show:
            item.destroy()
        self.results_imgtk = []

        # upload image
        ret = self._upload_image()
        if ret == -1:
            return
        elif ret == 1:
            return
        
        # set original image
        self._set_original()
        
        # detect
        self.detector.set_image(self.original_image)
        self.detector.detect()

        # set result
        self._set_result()

        self.detected = True
    
    def visualization(self) -> None:
        if not self.detected:
            mb.showwarning("warning", "please first perform detection then visualize by clicking the `Start` button first")
            return
        
        # reset the variables
        self.color_shape_feature_imgtk = None
        self.color_shape_blue_mask_imgtk = None
        self.color_shape_green_mask_imgtk = None
        self.color_shape_blue_RoIs_imgtk = []
        self.color_shape_green_RoIs_imgtk = []
        self.color_shape_RoI_contours_imgtk = None
        self.color_shape_blue_RoIs_features_imgtk = []
        self.color_shape_green_RoIs_features_imgtk = []
        self.color_shape_blue_RoIs_rectified_imgtk = []
        self.color_shape_green_RoIs_rectified_imgtk = []
        self.color_shape_blue_characters_images_imgtk = []
        self.color_shape_green_characters_images_imgtk = []
        self.shape_color_feature_imgtk = None
        self.shape_color_mask_imgtk = None
        self.shape_color_RoIs_imgtk = []
        self.shape_color_RoI_contours_imgtk = None
        self.shape_color_RoIs_features_imgtk = []
        self.shape_color_RoIs_rectified_imgtk = []
        self.shape_color_blue_charactors_images_imgtk = []
        self.shape_color_green_charactors_images_imgtk = []

        visualization_window = tk.Toplevel(self)
        visualization_window.title("Visualization")
        color_shape_frame = ttk.Frame(visualization_window)
        shape_color_frame = ttk.Frame(visualization_window)
        color_shape_frame.grid(column=0, row=0, sticky=tk.W)
        shape_color_frame.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(color_shape_frame, text="color_shape_pipeline: ", font=('Arial',15)).grid(column=0, row=0, sticky=tk.W)
        ttk.Label(shape_color_frame, text="shape_color_pipeline: ", font=('Arial',15)).grid(column=0, row=0, sticky=tk.W)
        color_shape_content_frame = ttk.Frame(color_shape_frame)
        color_shape_content_frame.grid(column=0, row=1, sticky=tk.W)
        shape_color_content_frame = ttk.Frame(shape_color_frame)
        shape_color_content_frame.grid(column=0, row=1, sticky=tk.W)

        #----------
        color_shape_feature_frame = ttk.Frame(color_shape_content_frame)
        color_shape_feature_frame.grid(column=0, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(color_shape_feature_frame, text="color_shape_feature").grid(column=0, row=0)
        self.color_shape_feature_imgtk = self._get_imgtk(self.detector.color_shape_feature, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
        ttk.Label(color_shape_feature_frame, image=self.color_shape_feature_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        color_shape_blue_branch_frame = ttk.Frame(color_shape_content_frame)
        color_shape_blue_branch_frame.grid(column=1, row=0, sticky=tk.W)
        color_shape_green_branch_frame = ttk.Frame(color_shape_content_frame)
        color_shape_green_branch_frame.grid(column=1, row=1, sticky=tk.W)
        #----------
        color_shape_blue_branch_mask_frame = ttk.Frame(color_shape_blue_branch_frame)
        color_shape_blue_branch_mask_frame.grid(column=0, row=0, sticky=tk.W)
        color_shape_green_branch_mask_frame = ttk.Frame(color_shape_green_branch_frame)
        color_shape_green_branch_mask_frame.grid(column=0, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_mask_frame, text="blue_mask").grid(column=0, row=0)
        self.color_shape_blue_mask_imgtk = self._get_imgtk(self.detector.color_shape_blue_mask, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT, gray=True)
        ttk.Label(color_shape_blue_branch_mask_frame, image=self.color_shape_blue_mask_imgtk).grid(column=0, row=1, sticky=tk.W)
        ttk.Label(color_shape_green_branch_mask_frame, text="green_mask").grid(column=0, row=0)
        self.color_shape_green_mask_imgtk = self._get_imgtk(self.detector.color_shape_green_mask, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT, gray=True)
        ttk.Label(color_shape_green_branch_mask_frame, image=self.color_shape_green_mask_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        color_shape_RoI_contours_frame = ttk.Frame(color_shape_content_frame)
        color_shape_RoI_contours_frame.grid(column=2, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(color_shape_RoI_contours_frame, text="color_shape_RoI_contours").grid(column=0, row=0)
        self.color_shape_RoI_contours_imgtk = self._get_imgtk(self.detector.color_shape_RoI_contours_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
        ttk.Label(color_shape_RoI_contours_frame, image=self.color_shape_RoI_contours_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        color_shape_blue_branch_new_frame = ttk.Frame(color_shape_content_frame)
        color_shape_blue_branch_new_frame.grid(column=3, row=0, sticky=tk.W)
        color_shape_green_branch_new_frame = ttk.Frame(color_shape_content_frame)
        color_shape_green_branch_new_frame.grid(column=3, row=1, sticky=tk.W)
        #----------
        color_shape_blue_branch_RoIs_frame = ttk.Frame(color_shape_blue_branch_new_frame)
        color_shape_blue_branch_RoIs_frame.grid(column=0, row=0, sticky=tk.W)
        color_shape_green_branch_RoIs_frame = ttk.Frame(color_shape_green_branch_new_frame)
        color_shape_green_branch_RoIs_frame.grid(column=0, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_RoIs_frame, text="blue_RoIs").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_blue_RoIs)):
            color_shape_blue_RoI_imgtk = self._get_imgtk(self.detector.color_shape_blue_RoIs[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
            self.color_shape_blue_RoIs_imgtk.append(color_shape_blue_RoI_imgtk)
            ttk.Label(color_shape_blue_branch_RoIs_frame, image=color_shape_blue_RoI_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_blue_RoIs) == 0:
            ttk.Label(color_shape_blue_branch_RoIs_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        ttk.Label(color_shape_green_branch_RoIs_frame, text="green_RoIs").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_green_RoIs)):
            color_shape_green_RoI_imgtk = self._get_imgtk(self.detector.color_shape_green_RoIs[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
            self.color_shape_green_RoIs_imgtk.append(color_shape_green_RoI_imgtk)
            ttk.Label(color_shape_green_branch_RoIs_frame, image=color_shape_green_RoI_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_green_RoIs) == 0:
            ttk.Label(color_shape_green_branch_RoIs_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        color_shape_blue_branch_RoIs_features_frame = ttk.Frame(color_shape_blue_branch_new_frame)
        color_shape_blue_branch_RoIs_features_frame.grid(column=1, row=0, sticky=tk.W)
        color_shape_green_branch_RoIs_features_frame = ttk.Frame(color_shape_green_branch_new_frame)
        color_shape_green_branch_RoIs_features_frame.grid(column=1, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_RoIs_features_frame, text="blue_RoIs_features").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_blue_RoIs_features)):
            color_shape_blue_RoI_feature_imgtk = self._get_imgtk(self.detector.color_shape_blue_RoIs_features[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
            self.color_shape_blue_RoIs_features_imgtk.append(color_shape_blue_RoI_feature_imgtk)
            ttk.Label(color_shape_blue_branch_RoIs_features_frame, image=color_shape_blue_RoI_feature_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_blue_RoIs) == 0:
            ttk.Label(color_shape_blue_branch_RoIs_features_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        ttk.Label(color_shape_green_branch_RoIs_features_frame, text="green_RoIs_features").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_green_RoIs_features)):
            color_shape_green_RoI_feature_imgtk = self._get_imgtk(self.detector.color_shape_green_RoIs_features[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
            self.color_shape_green_RoIs_features_imgtk.append(color_shape_green_RoI_feature_imgtk)
            ttk.Label(color_shape_green_branch_RoIs_features_frame, image=color_shape_green_RoI_feature_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_green_RoIs) == 0:
            ttk.Label(color_shape_green_branch_RoIs_features_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        color_shape_blue_branch_RoIs_rectified_frame = ttk.Frame(color_shape_blue_branch_new_frame)
        color_shape_blue_branch_RoIs_rectified_frame.grid(column=2, row=0, sticky=tk.W)
        color_shape_green_branch_RoIs_rectified_frame = ttk.Frame(color_shape_green_branch_new_frame)
        color_shape_green_branch_RoIs_rectified_frame.grid(column=2, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_RoIs_rectified_frame, text="blue_RoIs_rectified").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_blue_RoIs_rectified)):
            if self.detector.color_shape_blue_RoIs_rectified[i] is None:
                self.color_shape_blue_RoIs_rectified_imgtk.append(None)
                ttk.Label(color_shape_blue_branch_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
            else:
                color_shape_blue_RoI_rectified_imgtk = self._get_imgtk(self.detector.color_shape_blue_RoIs_rectified[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
                self.color_shape_blue_RoIs_rectified_imgtk.append(color_shape_blue_RoI_rectified_imgtk)
                ttk.Label(color_shape_blue_branch_RoIs_rectified_frame, image=color_shape_blue_RoI_rectified_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_blue_RoIs) == 0:
            ttk.Label(color_shape_blue_branch_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        ttk.Label(color_shape_green_branch_RoIs_rectified_frame, text="green_RoIs_rectified").grid(column=0, row=0)
        for i in range(len(self.detector.color_shape_green_RoIs_rectified)):
            if self.detector.color_shape_green_RoIs_rectified[i] is None:
                self.color_shape_green_RoIs_rectified_imgtk.append(None)
                ttk.Label(color_shape_green_branch_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
            else:
                color_shape_green_RoI_rectified_imgtk = self._get_imgtk(self.detector.color_shape_green_RoIs_rectified[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
                self.color_shape_green_RoIs_rectified_imgtk.append(color_shape_green_RoI_rectified_imgtk)
                ttk.Label(color_shape_green_branch_RoIs_rectified_frame, image=color_shape_green_RoI_rectified_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.color_shape_green_RoIs) == 0:
            ttk.Label(color_shape_green_branch_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        color_shape_blue_branch_characters_images_frame = ttk.Frame(color_shape_blue_branch_new_frame)
        color_shape_blue_branch_characters_images_frame.grid(column=3, row=0, sticky=tk.W)
        color_shape_green_branch_characters_images_frame = ttk.Frame(color_shape_green_branch_new_frame)
        color_shape_green_branch_characters_images_frame.grid(column=3, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_characters_images_frame, text="blue_characters_images_split").grid(column=0, row=0, columnspan=7)                                     # use blue license plate feature
        _blue_filled = []
        for i in range(len(self.detector.color_shape_blue_characters_images)):
            color_shape_blue_characters_images_imgtk = [self._get_imgtk(color_shape_blue_characters_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH * 0.5, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT * 0.5, gray=True) for color_shape_blue_characters_image in self.detector.color_shape_blue_characters_images[i][0]]
            self.color_shape_blue_characters_images_imgtk.append(color_shape_blue_characters_images_imgtk)
            _blue_filled.append(self.detector.color_shape_blue_characters_images[i][1])
            for j in range(len(color_shape_blue_characters_images_imgtk)):
                ttk.Label(color_shape_blue_branch_characters_images_frame, image=color_shape_blue_characters_images_imgtk[j]).grid(column=j, row=self.detector.color_shape_blue_characters_images[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.color_shape_blue_RoIs)):
            if i not in _blue_filled:
                ttk.Label(color_shape_blue_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1, columnspan=7)       # use blue license plate feature
        if len(self.detector.color_shape_blue_RoIs) == 0:
            ttk.Label(color_shape_blue_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1, columnspan=7)             # use blue license plate feature
        ttk.Label(color_shape_green_branch_characters_images_frame, text="green_characters_images_split").grid(column=0, row=0, columnspan=8)                                   # use green license plate feature
        _green_filled = []
        for i in range(len(self.detector.color_shape_green_characters_images)):
            color_shape_green_characters_images_imgtk = [self._get_imgtk(color_shape_green_characters_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH * 0.5, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT * 0.5, gray=True) for color_shape_green_characters_image in self.detector.color_shape_green_characters_images[i][0]]
            self.color_shape_green_characters_images_imgtk.append(color_shape_green_characters_images_imgtk)
            _green_filled.append(self.detector.color_shape_green_characters_images[i][1])
            for j in range(len(color_shape_green_characters_images_imgtk)):
                ttk.Label(color_shape_green_branch_characters_images_frame, image=color_shape_green_characters_images_imgtk[j]).grid(column=j, row=self.detector.color_shape_green_characters_images[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.color_shape_green_RoIs)):
            if i not in _green_filled:
                ttk.Label(color_shape_green_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1, columnspan=8)      # use green license plate feature
        if len(self.detector.color_shape_green_RoIs) == 0:
            ttk.Label(color_shape_green_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1, columnspan=8)            # use green license plate feature
        #----------
        color_shape_blue_branch_characters_text_frame = ttk.Frame(color_shape_blue_branch_new_frame)
        color_shape_blue_branch_characters_text_frame.grid(column=4, row=0, sticky=tk.W)
        color_shape_green_branch_characters_text_frame = ttk.Frame(color_shape_green_branch_new_frame)
        color_shape_green_branch_characters_text_frame.grid(column=4, row=0, sticky=tk.W)
        ttk.Label(color_shape_blue_branch_characters_text_frame, text="blue_characters_text_split").grid(column=0, row=0)
        _blue_filled = []
        for i in range(len(self.detector.color_shape_blue_characters_results)):
            _blue_filled.append(self.detector.color_shape_blue_characters_results[i][1])
            ttk.Label(color_shape_blue_branch_characters_text_frame, text=str(self.detector.color_shape_blue_characters_results[i][0]), font=('Times',20)).grid(column=0, row=self.detector.color_shape_blue_characters_results[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.color_shape_blue_RoIs)):
            if i not in _blue_filled:
                ttk.Label(color_shape_blue_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
        if len(self.detector.color_shape_blue_RoIs) == 0:
            ttk.Label(color_shape_blue_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        ttk.Label(color_shape_green_branch_characters_text_frame, text="green_characters_text_split").grid(column=0, row=0)
        _green_filled = []
        for i in range(len(self.detector.color_shape_green_characters_results)):
            _green_filled.append(self.detector.color_shape_green_characters_results[i][1])
            ttk.Label(color_shape_green_branch_characters_text_frame, text=str(self.detector.color_shape_green_characters_results[i][0]), font=('Times',20)).grid(column=0, row=self.detector.color_shape_green_characters_results[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.color_shape_green_RoIs)):
            if i not in _green_filled:
                ttk.Label(color_shape_green_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
        if len(self.detector.color_shape_green_RoIs) == 0:
            ttk.Label(color_shape_green_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)

        #----------
        shape_color_feature_frame = ttk.Frame(shape_color_content_frame)
        shape_color_feature_frame.grid(column=0, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_feature_frame, text="shape_color_feature").grid(column=0, row=0)
        self.shape_color_feature_imgtk = self._get_imgtk(self.detector.shape_color_feature, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
        ttk.Label(shape_color_feature_frame, image=self.shape_color_feature_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        shape_color_mask_frame = ttk.Frame(shape_color_content_frame)
        shape_color_mask_frame.grid(column=1, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_mask_frame, text="shape_color_mask").grid(column=0, row=0)
        self.shape_color_mask_imgtk = self._get_imgtk(self.detector.shape_color_mask, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT, gray=True)
        ttk.Label(shape_color_mask_frame, image=self.shape_color_mask_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        shape_color_RoIs_frame = ttk.Frame(shape_color_content_frame)
        shape_color_RoIs_frame.grid(column=2, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_RoIs_frame, text="shape_color_RoIs").grid(column=0, row=0)
        self.shape_color_RoIs_imgtk = [self._get_imgtk(shape_color_RoI, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT) for shape_color_RoI in self.detector.shape_color_RoIs]
        for i in range(len(self.shape_color_RoIs_imgtk)):
            ttk.Label(shape_color_RoIs_frame, image=self.shape_color_RoIs_imgtk[i]).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.shape_color_RoIs) == 0:
            ttk.Label(shape_color_RoIs_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        shape_color_RoI_contours_frame = ttk.Frame(shape_color_content_frame)
        shape_color_RoI_contours_frame.grid(column=3, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_RoI_contours_frame, text="shape_color_RoI_contours").grid(column=0, row=0)
        self.shape_color_RoI_contours_imgtk = self._get_imgtk(self.detector.shape_color_RoI_contours_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
        ttk.Label(shape_color_RoI_contours_frame, image=self.shape_color_RoI_contours_imgtk).grid(column=0, row=1, sticky=tk.W)
        #----------
        shape_color_RoIs_features_frame = ttk.Frame(shape_color_content_frame)
        shape_color_RoIs_features_frame.grid(column=4, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_RoIs_features_frame, text="shape_color_RoIs_features").grid(column=0, row=0)
        self.shape_color_RoIs_features_imgtk = [self._get_imgtk(shape_color_RoI_feature, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT, gray=True) for shape_color_RoI_feature in self.detector.shape_color_RoIs_features]
        for i in range(len(self.shape_color_RoIs_features_imgtk)):
            ttk.Label(shape_color_RoIs_features_frame, image=self.shape_color_RoIs_features_imgtk[i]).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.shape_color_RoIs) == 0:
            ttk.Label(shape_color_RoIs_features_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        shape_color_RoIs_rectified_frame = ttk.Frame(shape_color_content_frame)
        shape_color_RoIs_rectified_frame.grid(column=5, row=0, rowspan=2, sticky=tk.W)
        ttk.Label(shape_color_RoIs_rectified_frame, text="shape_color_RoIs_rectified").grid(column=0, row=0)
        for i in range(len(self.detector.shape_color_RoIs_rectified)):
            if self.detector.shape_color_RoIs_rectified[i] is None:
                self.shape_color_RoIs_rectified_imgtk.append(None)
                ttk.Label(shape_color_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
            else:
                shape_color_RoI_rectified_imgtk = self._get_imgtk(self.detector.shape_color_RoIs_rectified[i], GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT)
                self.shape_color_RoIs_rectified_imgtk.append(shape_color_RoI_rectified_imgtk)
                ttk.Label(shape_color_RoIs_rectified_frame, image=shape_color_RoI_rectified_imgtk).grid(column=0, row=i+1, sticky=tk.W)
        if len(self.detector.shape_color_RoIs) == 0:
            ttk.Label(shape_color_RoIs_rectified_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        #----------
        shape_color_blue_branch_frame = ttk.Frame(shape_color_content_frame)
        shape_color_blue_branch_frame.grid(column=6, row=0, sticky=tk.W)
        shape_color_green_branch_frame = ttk.Frame(shape_color_content_frame)
        shape_color_green_branch_frame.grid(column=6, row=1, sticky=tk.W)
        #----------
        shape_color_blue_branch_characters_images_frame = ttk.Frame(shape_color_blue_branch_frame)
        shape_color_blue_branch_characters_images_frame.grid(column=0, row=0, sticky=tk.W)
        shape_color_green_branch_characters_images_frame = ttk.Frame(shape_color_green_branch_frame)
        shape_color_green_branch_characters_images_frame.grid(column=0, row=0, sticky=tk.W)
        ttk.Label(shape_color_blue_branch_characters_images_frame, text="blue_characters_images_split").grid(column=0, row=0, columnspan=7)                                     # use blue license plate feature
        _blue_filled = []
        for i in range(len(self.detector.shape_color_blue_characters_images)):
            shape_color_blue_charactors_images_imgtk = [self._get_imgtk(shape_color_blue_characters_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH * 0.5, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT * 0.5, gray=True) for shape_color_blue_characters_image in self.detector.shape_color_blue_characters_images[i][0]]
            self.shape_color_blue_charactors_images_imgtk.append(shape_color_blue_charactors_images_imgtk)
            _blue_filled.append(self.detector.shape_color_blue_characters_images[i][1])
            for j in range(len(shape_color_blue_charactors_images_imgtk)):
                ttk.Label(shape_color_blue_branch_characters_images_frame, image=shape_color_blue_charactors_images_imgtk[j]).grid(column=j, row=self.detector.shape_color_blue_characters_images[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.shape_color_RoIs)):
            if i not in _blue_filled:
                ttk.Label(shape_color_blue_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1, columnspan=7)       # use blue license plate feature
        if len(self.detector.shape_color_blue_characters_images) == 0:
            ttk.Label(shape_color_blue_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1, columnspan=7)             # use blue license plate feature
        ttk.Label(shape_color_green_branch_characters_images_frame, text="green_characters_images_split").grid(column=0, row=0, columnspan=8)                                   # use green license plate feature
        _green_filled = []
        for i in range(len(self.detector.shape_color_green_characters_images)):
            shape_color_green_charactors_images_imgtk = [self._get_imgtk(shape_color_green_characters_image, GUIConfig.VISUALIZATION_SHOW_MAX_WIDTH * 0.5, GUIConfig.VISUALIZATION_SHOW_MAX_HEIGHT * 0.5, gray=True) for shape_color_green_characters_image in self.detector.shape_color_green_characters_images[i][0]]
            self.shape_color_green_charactors_images_imgtk.append(shape_color_green_charactors_images_imgtk)
            _green_filled.append(self.detector.shape_color_green_characters_images[i][1])
            for j in range(len(shape_color_green_charactors_images_imgtk)):
                ttk.Label(shape_color_green_branch_characters_images_frame, image=shape_color_green_charactors_images_imgtk[j]).grid(column=j, row=self.detector.shape_color_green_characters_images[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.shape_color_RoIs)):
            if i not in _green_filled:
                ttk.Label(shape_color_green_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1, columnspan=8)      # use green license plate feature
        if len(self.detector.shape_color_green_characters_images) == 0:
            ttk.Label(shape_color_green_branch_characters_images_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1, columnspan=8)            # use green license plate feature
        #----------
        shape_color_blue_branch_characters_text_frame = ttk.Frame(shape_color_blue_branch_frame)
        shape_color_blue_branch_characters_text_frame.grid(column=1, row=0, sticky=tk.W)
        shape_color_green_branch_characters_text_frame = ttk.Frame(shape_color_green_branch_frame)
        shape_color_green_branch_characters_text_frame.grid(column=1, row=0, sticky=tk.W)
        ttk.Label(shape_color_blue_branch_characters_text_frame, text="blue_characters_text_split").grid(column=0, row=0)
        _blue_filled = []
        for i in range(len(self.detector.shape_color_blue_characters_results)):
            _blue_filled.append(self.detector.shape_color_blue_characters_results[i][1])
            ttk.Label(shape_color_blue_branch_characters_text_frame, text=str(self.detector.shape_color_blue_characters_results[i][0]), font=('Times',20)).grid(column=0, row=self.detector.shape_color_blue_characters_results[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.shape_color_RoIs)):
            if i not in _blue_filled:
                ttk.Label(shape_color_blue_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
        if len(self.detector.shape_color_blue_characters_results) == 0:
            ttk.Label(shape_color_blue_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)
        ttk.Label(shape_color_green_branch_characters_text_frame, text="green_characters_text_split").grid(column=0, row=0)
        _green_filled = []
        for i in range(len(self.detector.shape_color_green_characters_results)):
            _green_filled.append(self.detector.shape_color_green_characters_results[i][1])
            ttk.Label(shape_color_green_branch_characters_text_frame, text=str(self.detector.shape_color_green_characters_results[i][0]), font=('Times',20)).grid(column=0, row=self.detector.shape_color_green_characters_results[i][1]+1, sticky=tk.W)
        for i in range(len(self.detector.shape_color_RoIs)):
            if i not in _green_filled:
                ttk.Label(shape_color_green_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=i+1)
        if len(self.detector.shape_color_green_characters_results) == 0:
            ttk.Label(shape_color_green_branch_characters_text_frame, text="no detected", foreground="red", font=('Arial',15)).grid(column=0, row=1)

    def reset(self) -> None:
        self.original_image = None
        self.original_show.destroy()
        self.original_show = ttk.Label(self.original_frame)
        self.original_show.pack(anchor="nw")
        self.original_imgtk = None
        for item in self.results_show:
            item.destroy()
        self.results_imgtk = []
        self.detected = False

    def exit(self) -> None:
        self.destroy()
        self.master.destroy()
