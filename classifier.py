import os
import time
import numpy as np
import cv2

from utils import MyThread
from config import GlobalConfig, ClassificationConfig

class ModelWrapper:
    def load(self, file_path:str) -> None:
        self.model = self.model.load(file_path)

    def save(self, file_path:str) -> None:
        self.model.save(file_path)

class SVMWrapper(ModelWrapper):
    def __init__(self, C:float, gamma:float) -> None:
        self.model = cv2.ml.SVM_create()
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, features:np.ndarray, labels:np.ndarray) -> None:
        self.model.train(features, cv2.ml.ROW_SAMPLE, labels)

    def predict(self, features:np.ndarray) -> np.ndarray:
        r = self.model.predict(features)
        return r[1].ravel()


def classify_preprocess(image:np.ndarray) -> np.ndarray:
    '''preprocess image for character preprocess
    Params:
        image: input 20x20 gray image
    ----------
    Return:
        feature: preprocessed 128d image feature
    '''
    # deskew
    m = cv2.moments(image)
    if abs(m['mu02']) < ClassificationConfig.DESKEW_EPSILON:
        image_deskew = image.copy()
    else:
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * ClassificationConfig.IMAGE_SIZE * skew], [0, 1, 0]])
        image_deskew = cv2.warpAffine(image, M, (ClassificationConfig.IMAGE_SIZE, ClassificationConfig.IMAGE_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    
    # HOG
    gx = cv2.Sobel(image_deskew, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image_deskew, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy)
    angle_bin = np.int32(ClassificationConfig.BIN_NUM * angle / (2 * np.pi))
    angle_cells = angle_bin[:ClassificationConfig.IMAGE_SIZE//2, :ClassificationConfig.IMAGE_SIZE//2], angle_bin[ClassificationConfig.IMAGE_SIZE//2:, :ClassificationConfig.IMAGE_SIZE//2], angle_bin[:ClassificationConfig.IMAGE_SIZE//2, ClassificationConfig.IMAGE_SIZE//2:], angle_bin[ClassificationConfig.IMAGE_SIZE//2:, ClassificationConfig.IMAGE_SIZE//2:]
    magnitude_cells = magnitude[:ClassificationConfig.IMAGE_SIZE//2, :ClassificationConfig.IMAGE_SIZE//2], magnitude[ClassificationConfig.IMAGE_SIZE//2:, :ClassificationConfig.IMAGE_SIZE//2], magnitude[:ClassificationConfig.IMAGE_SIZE//2, ClassificationConfig.IMAGE_SIZE//2:], magnitude[ClassificationConfig.IMAGE_SIZE//2:, ClassificationConfig.IMAGE_SIZE//2:]
    hists = [np.bincount(b.ravel(), m.ravel(), ClassificationConfig.BIN_NUM) for b, m in zip(angle_cells, magnitude_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    feature =  hist / (hist.sum() + ClassificationConfig.HELLINGER_EPSILON)
    feature = np.sqrt(feature)
    feature /= np.linalg.norm(feature) + ClassificationConfig.HELLINGER_EPSILON
    
    return feature


def train_en(C:float, gamma:float) -> SVMWrapper:
    if GlobalConfig.INFO:
        print('[Classifier] Start training English SVM model...')
    start_time = time.time()
    model_en = SVMWrapper(C, gamma)
    features = []
    labels = []

    for root, dirs, files in os.walk(GlobalConfig.DATASET_EN_PATH):
        if os.path.basename(root) == GlobalConfig.DATASET_EN_TOP_PATH:
            continue
        
        label = ClassificationConfig.LABELS_EN[os.path.basename(root)]
        for filename in files:
            image = cv2.imread(os.path.join(root, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = classify_preprocess(image)
            features.append(feature)
            labels.append(label)
    
    features = np.float32(features)
    labels = np.int32(labels)
    model_en.train(features, labels)
    end_time = time.time()
    if GlobalConfig.INFO:
        print(f'[Classifier] English SVM model training finished in {end_time - start_time} seconds.')

    validation_result = model_en.predict(features)
    mask = validation_result==labels
    correct = np.count_nonzero(mask)
    if GlobalConfig.INFO:
        print(f'[Classifier] English SVM model validation accuracy: {correct*100.0 / validation_result.size}%')

    return model_en

def train_zh(C:float, gamma:float) -> SVMWrapper:
    if GlobalConfig.INFO:
        print('[Classifier] Start training Chinese SVM model...')
    start_time = time.time()
    model_zh = SVMWrapper(C, gamma)
    features = []
    labels = []

    for root, dirs, files in os.walk(GlobalConfig.DATASET_ZH_PATH):
        if os.path.basename(root) == GlobalConfig.DATASET_ZH_TOP_PATH:
            continue

        label = ClassificationConfig.LABELS_ZH[os.path.basename(root)][0]
        for filename in files:
            image = cv2.imread(os.path.join(root, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = classify_preprocess(image)
            features.append(feature)
            labels.append(label)
    
    features = np.float32(features)
    labels = np.int32(labels)
    model_zh.train(features, labels)
    end_time = time.time()
    if GlobalConfig.INFO:
        print(f'[Classifier] Chinese SVM model training finished in {end_time - start_time} seconds.')

    validation_result = model_zh.predict(features)
    mask = validation_result==labels
    correct = np.count_nonzero(mask)
    if GlobalConfig.INFO:
        print(f'[Classifier] Chinese SVM model validation accuracy: {correct*100.0 / validation_result.size}%')

    return model_zh


if __name__ == '__main__':
    thread_en = MyThread(train_en, (ClassificationConfig.EN_SVM_C, ClassificationConfig.EN_SVM_GAMMA))
    thread_zh = MyThread(train_zh, (ClassificationConfig.ZH_SVM_C, ClassificationConfig.ZH_SVM_GAMMA))
    thread_en.start()
    thread_zh.start()
    thread_en.join()
    thread_zh.join()
    model_en = thread_en.get_result()
    model_zh = thread_zh.get_result()
    model_en.save(GlobalConfig.SVM_EN_PATH)
    model_zh.save(GlobalConfig.SVM_ZH_PATH)
