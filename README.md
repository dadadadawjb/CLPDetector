# CLPDetector
A color-shape and shape-color two pipelines based Chinese License Plates Detector.
Aim at both taking the efficient advantages of traditional handcraft features extraction computer vision algorithms to make up for deep learning based algorithms, and making use of multiple priori features instead of single feature to improve the robustness and accuracy of the detection algorithm.

Task:

![LicensePlateTask](assets/LicensePlateDetection.png)

CLPDetector pipeline:

![CLPDetectorPipeline](assets/CLPDetectorPipeline.png)

## Support Features
* Color-based RoI proposal and shape-based RoI proposal
* Shape-based plate rectification and color-based plate rectification
* Wave-based characters split
* SVM-based characters classification
* Two pipelines parallel threading
* Easy to use GUI

## Get Started
```bash
# install dependencies
pip install -r requirements.txt

# train the SVM models
python classifier.py

# test all images together
python detector.py

# run the GUI CLPDetector
python main.py
```

## Demo
![Demo](assets/demo.gif)

## Results
* Classifier Result

![ClassifierResult](assets/ClassifierResult.png)

* Detector Result

![DetectorResult](assets/DetectorResult.png)

* Detection Test Result

![DetectionResult1](assets/Result1.png)
![DetectionResult1](assets/Result2.png)
![DetectionResult1](assets/Result3.png)
![DetectionResult1](assets/Result4.png)
![DetectionResult1](assets/Result5.png)
![DetectionResult1](assets/Result6.png)
![DetectionResult1](assets/Result7.png)
![DetectionResult1](assets/Result8.png)
![DetectionResult1](assets/Result9.png)

* Visualization Result

![Visualization](assets/Visualization.png)

## Structures
* Color-based RoI Proposal

![ColorRoIProposal](assets/ColorRoIProposal.png)

* Shape-based RoI Proposal

![ShapeRoIProposal](assets/ShapeRoIProposal.png)

* Color-based Rectification

![ColorRectification](assets/ColorRectification.png)

* Shape-based Rectification

![ShapeRectification](assets/ShapeRectification.png)

* Wave-based Characters Split

![CharactersSplit](assets/CharactersSplit.png)

* SVM-based Characters Classification

![CharactersClassification](assets/CharactersClassification.png)

## TODO List
- [x] core detection algorithms implementation
  - [x] color-based ROI proposal
  - [x] shape-based ROI proposal
  - [x] rectification for ROIs from color proposal
  - [x] rectification for ROIs from shape proposal
  - [x] characters split
  - [x] characters classification
  - [x] characters classifier models training
- [x] core detection algorithms refactor (`detector.py - ChineseLicensePlateDetector`)
  - [x] detect wrapper (`detect()`)
    - [x] color shape pipeline (`color_shape_pipeline()`)
      - [x] preprocess (`color_shape_preprocess()`)
      - [x] color RoI proposal (`color_RoI_proposal()`)
      - [x] shape rectification (`shape_rectification()`)
      - [x] characters split (`color_shape_characters_preprocess()`)
      - [x] characters classification (`characters_classify()`, `classifier.py`)
    - [x] shape color pipeline (`shape_color_pipeline()`)
      - [x] preprocess (`shape_color_preprocess()`)
      - [x] shape RoI proposal (`shape_RoI_proposal()`)
      - [x] color rectification (`color_rectification()`)
      - [x] characters split (`shape_color_characters_preprocess()`)
      - [x] characters classification (`characters_classify()`, `classifier.py`)
- [x] GUI design
  - [x] main window (original image, detection results)
  - [x] visualization window (two pipelines intermediate results show)
