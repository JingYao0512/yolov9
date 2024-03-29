import onnxruntime as rt
import openvino as ov
import numpy as np
import datetime
import cv2 as cv

modelpath = "C:/Users/USER/Desktop/yolov9/runs/train/lung2/weights/best.onnx"

## transfer model from onnx to openvino
ov_model = ov.convert_model(modelpath)
ov.save_model(ov_model, modelpath.replace("best.onnx","best.xml"))
print("Finish!")


