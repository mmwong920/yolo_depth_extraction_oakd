#!/usr/bin/env python3

import cv2
import depthai as dai
import onnxruntime
import numpy as np
import torch
import tensorflow as tf

# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
# session = ort.InferenceSession('yolov5s.onnx', providers=providers)
session = onnxruntime.InferenceSession("best.onnx",providers=['CPUExecutionProvider'])
first_input_name = session.get_inputs()[0].name
first_output_name = session.get_outputs()[0].name
print(first_input_name)
print(first_output_name)


# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        results = session.run([first_output_name],{str(first_input_name):np.expand_dims(inRgb.getFrame().astype("float32"),axis=0)})
        results = torch.from_numpy(results)
        out = tf.image.non_max_suppression(output, conf_thres=0.7, iou_thres=0.5)
        print(results)
        # print(np.array(inRgb.getFrame()).shape)
        cv2.imshow("rgb", inRgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break