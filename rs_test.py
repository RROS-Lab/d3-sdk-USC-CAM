## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
## Credit to Intel and Addison Sears-Collins for some of this code

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

classes = ["background", "person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
           "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
           "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Colors we will use for the object labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

pb = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

cvNet = cv2.dnn.readNetFromTensorflow(pb, pbt)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        rows = color_image.shape[0]
        cols = color_image.shape[1]
        cvNet.setInput(cv2.dnn.blobFromImage(color_image, size=(300, 300), swapRB=True, crop=False))

        # Run object detection
        cvOut = cvNet.forward()

        # Go through each object detected and label it
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.3:

                idx = int(detection[1])  # prediction class index.


                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv2.rectangle(color_image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(classes[idx], score * 100)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(color_image, label, (int(left), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

                x = (left + right) / 2
                y = (top + bottom) / 2
                cv2.circle(color_image, (int(x), int(y)), 30, (191, 64, 191), thickness=10)
                cv2.circle(depth_image, (int(x), int(y)), 5, (191, 64, 191), thickness=1)
                object_depth = depth_frame.get_distance(int(x),int(y))
                print(label + " Centroid location: (" + str(x) + "," + str(y) + "," + str(object_depth) + ")")

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))



        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()