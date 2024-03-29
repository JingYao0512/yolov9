from pathlib import Path

import openvino.runtime as ov
from openvino.runtime import Layout, Type

import numpy as np
import cv2 as cv 
import argparse
import os


names=["nodule", "others"]



# python test.py --model=C:/Users/USER/Desktop/yolov9/runs/train/lung10/weights/best.xml --data_path=combine.png
def longEdgeImageProcess(image, long_edge_size = 1024, short_edge_size = 1024):

    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]  # width / height
    direction = ""
    print(image.shape[1], image.shape[0])
    # Determine the dimensions for resizing while maintaining aspect ratio
    if aspect_ratio >= 1: # width > height
        new_height = int(long_edge_size / aspect_ratio)
        new_width = long_edge_size 

        pad = short_edge_size - new_height
        direction = "horizontal"
        ratio = image.shape[1] / long_edge_size
    else:
        new_height = long_edge_size
        new_width = int(long_edge_size * aspect_ratio)
        print(f"new width: {new_width}")
        
        ratio = image.shape[0] / long_edge_size
        print(new_width)
        if(new_width%2 == 1):
            new_width += 1
        pad = short_edge_size - new_width

    # Resize the image
    resized_image = cv.resize(image, (new_width, new_height))
    print(resized_image.shape)
    # # Perform zero-padding on the long side
    # top_pad = 0
    inverse_flag = False
    top_pad = 0
    # print(f"pad: {pad}")
    # if aspect_ratio < 1:
    #     resized_image = np.transpose(resized_image, (1,0,2))
    #     inverse_flag = True

    print("shape:", resized_image.shape)
    if pad > 0:
        top_pad = pad // 2
        print(top_pad)
        bottom_pad = pad - top_pad
        if aspect_ratio >= 1:  
            resized_image = cv.copyMakeBorder(resized_image, top_pad, bottom_pad, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])     
        else:

            resized_image = cv.copyMakeBorder(resized_image, 0, 0, top_pad, bottom_pad, cv.BORDER_CONSTANT, value=[0, 0, 0])
    print(resized_image.shape)
    return resized_image, inverse_flag, direction, top_pad, ratio



def shortEdgeImageProcess(image, short_edge_size=1024, long_edge_size=640):

    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]  # width / height

    # Determine the dimensions for resizing while maintaining aspect ratio
    direction = ""
    if aspect_ratio >= 1:
        new_height = short_edge_size
        new_width = int(short_edge_size * aspect_ratio)
        pad = long_edge_size - new_width
        direction = "horizontal"
        ratio = image.shape[0] / short_edge_size
    else:
        new_width = short_edge_size
        new_height = int(short_edge_size / aspect_ratio)
        pad = long_edge_size - new_height
        direction = "veritical"
        ratio = image.shape[1] / short_edge_size

    # Resize the image
    resized_image = cv.resize(image, (new_width, new_height))



    # Perform zero-padding on the long side
    top_pad = 0
    inverse_flag = False
    if pad > 0:
        top_pad = pad // 2
        print(top_pad)

        bottom_pad = pad - top_pad
        if aspect_ratio >= 1:       
            resized_image = cv.copyMakeBorder(resized_image, 0, 0, top_pad, bottom_pad, cv.BORDER_CONSTANT, value=[0, 0, 0])
            resized_image = np.transpose(resized_image, (1,0,2))
            inverse_flag = True
        else:
            resized_image = cv.copyMakeBorder(resized_image, top_pad, bottom_pad, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])

    return resized_image, inverse_flag, direction, top_pad, ratio

#  ==========================================================================
#  ==========================================================================
#  ==========================================================================
def preprocess_warpAffine(image, dst_width = 640, dst_height = 640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
        ], dtype=np.float32)

    img_pre = cv.warpAffine(image, M, (dst_width, dst_height), flags = cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT, borderValue=(114,114,114))

    IM = cv.invertAffineTransform(M)

    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]

    return img_pre, IM

def postprocess(pred, IM=[], conf_thres=0.8, iou_thres=0.45):

    boxes = []
    for item in pred:
        cx, cy, w, h = item[:4]
        label = item[4:].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue

        left   = cx - w * 0.5
        top    = cy - h * 0.5
        right  = cx + w * 0.5
        bottom = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label])

    boxes = np.array(boxes)
    if(boxes.shape[0] == 0):
        return NMS(boxes, iou_thres)
    lr = boxes[:, [0, 2]]
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]
    boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)

    return NMS(boxes, iou_thres)

def iou(box1, box2):

    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    left   = max(box1[0], box2[0])
    top    = max(box1[1], box2[1])
    right  = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])

    union = max((right - left), 0) * max((bottom - top), 0)
    cross = area_box(box1) + area_box(box2) - union

    if( cross == 0 or union == 0):
        return 0 
    return union / cross  

def NMS(boxes, iou_thres):

    remove_flags = [False] * len(boxes)


    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue

        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if(remove_flags):
                continue

            jbox = boxes[j]
            if(ibox[5] != jbox[5]):
                continue

            if(iou(ibox, jbox) > iou_thres):
                remove_flags[j] = True

    return keep_boxes

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)                



def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

if __name__ == "__main__":
    #main()
    modelpath = "C:/Users/USER/Desktop/yolov9/runs/train/lung10/weights/best.xml"

    # present = datetime.datetime.now()
    core = ov.Core()
    ov_model = core.read_model(modelpath)
    compiled_model = core.compile_model(ov_model, "AUTO")
    infer_request = compiled_model.create_infer_request()

    test_root_dir = "C:/Users/USER/Desktop/NCKU/dataset/test/nodule"
    positive_count = 0 
    negative_count = 0
    positive_samples = []
    loaded_list = np.load('positive_samples.npy')
    for test_image_path in loaded_list:
        
        print("="*64)
        print(f"Image Path:{test_image_path}")
        image = cv.imread(test_image_path)
        img_pre, IM = preprocess_warpAffine(image)
        # image, inverse_flag, direction, top_pad, ratio = longEdgeImageProcess(image, 640, 640)
        
        # print(f"image shape: {img_pre.shape}")
        # image = np.transpose(image, (2,0,1))
        # image = image[np.newaxis,:,:,:]

        input_tensor = ov.Tensor(array=img_pre.astype(np.float32), shared_memory=False)
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()

        #print( np.squeeze(infer_request.get_output_tensor(0).data)[:,0])

        detections = np.squeeze(infer_request.get_output_tensor(0).data).T


        boxes = postprocess(detections, IM)
        if(len(boxes) > 0):
            positive_count += 1
            #positive_samples.append(test_image_path)
        else:
            negative_count += 1

        for obj in boxes:
            left, top, right, bottom =  int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            cv.rectangle(image, (left, top), (right, bottom), color=color, thickness=2, lineType=cv.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            #print(caption)
            w, h = cv.getTextSize(caption, 0, 1, 2)[0]
            cv.rectangle(image, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv.putText(image, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    #cv.imwrite("infer.png", image)
    print("="*64)
    print(f"Positive Nodule Samples: {positive_count}")
    print(f"negative Nodule Samples: {negative_count}")


   