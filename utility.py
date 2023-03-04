from PIL import Image, ImageDraw
import xml.dom.minidom
import numpy as np


def check_yolo_label(image, x_center_yolo, y_center_yolo, width_drone_yolo, height_drone_yolo):
    width, height = image.size
    x0 = (x_center_yolo - width_drone_yolo / 2) * width
    y0 = (y_center_yolo - height_drone_yolo / 2) * height

    x1 = (x_center_yolo + width_drone_yolo / 2) * width
    y1 = (y_center_yolo - height_drone_yolo / 2) * height

    x2 = (x_center_yolo + width_drone_yolo / 2) * width
    y2 = (y_center_yolo + height_drone_yolo / 2) * height

    x3 = (x_center_yolo - width_drone_yolo / 2) * width
    y3 = (y_center_yolo + height_drone_yolo / 2) * height

    draw_image = ImageDraw.Draw(image)
    draw_image.line(((x0, y0), (x1, y1), (x2, y2), (x3, y3), (x0, y0)), fill=(255, 0, 0))
    image.show()


def parse_xml(path_xml):
    dom = xml.dom.minidom.parse(path_xml)
    coords_drons = []
    for i in range(len(dom.getElementsByTagName('xmin'))):
        xmin = int(dom.getElementsByTagName('xmin')[i].childNodes[0].data)
        ymin = int(dom.getElementsByTagName('ymin')[i].childNodes[0].data)
        xmax = int(dom.getElementsByTagName('xmax')[i].childNodes[0].data)
        ymax = int(dom.getElementsByTagName('ymax')[i].childNodes[0].data)
        coords_drons.append((xmin, ymin, xmax, ymax))

    width = int(dom.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(dom.getElementsByTagName('height')[0].childNodes[0].data)
    return coords_drons, width, height


def convert_yolo(coords):
    x_center = float(coords[0])
    y_center = float(coords[1])
    width = float(coords[2])
    height = float(coords[3])
    return np.array([x_center - width, y_center - height, x_center + width, y_center + height])


def get_iou_yolo(ground_truth, pred):
    return get_iou(convert_yolo(ground_truth), convert_yolo(pred))


def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.

    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
