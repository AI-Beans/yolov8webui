# Gradio YOLOv8 Det v0.2.5
# 创建人：曾逸夫
# 创建时间：2023-10-21
# pip install gradio==3.50.2

import argparse
import csv
import gc
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import font_manager
from ultralytics import YOLO

# pip_source = "https://pypi.tuna.tsinghua.edu.cn/simple some-package"
# os.system(f"pip install gradio --upgrade -i {pip_source}")
# os.system(f"pip install ultralytics --upgrade -i {pip_source}")

ROOT_PATH = sys.path[0]  # 项目根目录

# --------------------- 字体库 ---------------------
SimSun_path = f"{ROOT_PATH}/fonts/SimSun.ttf"  # 宋体文件路径
TimesNesRoman_path = f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"  # 新罗马字体文件路径
# 宋体
SimSun = font_manager.FontProperties(fname=SimSun_path, size=12)
# 新罗马字体
TimesNesRoman = font_manager.FontProperties(fname=TimesNesRoman_path, size=12)

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from util.fonts_opt import is_fonts

ROOT_PATH = sys.path[0]  # 根目录

# Gradio YOLOv8 Det版本
GYD_VERSION = "Gradio YOLOv8 Det v0.2.5"

# 文件后缀
suffix_list = [".csv", ".yaml"]

# 字体大小
FONTSIZE = 25

# 目标尺寸
obj_style = ["小目标", "中目标", "大目标"]


def parse_args(known=False):
    parser = argparse.ArgumentParser(description="Gradio YOLOv8 Det v0.2.5")
    parser.add_argument("--model_type", "-mt", default="online", type=str, help="model type")
    parser.add_argument("--source", "-src", default="upload", type=str, help="image input source")
    parser.add_argument("--source_video", "-src_v", default="upload", type=str, help="video input source")
    parser.add_argument("--img_tool", "-it", default="editor", type=str, help="input image tool")
    parser.add_argument("--model_name", "-mn", default="yolov8s", type=str, help="model name")
    parser.add_argument(
        "--model_cfg",
        "-mc",
        default="./model_config/model_name_all.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--cls_name",
        "-cls",
        default="./cls_name/cls_name_zh.yaml",
        type=str,
        help="cls name",
    )
    parser.add_argument(
        "--nms_conf",
        "-conf",
        default=0.5,
        type=float,
        help="model NMS confidence threshold",
    )
    parser.add_argument("--nms_iou", "-iou", default=0.45, type=float, help="model NMS IoU threshold")
    parser.add_argument(
        "--device",
        "-dev",
        default="cuda:0",
        type=str,
        help="cuda or cpu",
    )
    parser.add_argument("--inference_size", "-isz", default=640, type=int, help="model inference size")
    parser.add_argument("--max_detnum", "-mdn", default=50, type=float, help="model max det num")
    parser.add_argument("--slider_step", "-ss", default=0.05, type=float, help="slider step")
    parser.add_argument(
        "--is_login",
        "-isl",
        action="store_true",
        default=False,
        help="is login",
    )
    parser.add_argument('--usr_pwd',
                        "-up",
                        nargs='+',
                        type=str,
                        default=["admin", "admin"],
                        help="user & password for login")
    parser.add_argument(
        "--is_share",
        "-is",
        action="store_true",
        default=False,
        help="is login",
    )
    parser.add_argument("--server_port", "-sp", default=7861, type=int, help="server port")

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


# yaml文件解析
def yaml_parse(file_path):
    return yaml.safe_load(open(file_path, encoding="utf-8").read())


# yaml csv 文件解析
def yaml_csv(file_path, file_tag):
    file_suffix = Path(file_path).suffix
    if file_suffix == suffix_list[0]:
        # 模型名称
        file_names = [i[0] for i in list(csv.reader(open(file_path)))]  # csv版
    elif file_suffix == suffix_list[1]:
        # 模型名称
        file_names = yaml_parse(file_path).get(file_tag)  # yaml版
    else:
        print(f"{file_path}格式不正确！程序退出！")
        sys.exit()

    return file_names


# 检查网络连接
def check_online():
    # 参考：https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


# 模型加载
def model_loading(img_path, conf, iou, infer_size, yolo_model="yolov8n.pt"):
    model = YOLO(yolo_model)

    results = model(source=img_path, imgsz=infer_size, conf=conf, iou=iou)
    results = list(results)[0]
    return results


# 标签和边界框颜色设置
def color_set(cls_num):
    color_list = []
    for i in range(cls_num):
        color = tuple(np.random.choice(range(256), size=3))
        # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        color_list.append(color)

    return color_list


# 随机生成浅色系或者深色系
def random_color(cls_num, is_light=True):
    color_list = []
    for i in range(cls_num):
        color = (
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
        )
        color_list.append(color)

    return color_list


# 检测绘制
def pil_draw(img, score_l, bbox_l, cls_l, cls_index_l, textFont, color_list):
    img_pil = ImageDraw.Draw(img)
    id = 0

    for score, (xmin, ymin, xmax, ymax), label, cls_index in zip(score_l, bbox_l, cls_l, cls_index_l):
        img_pil.rectangle([xmin, ymin, xmax, ymax], fill=None, outline=color_list[cls_index], width=2)  # 边界框
        countdown_msg = f"{id}-{label} {score:.2f}"
        # text_w, text_h = textFont.getsize(countdown_msg)  # 标签尺寸 pillow 9.5.0
        # left, top, left + width, top + height
        # 标签尺寸 pillow 10.0.0
        text_xmin, text_ymin, text_xmax, text_ymax = textFont.getbbox(countdown_msg)
        # 标签背景
        img_pil.rectangle(
            # (xmin, ymin, xmin + text_w, ymin + text_h), # pillow 9.5.0
            (xmin, ymin, xmin + text_xmax - text_xmin, ymin + text_ymax - text_ymin),  # pillow 10.0.0
            fill=color_list[cls_index],
            outline=color_list[cls_index],
        )

        # 标签
        img_pil.multiline_text(
            (xmin, ymin),
            countdown_msg,
            fill=(0, 0, 0),
            font=textFont,
            align="center",
        )

        id += 1

    return img


# 绘制多边形
def polygon_drawing(img_mask, canvas, color_seg):
    # ------- RGB转BGR -------
    color_seg = list(color_seg)
    color_seg[0], color_seg[2] = color_seg[2], color_seg[0]
    color_seg = tuple(color_seg)
    # 定义多边形的顶点
    pts = np.array(img_mask, dtype=np.int32)

    # 多边形绘制
    cv2.drawContours(canvas, [pts], -1, color_seg, thickness=-1)


# 输出分割结果
def seg_output(img_path, seg_mask_list, color_list, cls_list):
    img = cv2.imread(img_path)
    img_c = img.copy()

    # w, h = img.shape[1], img.shape[0]

    # 获取分割坐标
    for seg_mask, cls_index in zip(seg_mask_list, cls_list):
        img_mask = []
        for i in range(len(seg_mask)):
            # img_mask.append([seg_mask[i][0] * w, seg_mask[i][1] * h])
            img_mask.append([seg_mask[i][0], seg_mask[i][1]])

        polygon_drawing(img_mask, img_c, color_list[int(cls_index)])  # 绘制分割图形

    img_mask_merge = cv2.addWeighted(img, 0.3, img_c, 0.7, 0)  # 合并图像

    return img_mask_merge


# YOLOv8图片检测函数
def yolo_det_img(img_path, model_name, infer_size, conf, iou):

    global model, model_name_tmp, device_tmp

    s_obj, m_obj, l_obj = 0, 0, 0

    area_obj_all = []  # 目标面积

    score_det_stat = []  # 置信度统计
    bbox_det_stat = []  # 边界框统计
    cls_det_stat = []  # 类别数量统计
    cls_index_det_stat = []  # 类别索引统计

    # 模型加载
    predict_results = model_loading(img_path, conf, iou, infer_size, yolo_model=f"{model_name}.pt")
    # 检测参数
    xyxy_list = predict_results.boxes.xyxy.cpu().numpy().tolist()
    conf_list = predict_results.boxes.conf.cpu().numpy().tolist()
    cls_list = predict_results.boxes.cls.cpu().numpy().tolist()

    # 颜色列表
    color_list = random_color(len(model_cls_name_cp), True)

    # 图像分割
    if (model_name[-3:] == "seg"):
        # masks_list = predict_results.masks.xyn
        masks_list = predict_results.masks.xy
        img_mask_merge = seg_output(img_path, masks_list, color_list, cls_list)
        img = Image.fromarray(cv2.cvtColor(img_mask_merge, cv2.COLOR_BGRA2RGBA))
    else:
        img = Image.open(img_path)

    # 判断检测对象是否为空
    if (xyxy_list != []):

        # ---------------- 加载字体 ----------------
        yaml_index = cls_name.index(".yaml")
        cls_name_lang = cls_name[yaml_index - 2:yaml_index]

        if cls_name_lang == "zh":
            # 中文
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/SimSun.ttf"), size=FONTSIZE)
        elif cls_name_lang in ["en", "ru", "es", "ar"]:
            # 英文、俄语、西班牙语、阿拉伯语
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"), size=FONTSIZE)
        elif cls_name_lang == "ko":
            # 韩语
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/malgun.ttf"), size=FONTSIZE)

        for i in range(len(xyxy_list)):
            obj_cls_index = int(cls_list[i])  # 类别索引
            cls_index_det_stat.append(obj_cls_index)

            obj_cls = model_cls_name_cp[obj_cls_index]  # 类别
            cls_det_stat.append(obj_cls)

            # ------------ 边框坐标 ------------
            x0 = int(xyxy_list[i][0])
            y0 = int(xyxy_list[i][1])
            x1 = int(xyxy_list[i][2])
            y1 = int(xyxy_list[i][3])

            bbox_det_stat.append((x0, y0, x1, y1))

            conf = float(conf_list[i])  # 置信度
            score_det_stat.append(conf)

            # ---------- 加入目标尺寸 ----------
            w_obj = x1 - x0
            h_obj = y1 - y0
            area_obj = w_obj * h_obj
            area_obj_all.append(area_obj)

        det_img = pil_draw(img, score_det_stat, bbox_det_stat, cls_det_stat, cls_index_det_stat, textFont, color_list)

        # -------------- 目标尺寸计算 --------------
        for i in range(len(area_obj_all)):
            if (0 < area_obj_all[i] <= 32 ** 2):
                s_obj = s_obj + 1
            elif (32 ** 2 < area_obj_all[i] <= 96 ** 2):
                m_obj = m_obj + 1
            elif (area_obj_all[i] > 96 ** 2):
                l_obj = l_obj + 1

        sml_obj_total = s_obj + m_obj + l_obj
        objSize_dict = {}
        objSize_dict = {obj_style[i]: [s_obj, m_obj, l_obj][i] / sml_obj_total for i in range(3)}

        # ------------ 类别统计 ------------
        clsRatio_dict = {}
        clsDet_dict = Counter(cls_det_stat)
        clsDet_dict_sum = sum(clsDet_dict.values())
        for k, v in clsDet_dict.items():
            clsRatio_dict[k] = v / clsDet_dict_sum

        return det_img, objSize_dict, clsRatio_dict
    else:
        print("图片目标不存在！")
        return None, None, None


def main(args):
    gr.close_all()

    global model_cls_name_cp, cls_name

    source = args.source
    img_tool = args.img_tool
    nms_conf = args.nms_conf
    nms_iou = args.nms_iou
    model_name = args.model_name
    model_cfg = args.model_cfg
    cls_name = args.cls_name
    inference_size = args.inference_size
    slider_step = args.slider_step

    is_fonts(f"{ROOT_PATH}/fonts")  # 检查字体文件

    model_names = yaml_csv(model_cfg, "model_names")  # 模型名称
    model_cls_name = yaml_csv(cls_name, "model_cls_name")  # 类别名称

    model_cls_name_cp = model_cls_name.copy()  # 类别名称

    # ------------------- 图片模式输入组件 -------------------
    inputs_img = gr.Image(image_mode="RGB", source=source, tool=img_tool, type="filepath", label="原始图片")
    inputs_model01 = gr.Dropdown(choices=model_names, value=model_name, type="value", label="模型")
    inputs_size01 = gr.Slider(384, 1536, step=128, value=inference_size, label="推理尺寸")
    input_conf01 = gr.Slider(0, 1, step=slider_step, value=nms_conf, label="置信度阈值")
    inputs_iou01 = gr.Slider(0, 1, step=slider_step, value=nms_iou, label="IoU 阈值")

    # ------------------- 图片模式输入参数 -------------------
    inputs_img_list = [
        inputs_img,  # 输入图片
        inputs_model01,  # 模型
        inputs_size01,  # 推理尺寸
        input_conf01,  # 置信度阈值
        inputs_iou01,  # IoU阈值
    ]

    # ------------------- 图片模式输出组件 -------------------
    outputs_img = gr.Image(type="pil", label="检测图片")
    outputs_objSize = gr.Label(label="目标尺寸占比统计")
    outputs_clsSize = gr.Label(label="类别检测占比统计")

    # ------------------- 图片模式输出参数 -------------------
    outputs_img_list = [outputs_img, outputs_objSize, outputs_clsSize]

    # 标题
    title = "Gradio YOLOv8 Det"

    # 描述
    description = "<div align='center'>基于 YOLOv8 的目标检测与图像分割系统</div>"

    # 示例图片
    examples_imgs = [
        [
            "./img_examples/bus.jpg",
            "yolov8s",
            640,
            0.6,
            0.5,],
        [
            "./img_examples/giraffe.jpg",
            "yolov8l",
            320,
            0.5,
            0.45,],
        [
            "./img_examples/zidane.jpg",
            "yolov8m",
            640,
            0.6,
            0.5,],
        [
            "./img_examples/Millenial-at-work.jpg",
            "yolov8x",
            1280,
            0.5,
            0.5,],
        [
            "./img_examples/bus.jpg",
            "yolov8s-seg",
            640,
            0.6,
            0.5,],
        [
            "./img_examples/Millenial-at-work.jpg",
            "yolov8x-seg",
            1280,
            0.5,
            0.5,],]

    # 接口
    gyd_img = gr.Interface(
        fn=yolo_det_img,
        inputs=inputs_img_list,
        outputs=outputs_img_list,
        title=title,
        description=description,
        examples=examples_imgs,
        cache_examples=False,
        flagging_dir="run",  # 输出目录
        allow_flagging="manual",
        flagging_options=["good", "generally", "bad"],
    )

    return gyd_img


if __name__ == "__main__":
    args = parse_args()
    gyd_img = main(args)
    is_share = args.is_share

    gyd_img.launch(
        inbrowser=True,  # 自动打开默认浏览器
        show_tips=True,  # 自动显示gradio最新功能
        share=is_share,  # 项目共享，其他设备可以访问
        favicon_path="./icon/logo.ico",  # 网页图标
        show_error=True,  # 在浏览器控制台中显示错误信息
        quiet=True,  # 禁止大多数打印语句
    )
