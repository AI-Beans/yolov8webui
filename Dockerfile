# Gradio YOLOv8 Det, GPL-3.0 License
# 创建人：KKD
# 创建时间：2023-12-06

# 使用合适的基础镜像
FROM python:3.9

# 将工作目录设置为应用程序根目录
WORKDIR /app

# 将应用程序代码复制到镜像中的/app目录
COPY . /app


# 安装必要的依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序需要的端口
EXPOSE 5566

# 运行应用程序
CMD ["python", "gradio_yolov8_det_v1.py"]