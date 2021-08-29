FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install g++ python3-pip python3.7-dev git  -y
RUN ln -s /usr/bin/python3.7 /usr/bin/python
ENV DEBIAN_FRONTEND=noninteractive
RUN python -m pip install torch==1.4.0 torchvision==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scikit-build -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install cmake -y
RUN python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install attr -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install jittor -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m jittor.test.test_example

COPY . /workspace
# 确定容器启动时程序运行路径
WORKDIR /workspace
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3
# 确定容器启动命令。以 python 示例，python 表示编译器，run.py 表示执 行文件，/input_path 和/output_path 为容器内绝对路径，测评时会自动将 测试数据挂载到容器内/input_path 路径，无需修改
CMD ["python", "run.py", "/input_path", "/output_path"]
# sudo docker build -t segform_sar .
# sudo docker run --rm -it --gpus all -v /home/gmh/project/yizhang/PFSegNets/gaofen/sar/val/image:/input_path -v test_img:/output_path segform_sar
# sudo docker login --username=2570219563@qq.com registry.cn-beijing.aliyuncs.com
# Aly970819.
# sudo docker tag segform_sar registry.cn-beijing.aliyuncs.com/glotwo/sar:segform
# sudo docker push registry.cn-beijing.aliyuncs.com/glotwo/sar:segform