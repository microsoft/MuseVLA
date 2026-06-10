# 使用官方的 ROS Noetic 基础镜像
FROM osrf/ros:noetic-desktop-full

# 设置非交互式安装，避免构建时卡住
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装必要依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# 设置conda环境变量
ENV PATH="/opt/miniconda/bin:$PATH"

# 初始化conda
RUN conda init bash

# 接受conda TOS条款
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建conda环境并安装pinocchio（必须用conda安装的包）
RUN conda create -n datahub python=3.8 -y
RUN echo "source activate datahub" >> ~/.bashrc

# 安装pinocchio（必须通过conda安装）
RUN /opt/miniconda/bin/conda install -n datahub conda-forge::pinocchio -y

# 创建工作目录
WORKDIR /app

# 使用pip在conda环境中安装所有必要的包
RUN /opt/miniconda/envs/datahub/bin/pip install --no-cache-dir numpy opencv-python tqdm pillow imageio PyYAML pycryptodomex scipy python-gnupg rospkg

# 设置默认命令，当容器启动时，会进入 bash
# 这样你就可以手动运行脚本了
CMD ["bash"]
