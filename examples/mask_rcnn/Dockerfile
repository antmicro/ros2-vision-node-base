# syntax=docker/dockerfile:1
FROM ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo

RUN $PIPINST --index-url https://download.pytorch.org/whl/cu118 \
    torch \
    torchvision \
    typing_extensions==4.4.0

# LibTorch
RUN curl https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip --output libtorch.zip && \
    unzip -q libtorch.zip && \
    cp -r libtorch/include/* /usr/include/ && \
    cp -r libtorch/lib/* /usr/lib/ && \
    cp -r libtorch/share/cmake/* /usr/share/cmake/ && \
    rm -rf libtorch libtorch.zip

RUN $PIPINST \
    "git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02" \
    "kenning[object_detection,reports] @ git+https://github.com/antmicro/kenning.git" && \
    python3 -m pip uninstall -y kenning opencv-python
