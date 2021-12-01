FROM nvidia/cuda:11.4.0-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y ffmpeg libsm6 libxext6 python3 python3-pip
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
WORKDIR /CT-COVID
COPY . /CT-COVID
EXPOSE 5000
CMD ["python3", "src/api.py"]
