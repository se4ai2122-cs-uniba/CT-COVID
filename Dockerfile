FROM nvidia/cuda:11.4.0-base-ubuntu20.04
COPY requirements.txt requirements.txt
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y python3 && apt install -y python3-pip \
&& python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
WORKDIR /CT-COVID
COPY . /CT-COVID
EXPOSE 5000
CMD ["python3", "src/api.py"]