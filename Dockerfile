FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt update &&\
    apt upgrade -y &&\
    apt install -y libgl1 &&\
    apt install -y libglib2.0-0 &&\
    apt install -y gdal-bin &&\
    apt install -y python3-pip

WORKDIR /app

ADD requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

ADD helpers/*.py helpers/
ADD scripts/*.py scripts/

ADD setup.py .
RUN pip install .

ENTRYPOINT [""]
CMD ["stdl-objdet", "-h"]