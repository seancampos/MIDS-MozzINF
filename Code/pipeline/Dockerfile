FROM nvcr.io/nvidia/tensorflow:22.06-tf1-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1

RUN pip3 install -U pip
RUN pip3 install "librosa==0.9.2" "prefect>=2.0b" "scikit-image==0.19.2" "prefect-dask"
RUN prefect config set PREFECT_ORION_UI_API_URL=http://100.77.119.160:4200/api

CMD ["prefect", "orion", "start", "--host", "0.0.0.0"]
