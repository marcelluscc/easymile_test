FROM tensorflow/tensorflow:1.15.2-gpu-py3
COPY src /work/src
WORKDIR /work
RUN mkdir -p saved_model
# RUN pip install -r requirements.txt
CMD [ "bash" ]