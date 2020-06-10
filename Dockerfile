FROM tensorflow/tensorflow:1.15.2-gpu-py3
WORKDIR /work
COPY src src
COPY training_checkpoints training_checkpoints
COPY requirements.txt requirements.txt
COPY sample sample
RUN mkdir -p saved_model
RUN pip install -r requirements.txt
CMD [ "bash" ]