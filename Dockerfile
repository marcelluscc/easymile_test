FROM tensorflow/tensorflow:1.15.2-gpu-py3
COPY . .
RUN pip install -r requirements.txt
CMD [ "python", "src/train.py" ]