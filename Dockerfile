FROM winglian/axolotl-cloud:main-20240416-py3.10-cu121-2.1.2

WORKDIR /app

RUN apt-get update && apt-get install -y git protobuf-compiler && apt-get clean
RUN pip install --upgrade pip
RUN pip install --force-reinstall transformers protobuf sentencepiece 
RUN pip install --force-reinstall "numpy<2"

COPY . .

#ENV CUDA_VISIBLE_DEVICES=""
CMD ["python", "main.py"]