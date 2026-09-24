FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt /var/task/
RUN pip install --no-cache-dir -r /var/task/requirements.txt

COPY model.onnx labels.json /var/task/
COPY lambda_function.py /var/task/
CMD [ "lambda_function.lambda_handler" ]
