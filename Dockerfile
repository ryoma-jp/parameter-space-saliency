FROM nvcr.io/nvidia/pytorch:26.04-py3

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
	&& rm -f /tmp/requirements.txt

WORKDIR /work

