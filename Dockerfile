FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG UID=1000
ARG GID=1000
ARG USERNAME=pss

RUN pip install notebook seaborn==0.13.1

RUN groupadd --gid ${GID} ${USERNAME} \
	&& useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME} \
	&& mkdir -p /work \
	&& chown -R ${UID}:${GID} /home/${USERNAME} /work

ENV HOME=/home/${USERNAME}
ENV SHELL=/bin/bash

WORKDIR /work
USER ${USERNAME}

