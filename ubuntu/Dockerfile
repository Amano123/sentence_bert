FROM tensorflow/tensorflow:2.7.3-gpu
LABEL maintainer="amano123"

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV USER ${USER_NAME}
ENV HOME /home/${USER}

ENV https_proxy "http://wwwproxy.osakac.ac.jp:8080"
ENV http_proxy "http://wwwproxy.osakac.ac.jp:8080"

# サーバーを日本に変更
# 多分早くなる
# RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# #パッケージインストール
RUN apt-get update \
&&  apt-get install -y --no-install-recommends \
                sudo \
                git \
                #japanase
                # language-pack-ja-base \
                # language-pack-ja \
                iputils-ping \
		locales  \
                python3 \
                python3-pip \
                gcc
                
# 日本語化
# RUN locale-gen ja_JP.UTF-8 
# ENV LANG ja_JP.UTF-8
# ENV LC_CTYPE ja_JP.UTF-8
# RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8


## make user
RUN set -x \
&&  echo "Starting with UID : ${USER_ID}, GID: ${GROUP_ID}" \
&&  adduser --disabled-password --gecos "" "${USER_NAME}" \
&&  echo "${USER_NAME}:${USER_NAME}" | chpasswd \
&&  echo "%${USER_NAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USER_NAME}  \
&&  chmod 0440 /etc/sudoers.d/${USER_NAME} \
&&  usermod -u ${USER_ID} ${USER_NAME} \
&&  groupmod -g ${GROUP_ID} ${USER_NAME} 

# ディレクトリを指定
USER $USER_NAME

# RUN pip install setuptools
RUN python3 -m pip install pandas tqdm \
        sentence-transformers \
        ipykernel

WORKDIR ${HOME}/workspace