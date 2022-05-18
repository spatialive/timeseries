FROM python:3.8.13-alpine3.15

LABEL maintainer="Renato Gomes <renatogomessilverio@gmail.com>"

# Clone app and npm install on server
ENV URL_TO_APPLICATION_GITHUB="https://github.com/lapig-ufg/timeseries.git"
ENV BRANCH="main"

RUN /bin/sh -c "apk add --no-cache bash" && \
    apk update && apk add figlet git curl wget  && \
    mkdir -p /APP && cd /APP && git clone -b ${BRANCH} ${URL_TO_APPLICATION_GITHUB} && \
    cd timeseries/ && pip3 install -r requirements.txt && \
    echo 'figlet -t "Lapig Docker Timeseries"' >> ~/.bashrc && \
    chmod +x /APP/timeseries/api.py

WORKDIR /APP

CMD [ "/bin/bash", "-c", "cd /APP/timeseries && python3 api.py; tail -f /dev/null"]

ENTRYPOINT [ "/APP/Monitora.sh"]
