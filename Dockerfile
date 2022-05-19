FROM python:3.8.13-slim

LABEL maintainer="Renato Gomes <renatogomessilverio@gmail.com>"

# Clone app and npm install on server
ENV URL_TO_APPLICATION_GITHUB="https://github.com/spatialive/timeseries.git"
ENV BRANCH="main"


RUN apt-get update && apt-get -y install figlet procps net-tools curl python3-dev build-essential wget git && \
    mkdir -p /APP && cd /APP && git clone -b ${BRANCH} ${URL_TO_APPLICATION_GITHUB} && \
    cd timeseries/ && pip3 install -r requirements.txt && \
    echo 'figlet -t "Lapig Docker Timeseries Sentinel"' >> ~/.bashrc && \
    chmod +x /APP/timeseries/api.py
    
WORKDIR /APP

CMD [ "/bin/bash", "-c", "cd /APP/timeseries && python3 api.py; tail -f /dev/null"]

ENTRYPOINT [ "/APP/Monitora.sh"]
