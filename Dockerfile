FROM registry.lapig.iesa.ufg.br/lapig-images-prod/time_series_sentinel:base

LABEL maintainer="Renato Gomes <renatogomessilverio@gmail.com>"

# Clone app and npm install on server
ENV URL_TO_APPLICATION_GITHUB="https://github.com/spatialive/timeseries.git"
ENV BRANCH="main"

RUN apt-get update && apt-get -y install figlet procps net-tools curl python3-dev build-essential wget git && \
    if [ -d "/APP/timeseries" ]; then rm -Rf /APP/timeseries; fi && \
    mkdir -p /APP && cd /APP && git clone -b ${BRANCH} ${URL_TO_APPLICATION_GITHUB} && \
    cd timeseries/ && pip3 install -r requirements.txt && \
    chmod +x /APP/timeseries/api.py
    
WORKDIR /APP

CMD [ "/bin/bash", "-c", "cd /APP/timeseries && python3 api.py; tail -f /dev/null"]

ENTRYPOINT [ "/APP/Monitora.sh"]
