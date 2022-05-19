FROM registry.lapig.iesa.ufg.br/lapig-images-prod/time_series_sentinel:latest

LABEL maintainer="Renato Gomes <renatogomessilverio@gmail.com>"

# Clone app and npm install on server
ENV URL_TO_APPLICATION_GITHUB="https://github.com/spatialive/timeseries.git"
ENV BRANCH="main"

RUN /bin/sh -c "apk add --no-cache bash build-dependencies musl-dev linux-headers g++ gcc gfortran python3-dev \
    py-pip build-base wget freetype-dev libpng-dev openblas-dev" && \
    apk update && apk add figlet git curl wget  && \
    if [ -d "/APP/timeseries" ]; then rm -Rf /APP/timeseries; fi && \
    mkdir -p /APP && cd /APP && git clone -b ${BRANCH} ${URL_TO_APPLICATION_GITHUB} && \
    cd timeseries/ && pip3 install -r requirements.txt && \
    echo 'figlet -t "Lapig Docker Timeseries Sentinel"' >> ~/.bashrc && \
    chmod +x /APP/timeseries/api.py
    
WORKDIR /APP

CMD [ "/bin/sh", "-c", "cd /APP/timeseries && python3 api.py; tail -f /dev/null"]

ENTRYPOINT [ "/APP/Monitora.sh"]
