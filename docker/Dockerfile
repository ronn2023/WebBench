FROM circleci/node:16.13-browsers

USER root

RUN apt-get -y update
RUN npm i -g lighthouse browsertime
RUN sudo apt install ffmpeg

ADD . /tmp

CMD ["lighthouse", "browsertime"]
