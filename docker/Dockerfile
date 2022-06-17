FROM merlin:latest as builder
MAINTAINER Cody.Hao.Yu cody@falcon-computing.com
LABEL Version "0.1"

RUN python3.6 -m pip install --trusted-host pypi.python.org importlib-metadata --upgrade

RUN python3.6 -m pip install --trusted-host pypi.python.org setuptools \
    future mypy pylint pytest \
    pytest-cov pytest-mock Sphinx sphinx_rtd_theme yapf tox argparse jsonpickle pickledb \
    redis texttable matplotlib --upgrade

RUN yum install -y redis faketime

COPY ./docker/dse /usr/local/bin
RUN chmod 755 /usr/local/bin/dse
COPY ./docker/autodse /usr/local/bin
RUN chmod 755 /usr/local/bin/autodse
COPY ./docker/checkds /usr/local/bin
RUN chmod 755 /usr/local/bin/checkds
COPY ./docker/single /usr/local/bin
RUN chmod 755 /usr/local/bin/single
COPY ./docker/ds_generator /usr/local/bin
RUN chmod 711 /usr/local/bin/ds_generator

COPY ./docker/entrypoint.sh /usr/local/bin
RUN chmod 755 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

ADD . /opt/merlin_dse
ENV PYTHONPATH /opt/merlin_dse:${PYTHONPATH}

RUN find /opt/merlin_dse/autodse -type d -exec chmod 755 {} \;
RUN find /opt/merlin_dse/autodse -type f -exec chmod 644 {} \;
