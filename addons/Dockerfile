FROM jinaai/jina:master-py38-perf

RUN echo 'debug 3'
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
