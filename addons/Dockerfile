FROM jinaai/jina:master-py38-perf

RUN echo 'dev 4'
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
