FROM jinaai/jina:master-py38-perf

COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
