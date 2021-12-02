FROM jinaai/jina:2-py37-perf

RUN apt-get update && apt-get install --no-install-recommends -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
