# Deploy QPLite Using Helm

## Prerequisite
- Kubernetes Cluster Setup
- Clone [this](https://github.com/jina-ai/pqlite.git) git repo

## Step 1: Clone Helm Repository

```bash
$ git clone https://github.com/jina-ai/pqlite.git \
  && cd pqlite
```

## Step 2: Adding a Bitnami Chart dependency

Chart dependencies are used to install other charts’ resources that a Helm chart may depend on.

In this chart, we are using bitnami/common as a plugin so we to need add this as a dependency.

When downloading a dependency for the first time, you should use the helm dependency update command.

```bash
$ helm dependency update ./pqlite-helm
```

## Step 3: (Optional) Create and Apply Persistent Storage Volume

The data in your Pqlite indexer need to persist across pod restarts.

To achieve this, create a PersistentVolume resource in a YAML file. This example `pqlite-storage.yaml` uses the following configuration:
```YAML
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pqlite-pv-volume
  labels:
    type: local
    app: pqlite
spec:
  storageClassName: manual
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/tmp/pqlite"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pqlite-pvc-claim
  labels:
    app: pqlite
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
```

Then apply the configuration with `kubectl apply`:

```bash
$ kubectl apply -f pqlite-storage.yaml
```

## Step 4: Install Helm Chart

Install the helm chart with the `helm install` command. Add `--set` flags to the command to connect the installation to the PVC you created and enable volume permissions:

```bash
$ helm install pqlite-test \
    --set dimension=256 \
    --set shardCount=3 \
    --set persistence.existingClaim=pqlite-pvc-claim \
    --set persistence.enabled=true \
    ./pqlite-helm
```

**Note:**

- `dimension`: required parameter to specify the dimension of the embeddings.
- `metric`: optional parameter to specify the distance metric to use for searching.
- `shardCount`: optional parameter to specify the number of shards.
- `persistence.*`: optional parameter, need to specify when PVC is needed.

## Step 5: Use in a Jina Flow

Expose port `4567` to access the service via `kubectl port-forward ...`

```bash
$ kubectl port-forward service/indexer-head 4567:8081
```

This example shows how to use `pqlite` as **external pod** in a Jina Flow

```python
import numpy as np
from jina import DocumentArray, Flow

PROTOCOL = 'grpc'
EXTERNAL_PORT_IN = 4567

f = Flow(protocol=PROTOCOL).add(
    name='indexer',
    external=True,
    host='localhost',
    port_in=EXTERNAL_PORT_IN,
)

with f:

    docs = DocumentArray.empty(50)
    docs.embeddings = np.random.random((50, 512)).astype(np.float32)

    result = f.post('/index', inputs=docs, request_size=1, return_results=True)
    result = f.post('/search', inputs=docs[:5], return_results=True)
```

## Conferences

- [https://phoenixnap.com/kb/postgresql-kubernetes](https://phoenixnap.com/kb/postgresql-kubernetes)
