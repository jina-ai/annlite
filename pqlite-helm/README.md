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

## Step 3: Create and Apply Persistent Storage Volume

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

Then apply the configuration with `kubectl`:

```bash
$ kubectl apply -f pqlite-storage.yaml
```

## Step 4: Install Helm Chart

Install the helm chart with the `helm install` command. Add `--set` flags to the command to connect the installation to the PVC you created and enable volume permissions:

```bash
$ helm install pqlite-test \
    --set shardCount=3 \
    --set persistence.existingClaim=pqlite-pvc-claim \
    --set persistence.enabled=true \
    ./pqlite-helm
```


## Conferences

- [https://phoenixnap.com/kb/postgresql-kubernetes](https://phoenixnap.com/kb/postgresql-kubernetes)
