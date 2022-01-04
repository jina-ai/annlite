from jina import Flow

f = Flow(port_expose=8080, protocol='http').add(
    name='indexer', uses='jinahub+docker://PQLiteIndexer/latest'
)
f.to_k8s_yaml('./deployment', k8s_namespace='pqlite-namespace')
