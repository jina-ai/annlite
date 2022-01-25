from jina import Flow

f = Flow(port_expose=8080, protocol='http').add(
    name='indexer',
    uses='jinahub+docker://PQLiteIndexer/latest',
    uses_with={'dim': 512},
    shards=3,
)
f.to_k8s_yaml('./deployment')
