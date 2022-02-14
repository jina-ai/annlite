from jina import Flow

# POLLING_PER_ENDPOINT:
# https://github.com/jina-ai/jina/blob/master/tests/unit/orchestrate/flow/flow-orchestrate/test_flow_routing.py#L126
f = Flow(port_expose=8080, protocol='http').add(
    name='indexer',
    uses='jinahub+docker://PQLiteIndexer/v0.2.3-rc',
    uses_with={'dim': 512},
    shards=3,
    polling={'/index': 'ANY', '/search': 'ALL', '*': 'ANY'},
)
f.to_k8s_yaml('./deployment', k8s_namespace='showtell')
