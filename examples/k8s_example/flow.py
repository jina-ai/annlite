from jina import Flow

# POLLING_PER_ENDPOINT:
# https://github.com/jina-ai/jina/blob/master/tests/unit/orchestrate/flow/flow-orchestrate/test_flow_routing.py#L126
f = Flow(port_expose=8080, protocol='grpc').add(
    name='indexer',
    uses='docker://numb3r3/pqlite-executor:latest',
    uses_with={'dim': 512},
    shards=3,
    polling={'/index': 'ANY', '/search': 'ALL', '*': 'ANY'},
    uses_after='jinahub+docker://MatchMerger/v0.3',
)
f.to_k8s_yaml('./deployment')
