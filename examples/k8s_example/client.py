import numpy as np
from jina import Client, DocumentArray, Flow

# from match_merger import MatchMerger

PROTOCOL = 'grpc'

f = Flow(protocol=PROTOCOL).add(
    name='indexer',
    uses='docker://numb3r3/pqlite-executor:latest',
    uses_with={'dim': 512},
    shards=3,
    uses_after='docker://numb3r3/pqlite-merger:latest',
    polling={'/index': 'ANY', '/search': 'ALL', '*': 'ANY'},
)  # .add(name='match_merger', uses=MatchMerger)

with f:
    client = Client(
        host='localhost',
        port=f.port_expose,
        protocol=PROTOCOL,
    )

    docs = DocumentArray.empty(50)
    docs.embeddings = np.random.random((50, 512)).astype(np.float32)

    result = f.post('/index', inputs=docs, request_size=1, return_results=True)
    print(f'=== FLOW Index ===')
    print(f'#index resp docs: {len(result)}\n')

    print(f'=== FLOW Search ===')
    result = f.post('/search', inputs=docs[:1], return_results=True)
    print(f'#query resp docs: {len(result)}')
    for r in result:
        print(f'#matches: {len(r.matches)}')
        for k, match in enumerate(r.matches):
            print(f'[{k}] matched doc: {match.id} -> ({match.scores["cosine"].value})')
        break

    print(f'=== Raw Client ===')
    result = client.post('/search', inputs=docs[:5], return_results=True)[0]
    # print(len(result))

    print(result.docs.summary())
    for r in result.docs:
        print(f'#matches: {len(r.matches)}')
        for k, match in enumerate(r.matches):
            print(f'[{k}] matched doc: {match.id} -> ({match.scores["cosine"].value})')
        break

    # status = client.post('/status', return_results=True)[0]
    # print(status.docs[0].to_dict())
    # f.block()


# client = Client(
#     host='localhost',
#     port='4567',
#     protocol='grpc',
# )
#
# N = 5
# docs = DocumentArray.empty(N)
# docs.embeddings = np.random.random((N, 512)).astype(np.float32)
#
# result = client.post('/index', inputs=docs, return_results=True)[0]
# print(result.docs.summary())
#
# result = client.post('/search', inputs=docs, return_results=True)[0]
# print(result.docs.summary())
#
#
# status = client.post('/status', return_results=True)
# print(status.docs[0].to_dict())
