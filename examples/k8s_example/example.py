import numpy as np
from jina import Client, DocumentArray, Flow

PROTOCOL = 'grpc'
EXTERNAL_HEAD_PORT_IN = 4567

f = Flow(protocol=PROTOCOL).add(
    name='indexer',
    external=True,
    host='localhost',
    port_in=EXTERNAL_HEAD_PORT_IN,
)

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

    # print(result.docs.summary())
    for r in result.docs:
        print(f'#matches: {len(r.matches)}')
        for k, match in enumerate(r.matches):
            print(f'[{k}] matched doc: {match.id} -> ({match.scores["cosine"].value})')
        break

    # status = client.post('/status', return_results=True)[0]
