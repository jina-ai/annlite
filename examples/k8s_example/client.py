import numpy as np
from jina import Client, DocumentArray, Flow

#
# f = Flow(port_expose=4567, protocol='http').add(
#     name='indexer',
#     uses='docker://numb3r3/pqlite-executor:latest',
#     uses_with={'dim': 512},
#     shards=3,
#     polling={'/index': 'ANY', '/search': 'ALL', '*': 'ANY'},
# )
#
# with f:
#     client = Client(
#         host='localhost',
#         port='4567',
#         protocol='http',
#     )
#
#     docs = DocumentArray.empty(5)
#     docs.embeddings = np.random.random((5, 512)).astype(np.float32)
#
#     result = client.post('/index', inputs=docs, return_results=True)[0]
#     print(result.docs.summary())
#     # status = client.post('/status', return_results=True)[0]
#     # print(status.docs[0].to_dict())
#     # f.block()


client = Client(
    host='localhost',
    port='4567',
    protocol='grpc',
)

N = 5
docs = DocumentArray.empty(N)
docs.embeddings = np.random.random((N, 512)).astype(np.float32)

result = client.post('/index', inputs=docs, return_results=True)[0]
print(result.docs.summary())
#
#
# status = client.post('/status', return_results=True)
# print(status.docs[0].to_dict())
