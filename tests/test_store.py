import pytest

from annlite.storage.kv import DocStorage


def test_get(tmpfile, docs):
    storage = DocStorage(tmpfile)

    storage.insert(docs)

    doc = storage.get('doc1')[0]
    assert doc.id == 'doc1'
    assert (doc.embedding == [1, 0, 0, 0]).all()

    docs = storage.get('doc7')
    assert len(docs) == 0


def test_update(tmpfile, docs, update_docs):
    storage = DocStorage(tmpfile)
    storage.insert(docs)

    storage.update(update_docs)

    doc = storage.get('doc1')[0]
    assert (doc.embedding == [0, 0, 0, 1]).all()


def test_delete(tmpfile, docs):
    storage = DocStorage(tmpfile)
    storage.insert(docs)
    storage.delete(['doc1'])
    docs = storage.get('doc1')
    assert len(docs) == 0


def test_clear(tmpfile, docs):
    storage = DocStorage(tmpfile)
    storage.insert(docs)

    assert storage.size == 6
    storage.clear()
    assert storage.size == 0


def test_batched_iterator(tmpfile, docs):
    storage = DocStorage(tmpfile)
    storage.insert(docs)
    for docs in storage.batched_iterator(batch_size=3):
        assert len(docs) == 3


@pytest.mark.parametrize('protocol', ['pickle', 'protobuf'])
def test_searalize(tmpfile, protocol, docs):
    storage = DocStorage(tmpfile, serialize_config={'protocol': protocol})
    storage.insert(docs)

    doc = storage.get('doc1')[0]
    assert doc.id == 'doc1'
    assert (doc.embedding == [1, 0, 0, 0]).all()

    docs = storage.get('doc7')
    assert len(docs) == 0
