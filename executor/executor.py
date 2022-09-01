from annlite.executor import AnnLiteIndexer as _AnnLiteIndexer


class AnnLiteIndex(_AnnLiteIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
