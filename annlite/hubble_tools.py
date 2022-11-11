import os
import platform
import shutil
import time
from pathlib import Path
from typing import Optional, Union

from filesplit.merge import Merge
from filesplit.split import Split
from loguru import logger

ignored_extn = ['.DS_Store']


def get_size(input: Path) -> float:
    import os

    return os.stat(str(input)).st_size / (1024 * 1024)


def make_archive(input: Path, output_name: str) -> Path:
    """
    This function will create a zip archive of the input file (tmp.zip) at the
    same folder of input path.
    """
    output_path = shutil.make_archive(
        os.path.join(str(input.parent), output_name),
        'zip',
        str(input.parent),
        str(input.name),
    )
    return Path(output_path)


class Uploader:
    def __init__(self, size_limit=1024, client=None):
        """
        This class create a filesplit object to split the file into small pieces and
        upload them on to hubble.
        :params size_limit: The max size of split files.
        :params client: hubble client used for uploading.
        """
        self.size_limit = size_limit
        self.client = client

    def upload_file(
        self, input: Path, target_name: str, type: str, cell_id: Union[int, str]
    ):
        logger.info(f'Start to upload single file: {input} to hubble ...')
        size = get_size(input)
        if size > self.size_limit:
            split_list = self._split_file(input)
            self.upload_directory(split_list, target_name, type, cell_id, merge=False)
            shutil.rmtree(split_list)
        else:
            if self._check_exists(target_name, type, input.name):
                return
            self._upload_hubble(input, target_name, type, input.name, cell_id)

    def upload_directory(
        self,
        input: Path,
        target_name: str,
        type: str,
        cell_id: Union[int, str],
        merge: bool = True,
    ):
        def _upload():
            if self._check_exists(target_name, type, str(idx) + '.zip'):
                return

            Path.mkdir(input.parent / str(idx))
            for f in split_list:
                shutil.copy(f, input.parent / str(idx))
            output_path = make_archive(input.parent / str(idx), str(idx) + '.zip')
            self._upload_hubble(
                output_path, target_name, type, str(idx) + '.zip', cell_id
            )
            Path(output_path).unlink()
            shutil.rmtree(input.parent / str(idx))

        logger.info(f'Start to upload directory: {input} to hubble ...')
        if merge:
            size_list = list(
                zip(list(input.iterdir()), [get_size(f) for f in list(input.iterdir())])
            )
            sorted_size_list = sorted(size_list, key=lambda x: x[1])

            split_list = []
            total_size = 0
            idx = 0

            for file_name, file_size in sorted_size_list:
                for extn in ignored_extn:
                    if extn in str(file_name):
                        continue
                if total_size + file_size > self.size_limit:
                    if len(split_list) == 0:
                        raise Exception(
                            f'The smallest file: {file_size} is bigger '
                            f'than size_limit. Please set a larger value '
                            f'of size_limit, now is {self.size_limit}MB.'
                        )
                    _upload()
                    idx += 1
                    total_size = 0
                    split_list = [file_name]
                else:
                    split_list.append(file_name)
                    total_size += file_size
            if len(split_list) > 0:
                _upload()
        else:
            for idx, file_name in enumerate(list(input.glob('*'))):
                if self._check_exists(target_name, type, str(file_name.name)):
                    continue
                self._upload_hubble(
                    file_name, target_name, type, str(file_name.name), cell_id
                )

    def archive_and_upload(
        self,
        target_name: str,
        type: str,
        file_name: str,
        cell_id: Union[int, str],
        root_path: Path,
        upload_folder: str,
    ):
        if self._check_exists(target_name, type, file_name):
            return
        upload_file = shutil.make_archive(
            os.path.join(str(root_path), f'{target_name}_{type}'),
            'zip',
            str(root_path),
            upload_folder,
        )

        logger.info(
            f'Start to upload: {upload_file} to hubble. '
            f'[target_name: {target_name}, '
            f'type: {type}, '
            f'file_name: {file_name}, '
            f'cell_id: {cell_id}].'
        )

        self.client.upload_artifact(
            f=upload_file,
            metadata={
                'name': target_name,
                'type': type,
                'file_name': file_name,
                'cell': cell_id,
            },
        )
        Path(upload_file).unlink()

    def _check_exists(self, target_name: str, type: str, file_name: str) -> bool:
        art_list = self.client.list_artifacts(
            filter={
                'metaData.name': target_name,
                'metaData.type': f'{type}',
                'metaData.file_name': f'{file_name}',
            }
        )
        if len(art_list['data']) != 0:
            logger.info(
                f'[target_name: {target_name}, type: {type}, file_name: {file_name}] '
                f'already exists on hubble, will skip it ...'
            )
            return True
        else:
            return False

    def _split_file(self, input: Path) -> Path:
        output_dir = input / f'{input}_split'
        if output_dir.exists():
            logger.info(
                f'Origin file: {str(input)} has already been split to: {output_dir}, will skip ...'
            )
            return output_dir
        Path.mkdir(output_dir)
        Split(str(input), str(output_dir)).bysize(size=self.size_limit * 1024 * 1024)
        num_files = len(list(output_dir.glob('*')))
        logger.info(
            f'Origin file: {str(input)} has been split '
            f'into {num_files} parts. Output file: {output_dir}'
        )
        return output_dir

    def _upload_hubble(
        self,
        upload_file: Path,
        target_name: str,
        type: str,
        file_name: str,
        cell_id: Union[str, int],
    ):

        logger.info(
            f'Start to upload: {upload_file} to hubble. '
            f'[target_name: {target_name}, '
            f'type: {type}, '
            f'file_name: {file_name}, '
            f'cell_id: {cell_id}].'
        )

        start_time = time.time()
        failed_times = 0

        while True:
            try:
                self.client.upload_artifact(
                    f=str(upload_file),
                    metadata={
                        'name': target_name,
                        'type': type,
                        'file_name': file_name,
                        'cell': cell_id,
                    },
                    show_progress=True,
                )
                break
            except Exception as e:
                logger.info(e)
                failed_times += 1
                if failed_times == 3:
                    logger.info(
                        f'Tried more than 3 times to upload {upload_file}, type is: {type}, will exist...'
                    )
                    return
                else:
                    continue

        logger.info(
            f'Takes {time.time() - start_time} seconds to upload {upload_file}.'
        )


class Merger:
    def __init__(self, restore_path, client):
        """
        This class creates an object to download and merge the split files from hubble.
        :param restore_path: tmp directory for downloading and merging files.
        :param client: hubble client used for merging files.
        """
        self.restore_path = restore_path
        self.restore_path.mkdir(parents=True)
        self.client = client

    def merge_file(self, inputdir: Path, outputdir: Path, outputfilename: Path):
        Merge(
            inputdir=str(inputdir),
            outputdir=str(outputdir),
            outputfilename=str(outputfilename),
        ).merge()

    def get_artifact_ids(self, art_list, type: str, cell_id: Optional[int] = None):
        ids = [
            [
                art['_id'],
                art['metaData']['type'],
                art['metaData']['file_name'],
                art['metaData']['cell'],
            ]
            for art in art_list['data']
            if type == art['metaData']['type']
        ]
        if cell_id:
            ids = [item for item in ids if int(item[3]) == cell_id]
        ids = [[item[0], item[1], item[2]] for item in ids]
        return ids

    def download(self, ids, download_folder):
        Path.mkdir(self.restore_path / download_folder)

        for ids, type, file_name in ids:
            self.client.download_artifact(
                id=ids,
                f=str(self.restore_path / download_folder / file_name),
                show_progress=True,
            )
