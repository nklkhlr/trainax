from typing import TextIO

from trainax._types import PathLike


class FileHandler:
    _files: dict[str, PathLike]
    _open_files: dict[str, TextIO]

    def __init__(self, files: dict[str, PathLike]):
        self._files = files
        self._open_files = {}

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, files):
        if self._open_files:
            raise RuntimeError("Cannot change file list from inside context")
        self._files = files

    def add_file(self, key: str, file: PathLike):
        if self._open_files:
            raise RuntimeError("Cannot change file list from inside context")
        self._files[key] = file

    def get_file_path(self, key: str):
        return self._files[key]

    def __enter__(self):
        self._open_files = {
            fkey: open(file, "w") for fkey, file in self._files.items()
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self._open_files.values():
            file.close()
        self._open_files = {}

    def __getitem__(self, key):
        try:
            return self._open_files[key]
        except KeyError as ke:
            if key in self._files:
                raise KeyError(f"File for key '{key}' not open") from ke
            raise KeyError(f"File for key '{key}' not in file handler") from ke

    def __setitem__(self, key, value):
        self.add_file(key, value)

    def __repr__(self):
        status = "closed"
        if self._open_files:
            status = "open"
        return (
            f"FileHandler("
            f"filekeys={','.join(list(self._files.keys()))}; "
            f"files {status})"
        )
