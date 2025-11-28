from typing import TextIO

from trainax._types import PathLike


class FileHandler:
    """Manage a collection of writable files for long-running callbacks.

    Acts as a tiny context manager that opens all declared files on entry and
    closes them on exit. During an active context the mapping is immutable, and
    attempting to mutate it raises `RuntimeError` to help catch accidental
    resource leaks.
    """

    _files: dict[str, PathLike]
    _open_files: dict[str, TextIO]

    def __init__(self, files: dict[str, PathLike]):
        """Create a handler around a mapping of keys to filesystem paths."""
        self._files = files
        self._open_files = {}

    @property
    def files(self):
        """dict[str, PathLike]: Current key-to-path mapping."""
        return self._files

    def set_files(self, files):
        """Replace the file mapping, ensuring no handles are currently open."""
        if self._open_files:
            raise RuntimeError("Cannot change file list from inside context")
        self._files = files

    def add_file(self, key: str, file: PathLike):
        """Register a new output file under `key`."""
        if self._open_files:
            raise RuntimeError("Cannot change file list from inside context")
        self._files[key] = file

    def get_file_path(self, key: str):
        """Return the registered path for `key` without opening it."""
        return self._files[key]

    def open(self):
        """Open all registered files in write mode."""
        self._open_files = {
            fkey: open(file, "w")
            for fkey, file in self._files.items()  # type: ignore
        }

    def close(self):
        """Close any open handles and reset internal state."""
        # TODO: provide safer semantics on crash or deletion.
        for file in self._open_files.values():
            file.close()
        self._open_files = {}

    def __enter__(self):
        """Enter a context by opening all registered files."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close all managed files on context exit."""
        self.close()

    def __del__(self):
        """Best-effort cleanup when the handler is garbage collected."""
        self.close()

    def __getitem__(self, key):
        """Return an open handle for `key` or raise informative errors."""
        try:
            return self._open_files[key]
        except KeyError as ke:
            if key in self._files:
                raise KeyError(f"File for key '{key}' not open") from ke
            raise KeyError(
                f"File for key '{key}' not in file handler. "
                "Please make sure you pass all files that are used by "
                "callbacks in the trainer class via the 'continuous_files' "
                "argument in the trainer constructor."
            ) from ke

    def __setitem__(self, key, value):
        """Alias to :meth:`add_file` allowing dict-like updates."""
        self.add_file(key, value)

    def __repr__(self):
        """Return a compact summary of the file handler state."""
        status = "open" if self._open_files else "closed"
        return (
            f"FileHandler("
            f"filekeys={','.join(list(self._files.keys()))}; "
            f"files {status})"
        )
