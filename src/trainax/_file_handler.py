from typing import TextIO

from trainax._types import PathLike


class FileHandler:
    """Manage a collection of writable files for long-running callbacks.

    Acts as a tiny context manager that opens all declared files on entry and
    closes them on exit.

    Attributes
    ----------
    files : dict[str, PathLike]
        Current key-to-path mapping.

    Methods
    -------
    open()
        Open all registered files in write mode.
    close()
        Close all open file handles.
    """

    _files: dict[str, PathLike]
    _open_files: dict[str, TextIO]

    def __init__(self, files: dict[str, PathLike]):
        """Create a handler around a mapping of keys to filesystem paths.

        Parameters
        ----------
        files : dict[str, PathLike]
            Mapping of string keys to filesystem paths for output files.

        Returns
        -------
        None
        """
        self._files = files
        self._open_files = {}

    @property
    def files(self):
        """Current key-to-path mapping.

        Returns
        -------
        dict[str, PathLike]
            Mapping of string keys to registered filesystem paths.
        """
        return self._files

    def open(self):
        """Open all registered files in write mode.

        Returns
        -------
        None
        """
        self._open_files = {
            fkey: open(file, "w")
            for fkey, file in self._files.items()  # type: ignore
        }

    def close(self):
        """Close any open handles and reset internal state.

        Returns
        -------
        None
        """
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

    def __getitem__(self, key):
        """Return an open handle for ``key`` or raise informative errors.

        Parameters
        ----------
        key : str
            Lookup key for the desired file handle.

        Returns
        -------
        TextIO
            The open writable file handle associated with ``key``.

        Raises
        ------
        KeyError
            If the key is registered but the file has not been opened yet, or
            if the key is not registered at all (with a hint to pass it via
            ``continuous_files`` in the trainer constructor).
        """
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

    def __repr__(self):
        """Return a compact summary of the file handler state."""
        status = "open" if self._open_files else "closed"
        return (
            f"FileHandler("
            f"filekeys={','.join(list(self._files.keys()))}; "
            f"files {status})"
        )
