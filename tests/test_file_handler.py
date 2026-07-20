import pytest

from trainax._file_handler import FileHandler


def test_file_handler_context_opens_and_closes(tmp_path):
    files = {"train": tmp_path / "train.txt"}
    handler = FileHandler(files)
    with handler as fh:
        fh["train"].write("hello\n")
        assert fh["train"].closed is False
    with pytest.raises(KeyError, match="File for key 'train' not open"):
        handler["train"]
    assert (tmp_path / "train.txt").read_text() == "hello\n"


def test_file_handler_missing_key_errors(tmp_path):
    handler = FileHandler({"train": tmp_path / "train.txt"})
    with pytest.raises(KeyError, match="File for key 'train' not open"):
        handler["train"]
    with handler:
        with pytest.raises(KeyError, match="File for key 'missing' not in"):
            handler["missing"]


def test_file_handler_repr(tmp_path):
    handler = FileHandler({"train": tmp_path / "train.txt"})
    assert "files closed" in repr(handler)
    with handler:
        assert "files open" in repr(handler)
