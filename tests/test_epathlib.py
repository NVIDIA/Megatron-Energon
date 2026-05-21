# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests the EPath class, our custom version of pathlib.Path"""

import logging
import multiprocessing
import os
import pickle
import struct
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from megatron.energon.epathlib import DEFAULT_PROFILE_NAME, EPath
from megatron.energon.flavors.webdataset.config import (
    INDEX_SQLITE_FILENAME,
    INFO_JSON_FILENAME,
    MAIN_FOLDER_NAME,
)
from tests.epath_s3_emulator import setup_s3_emulator


class TestEPath(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    def tearDown(self):
        pass

    def test_basic(self):
        """Some basic functionality tests"""

        p_rel = EPath("./subdir")
        p_abs = EPath("/tmp")

        p_comb = p_abs / p_rel
        # Those should not raise:
        assert not p_comb.is_file()
        assert not p_abs.is_file()
        assert p_abs.is_dir()

    def test_contextman(self):
        """Test the context manager"""

        tmp_file_path = "/tmp/testfile.bin"
        # First create a file
        with open(tmp_file_path, "wb") as f:
            f.write(struct.pack("H10s", 1337, b"1234567890"))

        # Test context manager reading
        p = EPath(tmp_file_path).open("rb")
        print(p)
        with p:
            b = p.read()
            assert isinstance(b, bytes)

            num, data = struct.unpack("H10s", b)
            logging.info(f"num: {num}")
            assert num == 1337
            assert data == b"1234567890"

        # Test context manager writing
        tmp_file_path2 = "/tmp/testfile2.bin"
        with EPath(tmp_file_path2).open("wb") as p:
            p.write(struct.pack("H10s", 1337, b"1234567890"))

    def test_localfs(self):
        """Test the local filesystem"""
        p = EPath("/tmp/testfile.bin")
        with p.open("wb") as f:
            f.write(b"dummycontent")
        assert p.is_file()
        assert p.size() == 12
        with p.open("rb") as f:
            assert f.read() == b"dummycontent"

        # Test relative paths
        revert_dir = os.getcwd()
        try:
            os.chdir("/tmp")
            p = EPath("testfile.bin")
            assert str(p) == "/tmp/testfile.bin"
            assert p.is_file()
            assert p.size() == 12
            with p.open("rb") as f:
                assert f.read() == b"dummycontent"

            p = EPath("nonexisting/../testfile.bin")
            assert str(p) == "/tmp/testfile.bin"

            p = EPath("../tmp/testfile.bin")
            assert str(p) == "/tmp/testfile.bin"
        finally:
            os.chdir(revert_dir)

        p.unlink()
        assert p.is_file() is False

    def test_glob(self):
        """Test the glob functionality"""

        # First create some files
        for i in range(10):
            with open(f"/tmp/epathtestfile_{i}.bin", "wb") as f:
                f.write(b"dummycontent")

        # Test globbing
        p = EPath("/tmp").glob("epathtestfile_*.bin")

        logging.info(f"p: {p}, type of p: {type(p)}")
        elems = list(p)
        assert len(elems) == 10
        for i, e in enumerate(elems):
            logging.info(f"glob_result[{i}]: {e}")
            assert isinstance(e, EPath)
            assert e.is_file()

        # Test globbing with a pattern
        p = EPath("/tmp").glob("epathtestfile_[0-3].bin")
        assert len(list(p)) == 4

    def test_dss_glob_walk(self):
        import megatron.energon.epathlib.epath as epath_mod

        orig_env_cache_dir = os.environ.get("NVDATASET_CACHE_DIR")
        orig_mod_cache_dir = epath_mod.NVDATASET_CACHE_DIR

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "nvds_cache"
            media_dir = cache_dir / "charts1234" / "v0" / "images"
            media_dir.mkdir(parents=True)
            (media_dir / "000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (media_dir / "001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (media_dir / "002.txt").write_bytes(b"dummy")

            try:
                os.environ["NVDATASET_CACHE_DIR"] = str(cache_dir)
                epath_mod.NVDATASET_CACHE_DIR = EPath(cache_dir)

                root = EPath("dss://charts1234@v0")
                found = sorted(root.glob("**/*.jpg"))

                print(found)

                assert [str(path) for path in found] == [
                    "dss://charts1234@v0/images/000.jpg",
                    "dss://charts1234@v0/images/001.jpg",
                ]
                assert [path.relative_to(root) for path in found] == [
                    "images/000.jpg",
                    "images/001.jpg",
                ]

                found = sorted(root.walk())
                print(found)
                assert [str(p) for p in found] == [
                    "dss://charts1234@v0/images/000.jpg",
                    "dss://charts1234@v0/images/001.jpg",
                    "dss://charts1234@v0/images/002.txt",
                ]
            finally:
                if orig_env_cache_dir is None:
                    os.environ.pop("NVDATASET_CACHE_DIR", None)
                else:
                    os.environ["NVDATASET_CACHE_DIR"] = orig_env_cache_dir
                epath_mod.NVDATASET_CACHE_DIR = orig_mod_cache_dir

    def test_s3_glob_walk(self):
        with setup_s3_emulator(profile_name="s3test_dss_walk") as s3_emulator:
            s3_emulator.put_object("test", "dir/file.txt", b"dummy")
            s3_emulator.put_object("test", "dir/subdir/file2.txt", b"dummy")
            s3_emulator.put_object("test", "dir/subdir/file3.blob", b"dummy")
            root = EPath("msc://s3test_dss_walk/test/dir")
            found = sorted(root.walk())
            print(found)
            assert [str(p) for p in found] == [
                "msc://s3test_dss_walk/test/dir/file.txt",
                "msc://s3test_dss_walk/test/dir/subdir/file2.txt",
                "msc://s3test_dss_walk/test/dir/subdir/file3.blob",
            ]

            found = sorted(root.glob("**/*.txt"))
            print(found)
            assert [str(p) for p in found] == [
                "msc://s3test_dss_walk/test/dir/file.txt",
                "msc://s3test_dss_walk/test/dir/subdir/file2.txt",
            ]

    def test_local_glob_walk(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = EPath(td)
            (td_path / "file.txt").write_text("dummy")
            (td_path / "subdir" / "file2.txt").write_text("dummy")
            (td_path / "subdir" / "file3.blob").write_text("dummy")
            root = EPath(td_path)
            found = sorted(root.walk())
            print(found)
            assert [str(p) for p in found] == [
                str(td_path / "file.txt"),
                str(td_path / "subdir" / "file2.txt"),
                str(td_path / "subdir" / "file3.blob"),
            ]
            found = sorted(root.glob("**/*.txt"))
            print(found)
            assert [str(p) for p in found] == [
                str(td_path / "file.txt"),
                str(td_path / "subdir" / "file2.txt"),
            ]

    def test_s3_path_resolution(self):
        """Test s3 path resolution"""
        rclone_config_path = EPath("/tmp/XDG_CONFIG_HOME/.config/rclone/rclone.conf")
        with rclone_config_path.open("w") as f:
            f.write(
                "\n".join(
                    [
                        "[s3]",
                        "type = s3",
                        "env_auth = false",
                        "access_key_id = dummy",
                        "secret_access_key = dummy",
                        "region = dummy",
                        "endpoint = https://localhost",
                    ]
                )
            )

        orig_xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = "/tmp/XDG_CONFIG_HOME/.config"
        os.environ["HOME"] = "/tmp/XDG_CONFIG_HOME"
        # Hack to clear the cache of the rclone config for msc to get the "s3" profile
        from multistorageclient.rclone import read_rclone_config

        read_rclone_config.cache_clear()
        try:
            # Test globbing
            p = EPath("msc://s3/tmp/path/subpath.txt")
            assert str(p) == "msc://s3/tmp/path/subpath.txt", str(p)

            p2 = p / ".." / "subpath2.txt"
            assert str(p2) == "msc://s3/tmp/path/subpath2.txt", str(p2)

            p3 = EPath("msc://s3/tmp/path/.././subpath.txt")
            assert str(p3) == "msc://s3/tmp/subpath.txt", str(p3)

            p4 = p3.parent / "../bla/bla/bla/../../../no/../subpath2.txt"
            assert str(p4) == "msc://s3/subpath2.txt", str(p4)

            # Test warning for deprecated rclone protocol
            with self.assertWarns((DeprecationWarning, FutureWarning)) as warning:
                # Test rclone backwards compatibility
                pr = EPath("rclone://s3/tmp/path/.././subpath.txt")
                assert str(pr) == "msc://s3/tmp/subpath.txt", str(pr)
            assert "deprecated" in str(warning.warnings[0].message)

            # Test pickle / unpickle
            p4serialized = pickle.dumps(p4)
            # No secret must be serialized
            assert b"dummy" not in p4serialized
        finally:
            if orig_xdg_config_home is not None:
                os.environ["XDG_CONFIG_HOME"] = orig_xdg_config_home
            else:
                del os.environ["XDG_CONFIG_HOME"]
            rclone_config_path.unlink()

    def test_multi_storage_client(self):
        """Test the Multi-Storage Client integration"""
        # Test path handling
        p = EPath(f"msc://{DEFAULT_PROFILE_NAME}/etc/resolv.conf")
        assert str(p) == "/etc/resolv.conf", str(p)
        assert p.is_file()

        p2 = p / ".." / "hosts"
        assert str(p2) == "/etc/hosts", str(p2)

        # Test glob
        p3 = EPath(f"msc://{DEFAULT_PROFILE_NAME}/etc/")
        assert p3.is_dir()
        for i in p3.glob("*.conf"):
            assert str(i).endswith(".conf")

        # Test open file
        assert p.size() > 0
        with p.open("r") as fp:
            assert len(fp.read()) > 0

        # Test move and delete
        p4 = EPath(f"msc://{DEFAULT_PROFILE_NAME}/tmp/random_file_0001")
        if p4.is_file():
            p4.unlink()
        with p4.open("w") as fp:
            fp.write("*****")
        assert p4.is_file()
        p5 = EPath(f"msc://{DEFAULT_PROFILE_NAME}/tmp/random_file_0002")
        if p5.is_file():
            p5.unlink()
        assert p5.is_file() is False
        p4.move(p5)
        assert p5.is_file()
        assert p4.is_file() is False
        p5.unlink()
        assert p5.is_file() is False

        # Test pickle / unpickle
        p5serialized = pickle.dumps(p5)
        p5unserialized = pickle.loads(p5serialized)
        assert p5unserialized == p5
        assert str(p5unserialized) == str(p5)

    def test_multiprocessing(self):
        """Test EPath in multiprocessing context"""
        p = EPath("/tmp/path/subpath.txt")

        orig_start_method = multiprocessing.get_start_method()
        try:
            multiprocessing.set_start_method("spawn", force=True)

            proc = multiprocessing.Process(target=_multiproc_test_func, args=(p, True))
            proc.start()
            proc.join()
            assert proc.exitcode == 0

            multiprocessing.set_start_method("fork", force=True)

            proc = multiprocessing.Process(target=_multiproc_test_func, args=(p, True))
            proc.start()
            proc.join()
            assert proc.exitcode == 0
        finally:
            multiprocessing.set_start_method(orig_start_method, force=True)

    def test_multiprocessing_msc(self):
        """Test EPath in multiprocessing context"""
        p = EPath(f"msc://{DEFAULT_PROFILE_NAME}/tmp/random_file_0001")
        with p.open("w") as fp:
            fp.write("*****")

        orig_start_method = multiprocessing.get_start_method()
        try:
            multiprocessing.set_start_method("spawn", force=True)

            proc = multiprocessing.Process(target=_multiproc_test_func, args=(p, True))
            proc.start()
            proc.join()
            assert proc.exitcode == 0

            multiprocessing.set_start_method("fork", force=True)

            proc = multiprocessing.Process(target=_multiproc_test_func, args=(p, True))
            proc.start()
            proc.join()
            assert proc.exitcode == 0
        finally:
            multiprocessing.set_start_method(orig_start_method, force=True)
            p.unlink()

    def test_msc_s3(self):
        # Test S3 with MSC
        with setup_s3_emulator(profile_name="s3test_msc"):
            p = EPath("msc://s3test_msc/test/dir/file.txt")
            assert not p.is_file()
            p.write_text("dummy")
            assert p.is_file()
            assert p.size() > 0
            assert p.read_text() == "dummy"
            assert EPath("msc://s3test_msc/test").is_dir()
            assert EPath("msc://s3test_msc/test/dir").is_dir()
            p.unlink()
            assert not p.is_file()
            assert not EPath("msc://s3test_msc/test").is_dir()
            assert not EPath("msc://s3test_msc/test/dir").is_dir()

    def test_msc_s3_dataprep_path_operations(self):
        """EPath operations used by remote webdataset ``prepare`` (glob, rb, meta, sqlite upload, move).

        Mirrors ``tools/prepare.py`` shard discovery, ``WebdatasetPreparator._preprocess_tar``
        binary reads, and ``SqliteIndexWriterAggregator`` uploading ``index.sqlite`` from disk.
        """
        profile = "s3test_msc_dataprep"
        with setup_s3_emulator(profile_name=profile):
            root = EPath(f"msc://{profile}/dataset_root")
            parts = root / "parts"
            (parts / "data-0.tar").write_bytes(b"shard0-bytes")
            (parts / "data-1.tar").write_bytes(b"shard1-bytes")

            found = sorted(root.glob("**/*.tar"))
            assert [p.name for p in found] == ["data-0.tar", "data-1.tar"]

            with (parts / "data-0.tar").open("rb") as f:
                assert f.read(6) == b"shard0"

            meta_dir = root / MAIN_FOLDER_NAME
            info = meta_dir / INFO_JSON_FILENAME
            info.write_text('{"shard_counts": {"parts/data-0.tar": 1}}')
            assert '"shard_counts"' in info.read_text()

            probe_idx = parts / "probe.idx"
            probe_idx_tmp = parts / "probe.idx.tmp"
            with probe_idx_tmp.open("wb") as out:
                out.write(struct.pack("QQ", 0, 512))
            assert probe_idx_tmp.size() == 16
            assert struct.unpack("QQ", probe_idx_tmp.read_bytes()) == (0, 512)

            probe_idx_tmp.move(probe_idx)

            assert probe_idx.size() == 16
            assert struct.unpack("QQ", probe_idx.read_bytes()) == (0, 512)
            probe_idx.unlink()

            with tempfile.NamedTemporaryFile(delete=False) as lf:
                lf.write(b"sqlite-placeholder")
                local_sqlite = lf.name
            try:
                EPath(local_sqlite).copy(meta_dir / INDEX_SQLITE_FILENAME)
                remote_db = meta_dir / INDEX_SQLITE_FILENAME
                assert remote_db.read_bytes() == b"sqlite-placeholder"
                remote_db.unlink()
            finally:
                Path(local_sqlite).unlink(missing_ok=True)

            info.unlink()
            for p in found:
                p.unlink()

    def test_msc_s3_stat_uses_stored_last_modified(self):
        profile = "s3test_msc_timestamps"
        with setup_s3_emulator(profile_name=profile) as state:
            remote_path = EPath(f"msc://{profile}/bucket/data.bin")
            original_mtime = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
            updated_mtime = datetime(2024, 1, 3, 3, 4, 5, tzinfo=timezone.utc)

            state.put_object("bucket", "data.bin", b"first", last_modified=original_mtime)

            first_stat = remote_path.stat()
            second_stat = remote_path.stat()
            assert int(first_stat.last_modified.timestamp()) == int(original_mtime.timestamp())
            assert second_stat.last_modified == first_stat.last_modified

            state.put_object("bucket", "data.bin", b"second", last_modified=updated_mtime)

            updated_stat = remote_path.stat()
            assert int(updated_stat.last_modified.timestamp()) == int(updated_mtime.timestamp())
            assert updated_stat.content_length == len(b"second")

    def test_dss_path_requires_version(self):
        with self.assertRaisesRegex(
            AssertionError,
            "DSS paths must include a dataset version separated by '@'",
        ):
            EPath("dss://charts1234")

    def test_metadataset_v2_dss_path_parsing_str(self):
        """Parse a MetadatasetV2 config and ensure DSS URLs stringify correctly as EPath."""

        # Ensure DSS paths are supported even when tests are run without NVDATASET_CACHE_DIR set
        # at process start (EPath reads it at import time).
        import megatron.energon.epathlib.epath as epath_mod
        from megatron.energon.metadataset.loader import load_dataset

        yaml_text = "\n".join(
            [
                "__module__: megatron.energon",
                "__class__: MetadatasetV2",
                "splits:",
                "  train:",
                "    blend:",
                "      - path: dss://charts1234_zh@v0",
                "        weight: 1",
                "        aux:",
                "          media_source: filesystem+dss://charts1234@v0",
            ]
        )

        orig_env_cache_dir = os.environ.get("NVDATASET_CACHE_DIR")
        orig_mod_cache_dir = epath_mod.NVDATASET_CACHE_DIR

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cache_dir = td_path / "nvds_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy DSS datasets in the cache dir so that `load_dataset()` can run
            # post-initialization without hitting missing-path errors.
            #
            # - charts1234_zh/v0: minimal "webdataset" marker (presence of .nv-meta/.info.json)
            # - charts1234/v0: folder with images (aux media source)
            webdataset_root = cache_dir / "charts1234_zh" / "v0"
            (webdataset_root / MAIN_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
            (webdataset_root / MAIN_FOLDER_NAME / INFO_JSON_FILENAME).write_text(
                "{}", encoding="utf-8"
            )

            media_root = cache_dir / "charts1234" / "v0"
            (media_root / "images").mkdir(parents=True, exist_ok=True)
            (media_root / "images" / "000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (media_root / "images" / "001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (media_root / MAIN_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
            (media_root / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME).write_text(
                "{}", encoding="utf-8"
            )

            mds_yaml_path = td_path / "metadataset_v2_dss.yaml"
            mds_yaml_path.write_text(yaml_text, encoding="utf-8")

            try:
                os.environ["NVDATASET_CACHE_DIR"] = str(cache_dir)
                epath_mod.NVDATASET_CACHE_DIR = EPath(cache_dir)

                mds_path = EPath(mds_yaml_path)
                mds = load_dataset(mds_path)

                train = mds.splits["train"]
                from megatron.energon.metadataset.metadataset_v2 import AuxFilesystemReference

                assert isinstance(train.blend[0].path, EPath)
                ds0 = train.blend[0].path

                assert train.blend[0].aux is not None
                aux_ref0 = train.blend[0].aux["media_source"]
                assert isinstance(aux_ref0, AuxFilesystemReference)
                assert isinstance(aux_ref0.fs_path, EPath)
                aux0 = aux_ref0.fs_path

                for p in (ds0, aux0):
                    print(f"Dataset: {str(p)}, url: {p.url}")

                assert ds0.url == "dss://charts1234_zh@v0"
                assert aux0.url == "dss://charts1234@v0"
                assert ds0.local_path() == cache_dir / "charts1234_zh" / "v0"
                assert aux0.local_path() == cache_dir / "charts1234" / "v0"
            finally:
                if orig_env_cache_dir is None:
                    os.environ.pop("NVDATASET_CACHE_DIR", None)
                else:
                    os.environ["NVDATASET_CACHE_DIR"] = orig_env_cache_dir
                epath_mod.NVDATASET_CACHE_DIR = orig_mod_cache_dir


def _multiproc_test_func(p: EPath, test_function: bool):
    """Helper function for multiprocessing test"""
    print(f"str: {str(p)}")
    if test_function:
        print(f"is_file: {p.is_file()}")


if __name__ == "__main__":
    unittest.main()
