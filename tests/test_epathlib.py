# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests the EPath class, our custom version of pathlib.Path"""

import logging
import multiprocessing
import os
import pickle
import struct
import sys
import unittest

from epath_s3_emulator import setup_s3_emulator

from megatron.energon.epathlib import EPath


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
        p = EPath("msc://default/etc/resolv.conf")
        assert str(p) == "/etc/resolv.conf", str(p)
        assert p.is_file()

        p2 = p / ".." / "hosts"
        assert str(p2) == "/etc/hosts", str(p2)

        # Test glob
        p3 = EPath("msc://default/etc/")
        assert p3.is_dir()
        for i in p3.glob("*.conf"):
            assert str(i).endswith(".conf")

        # Test open file
        assert p.size() > 0
        with p.open("r") as fp:
            assert len(fp.read()) > 0

        # Test move and delete
        p4 = EPath("msc://default/tmp/random_file_0001")
        p4.unlink()
        with p4.open("w") as fp:
            fp.write("*****")
        assert p4.is_file()
        p5 = EPath("msc://default/tmp/random_file_0002")
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
        p = EPath("msc://default/tmp/random_file_0001")
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
        with setup_s3_emulator():
            p = EPath("msc://s3/test/file.txt")
            assert not p.is_file()
            p.write_text("dummy")
            assert p.is_file()
            assert p.size() > 0
            assert p.read_text() == "dummy"
            p.unlink()
            assert not p.is_file()


def _multiproc_test_func(p: EPath, test_function: bool):
    """Helper function for multiprocessing test"""
    print(f"str: {str(p)}")
    if test_function:
        print(f"is_file: {p.is_file()}")


if __name__ == "__main__":
    unittest.main()
