# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests the EPath class, our custom version of pathlib.Path"""

import logging
import os
import struct
import sys
import unittest
from typing import Any, Tuple

from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.rclone_config import ConfigEntry

test_config = {
    "energon": ConfigEntry(
        name="energon",
        type="s3",
        provider="Other",
        access_key_id="energon",
        secret_access_key="f3a3dc2554a9d5b3ea8c51e180e81302",
        region="us-east-1",
        endpoint="pdx.s8k.io",
    )
}


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
        # logging.info(f"p_comb: {p_comb}")
        # logging.info(f"p_comb: {p_comb.internal_path}")

        # We don't want to work on relative paths
        self.assertRaises(AssertionError, lambda: p_rel.is_file())

        # Those should not raise:
        assert p_comb.is_absolute()
        _ = p_comb.is_file()
        _ = p_abs.is_file()

    def test_contextman(self):
        """Test the context manager"""

        tmp_file_path = "/tmp/testfile.bin"
        # First create a file
        with open(tmp_file_path, "wb") as f:
            f.write(struct.pack("H10s", 1337, b"1234567890"))

        # Test context manager reading
        p = EPath(tmp_file_path).open("rb")
        with p:
            b = p.read()
            assert isinstance(b, bytes)

            num, data = struct.unpack("H10s", b)
            logging.info(f"num: {num}")
            assert num == 1337
            assert data == b"1234567890"

            assert not p.closed

        assert p.closed

        # Test context manager writing
        tmp_file_path2 = "/tmp/testfile2.bin"
        with EPath(tmp_file_path2).open("wb") as p:
            p.write(struct.pack("H10s", 1337, b"1234567890"))

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

    def test_rclone_path(self):
        """Test the rclone path"""

        p = EPath(
            "rclone://energon/test/minicomp/shard_000000.tar",
            config_override=test_config,
        )
        logging.info(f"size: {p.size()}")
        logging.info(f"is file: {p.is_file()}")

        # Read first 12 bytes:
        with p.open("rb") as f:
            b = f.read(12)
            assert isinstance(b, bytes)
            logging.info(f"bytes: {b}, hex: {b.hex()}")

    def test_multiprocessing(self):
        """Test the multiprocessing functionality"""
        import multiprocessing

        def in_own_proc():
            EPath.prepare_forked_process()
            epath = EPath(
                "rclone://energon/test/minicomp/shard_000000.tar",
                config_override=test_config,
            )
            with epath.open("rb", block_size=1024 * 10) as f:
                cont = f.read(10)
                assert isinstance(cont, bytes)
                print(f"Forked process: Read 10 bytes from {p}: {cont.hex()}")

        somepath = EPath(
            "rclone://energon/test/minicomp/shard_000000.tar", config_override=test_config
        )
        with somepath.open("rb", block_size=1024 * 10) as f:
            cont = f.read(10)
            assert isinstance(cont, bytes)
            print(f"Parent proc: Read 10 bytes from {somepath}: {cont.hex()}")

        p = multiprocessing.Process(target=in_own_proc)
        print("Starting process.")
        p.start()
        p.join()
        print("Process done.")

    def test_multiprocessing_with_forked_epath(self):
        """Test if copying the EPath object works in a forked process."""

        import multiprocessing

        ctx = multiprocessing.get_context(
            method="fork"
        )  # Should be the default in Python<3.14, but let's be sure.

        def in_own_proc(passed_epath_instance: EPath):
            EPath.prepare_forked_process()
            print(f"In forked process with PID {os.getpid()}.")
            print(f"Passed EPath instance was created with PID {passed_epath_instance.fs._pid}.")

            assert passed_epath_instance.fs._pid != os.getpid()

            with passed_epath_instance.open("rb", block_size=1024 * 10) as f:
                cont = f.read(10)
                assert isinstance(cont, bytes)
                print(f"Forked process: Read 10 bytes from {p}: {cont.hex()}")

        somepath = EPath(
            "rclone://energon/test/minicomp/shard_000000.tar", config_override=test_config
        )
        with somepath.open("rb", block_size=1024 * 10) as f:
            cont = f.read(10)
            assert isinstance(cont, bytes)
            print(f"Parent proc: Read 10 bytes from {somepath}: {cont.hex()}")

        p = ctx.Process(target=in_own_proc, args=(somepath,))
        print("Starting process.")
        p.start()
        p.join()
        print("Process done.")


if __name__ == "__main__":
    unittest.main()
