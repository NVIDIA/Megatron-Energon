# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generator,
    Literal,
    TypedDict,
    Union,
)

import click
import numpy as np
from PIL import Image

from megatron.energon.tools.analyze_debug_merge import merge_log_reader

cpal = np.array(
    [
        [int(x) for x in line.split(" ")]
        for line in """255 255 255
1 0 103
213 255 0
255 0 86
158 0 142
14 76 161
255 229 2
0 95 57
0 255 0
149 0 58
255 147 126
164 36 0
0 21 68
145 208 203
98 14 0
107 104 130
0 0 255
0 125 181
106 130 108
0 174 126
194 140 159
190 153 112
0 143 156
95 173 78
255 0 0
255 0 246
255 2 157
104 61 59
255 116 163
150 138 232
152 255 82
167 87 64
1 255 254
255 238 232
254 137 0
189 198 255
1 208 255
187 136 0
117 68 177
165 255 210
255 166 254
119 77 0
122 71 130
38 52 0
0 71 84
67 0 44
181 0 255
255 177 103
255 219 102
144 251 146
126 45 210
189 211 147
229 111 254
222 255 116
0 255 120
0 155 255
0 100 1
0 118 255
133 169 0
0 185 23
120 130 49
0 255 198
255 110 65
232 94 190""".split("\n")
    ],
    dtype=np.int32,
)


class AutosizingHeatmapWriter:
    """Writes a heatmap, automatically resizing it if necessary."""

    def __init__(self, heatmap_samples: int, heatmap_steps: int, colorize: bool = True):
        self.heatmap = np.zeros((heatmap_samples, heatmap_steps, 3), dtype=np.int32)
        self.heatmap_sample_factor = 1
        self.heatmap_step_factor = 1

        self.heatmap_sample_max = -1
        self.heatmap_step_max = -1

        self.colors_size = cpal.shape[0] if colorize else 1

    def add(self, sample_id: int, step: int, src: int) -> None:
        """
        Add a point to the heatmap (i.e. increase count at that position).

        Args:
            sample_id: The sample id (y-axis)
            step: The step (x-axis)
            src: The source rank (colorizing)
        """
        # Resize heatmap?
        while self.heatmap.shape[0] * self.heatmap_sample_factor <= sample_id:
            self.heatmap[: self.heatmap.shape[0] // 2] = self.heatmap[::2] + self.heatmap[1::2]
            self.heatmap[self.heatmap.shape[0] // 2 :] = 0
            self.heatmap_sample_factor *= 2
            self.heatmap_sample_max = 0
        while self.heatmap.shape[1] * self.heatmap_step_factor <= step:
            self.heatmap[:, : self.heatmap.shape[1] // 2] = (
                self.heatmap[:, ::2] + self.heatmap[:, 1::2]
            )
            self.heatmap[:, self.heatmap.shape[1] // 2 :] = 0
            self.heatmap_step_factor *= 2
            self.heatmap_step_max = 0
        # Save point
        step //= self.heatmap_step_factor
        sample_id //= self.heatmap_sample_factor
        self.heatmap[sample_id, step] += cpal[src % self.colors_size]
        self.heatmap_step_max = max(self.heatmap_step_max, step)
        self.heatmap_sample_max = max(self.heatmap_sample_max, sample_id)

    def save(self, path: Union[Path, str], gain: float):
        """
        Save the heatmap to the given path.

        Args:
            path: The path to save the heatmap to.
            gain: The gain (=multiplication factor) for the heatmap.

        Returns:
            The maximum sample id and step id that were used in the heatmap.
        """
        heatmap = self.heatmap[: self.heatmap_sample_max + 1, : self.heatmap_step_max + 1]

        heatmap = heatmap.astype(np.float32)
        heatmap = np.clip(heatmap * gain / heatmap.max((0, 1)) * 255, 0, 255).astype(np.uint8)

        Image.fromarray(heatmap).save(path)
        return (
            self.heatmap_sample_max * self.heatmap_sample_factor,
            self.heatmap_step_max * self.heatmap_step_factor,
        )


@click.command(name="analyze-debug")
@click.argument(
    "log_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--heatmap-path",
    type=click.Path(exists=False, writable=True, dir_okay=False, path_type=Path),
    default=Path("heatmap.png"),
)
@click.option(
    "--heatmap-steps",
    type=int,
    default=1000,
    help="Size of the heatmap in step direction. All steps will be downscaled to this size.",
)
@click.option(
    "--heatmap-samples",
    type=int,
    default=1000,
    help="Size of the heatmap in sample direction. All samples will be downscaled to this size.",
)
@click.option(
    "--heatmap-gain",
    type=float,
    default=10,
    help="Gain (=multiplication factor) for the heatmap",
)
@click.option(
    "--no-colors",
    is_flag=True,
    default=False,
    help="If set, disable colorizing ranks.",
)
def command(
    log_path: Path,
    heatmap_path: Path,
    heatmap_steps: int,
    heatmap_samples: int,
    heatmap_gain: float,
    no_colors: bool,
):
    """Internal tool to analyze randomness.

    The LOG_PATH should point to the folder with the debug log, or to a single log file."""

    heatmap = AutosizingHeatmapWriter(heatmap_samples, heatmap_steps, colorize=not no_colors)

    print(f"Analyzing log {log_path}...")

    if log_path.is_dir():
        log_paths = list(log_path.glob("*.json"))
    else:
        log_paths = [log_path]

    print(f"Analyzing {len(log_paths)} logs...")

    loader_log_loader = LogLoader(log_paths)

    key_index: dict[str, int] = defaultdict(lambda: len(key_index))

    for entry in loader_log_loader.read_entries():
        if isinstance(entry, LogLoader.LoaderIterator):
            print(
                f"Loader rank={entry.rank} loader_id={entry.loader_id} iter_id={entry.iter_id} nw={entry.num_workers} ws={entry.world_size}"
            )
        elif isinstance(entry, LogLoader.Worker):
            print(
                f"Worker rank={entry.loader.rank} loader_id={entry.loader.loader_id} iter_id={entry.loader.iter_id} worker_id={entry.worker_id}"
            )
        # elif isinstance(entry, LogLoader.LoadSample):
        #     print(f"LoadSample {entry.worker.worker_id} {entry.worker.loader.loader_id} {entry.worker.loader.rank} {entry.worker.loader.num_workers} {entry.base_path} {entry.key} {entry.index} {entry.epoch} {entry.epoch_count}")
        elif isinstance(entry, LogLoader.YieldSample):
            # print(f"YieldSample rank={entry.worker.loader.rank} loader_id={entry.worker.loader.loader_id} iter_id={entry.worker.loader.iter_id} wrk_id={entry.worker.worker_id} sample_idx={entry.sample_idx} iter_idx={entry.iter_idx} global_sample_idx={entry.global_sample_idx} keys={entry.keys}")
            if entry.keys is not None:
                for key in entry.keys:
                    heatmap.add(
                        key_index[key], entry.global_sample_idx, src=entry.worker.loader.rank
                    )
        elif isinstance(entry, LogLoader.LoadNextEpoch):
            # print(f"LoadNextEpoch rank={entry.worker.loader.rank} loader_id={entry.worker.loader.loader_id} iter_id={entry.worker.loader.iter_id} wrk_id={entry.worker.worker_id} epoch_idx={entry.epoch_idx} epoch_sample_count={entry.epoch_sample_count}")
            pass
        elif isinstance(entry, LogLoader.StopIteration):
            # print(f"StopIteration rank={entry.loader.rank} loader_id={entry.loader.loader_id} iter_id={entry.loader.iter_id}")
            pass

    if len(key_index) == 0:
        raise click.ClickException("No data found in logs")

    print(f"Found {len(key_index)} unique sample keys, {heatmap.heatmap_step_max + 1} steps")

    # print(f"Heatmap factors: {heatmap_sample_factor} samples, {heatmap_step_factor} steps")
    # print(f"Heatmap max: {heatmap_sample_max} samples, {heatmap_step_max} steps")
    max_sample, max_step = heatmap.save(heatmap_path, heatmap_gain)
    print(f"Wrote heatmap to {heatmap_path}")
    print("Heatmap axes:")
    print(f"  x-axis: {max_step + 1} worker steps")
    print(f"  y-axis: {max_sample + 1} samples")


class LogEntry(TypedDict):
    """
    Chrome tracing log entry.
    *ph*ase values:
    - B: Begin
    - E: End
    - i: Instant
    - b: Begin (async)
    - e: End (async)
    - n: Instant (async)
    - C: Counter
    - M: Metadata
    - s: Flow start
    - t: Flow step
    - f: Flow end
    """

    ph: Literal["B", "E", "i", "b", "e", "n", "C", "M", "s", "t", "f"]
    name: str
    id: int
    ts: int
    pid: int
    tid: int
    args: dict
    s: Literal["t", "p", "g"]


class LogLoader:
    """Loads a chrome tracing log file. Extract specific information from it."""

    _re_pname = re.compile(r"^dprank(\d+)(?:_worker(\d+))?$")

    def __init__(self, paths: list[Path]):
        self._paths = paths

    def _log_reader(self, path: Path) -> Generator[LogEntry, None, None]:
        """Reads a log file and yields a tuple of the line and the ts."""
        had_end = False
        with open(path, "rb") as f:
            assert f.read(2) == b"[\n", "Log file must start with a JSON array"
            for line in f:
                if not line:
                    assert had_end, "Log file must end with a JSON array"
                if line.endswith(b"]\n"):
                    had_end = True
                else:
                    assert line.endswith(b",\n"), f"Log file must be newline-terminated: {line}"
                yield json.loads(line[:-2])
            assert had_end, "Log file must end with a JSON array"

    def _log_reader_all(self) -> Generator[LogEntry, None, None]:
        """Reads all log files and yields a tuple of the line and the ts."""
        if len(self._paths) == 1:
            yield from self._log_reader(self._paths[0])
        else:
            for entry in merge_log_reader(self._paths):
                yield json.loads(entry)

    @dataclass
    class LoaderIterator:
        world_size: int
        rank: int
        num_workers: int
        loader_id: int
        iter_id: int

    @dataclass
    class Worker:
        worker_id: int
        loader: "LogLoader.LoaderIterator"

    @dataclass
    class LoadSample:
        worker: "LogLoader.Worker"
        base_path: str
        key: str
        global_sample_index: int
        sample_count: int
        epoch_idx: int
        epoch_sample_count: int

    @dataclass
    class LoadNextEpoch:
        worker: "LogLoader.Worker"
        epoch_idx: int
        epoch_sample_count: int

    @dataclass
    class YieldSample:
        worker: "LogLoader.Worker"
        worker_sample_idx: int
        sample_idx: int
        iter_idx: int
        global_sample_idx: int
        keys: list[str] | None

    @dataclass
    class StopIteration:
        loader: "LogLoader.LoaderIterator"

    def read_entries(self):
        # Maps pid to (rank, worker_id|None)
        procs: dict[int, tuple[int, int | None]] = dict()
        # Maps (pid, tid) to worker_id|None, only for main threads
        proc_workers: dict[tuple[int, int], int | None] = dict()
        # Maps (pid, tid) to worker
        workers_by_pid_tid: dict[tuple[int, int], LogLoader.Worker] = dict()
        # Maps (rank, loader_id, worker_id) to worker
        workers_by_rank_loader_id_iter_id_worker_id: dict[
            tuple[int, int, int], LogLoader.Worker
        ] = dict()
        # Maps (rank, loader_id) to loader
        loaders_by_rank_loader_id: dict[tuple[int, int], LogLoader.LoaderIterator] = dict()
        # Maps (rank, loader_id, iter_id) to loader
        loaders_by_rank_loader_id_iter_id: dict[tuple[int, int, int], LogLoader.LoaderIterator] = (
            dict()
        )
        for log_entry in self._log_reader_all():
            ph = log_entry["ph"]
            name = log_entry.get("name")
            if ph == "M":
                if name == "process_name":
                    pid = log_entry["pid"]
                    pname = log_entry["args"]["name"]
                    m = self._re_pname.match(pname)
                    if m:
                        rank = int(m.group(1))
                        if m.group(2) is not None:
                            worker_id = int(m.group(2))
                        else:
                            worker_id = None
                        procs[log_entry["pid"]] = (rank, worker_id)
                if name == "thread_name":
                    thread_name = log_entry["args"]["name"]
                    pid = log_entry["pid"]
                    tid = log_entry["tid"]
                    if thread_name in ("main", "worker_main"):
                        proc_workers[(pid, tid)] = procs[pid][1]
            if ph == "n":
                if name == "WebdatasetSampleLoaderDataset._slices_iter.yield":
                    yield LogLoader.LoadSample(
                        worker=workers_by_pid_tid[(log_entry["pid"], log_entry["tid"])],
                        base_path=log_entry["args"]["base_path"],
                        key=log_entry["args"]["key"],
                        global_sample_index=log_entry["args"]["global_sample_index"],
                        sample_count=log_entry["args"]["sample_count"],
                        epoch_idx=log_entry["args"]["epoch_idx"],
                        epoch_sample_count=log_entry["args"]["epoch_sample_count"],
                    )
                elif name == "WebdatasetSampleLoaderDataset._slices_iter.next_epoch":
                    yield LogLoader.LoadNextEpoch(
                        worker=workers_by_pid_tid[(log_entry["pid"], log_entry["tid"])],
                        epoch_idx=log_entry["args"]["epoch_idx"],
                        epoch_sample_count=log_entry["args"]["epoch_sample_count"],
                    )
                elif name in ("SavableDataLoader.yield", "BasicDataLoader.yield"):
                    rank = procs[log_entry["pid"]][0]
                    yield LogLoader.YieldSample(
                        worker=workers_by_rank_loader_id_iter_id_worker_id[
                            (rank, log_entry["args"]["loader_id"], log_entry["args"]["worker_id"])
                        ],
                        worker_sample_idx=log_entry["args"]["worker_sample_idx"],
                        sample_idx=log_entry["args"]["sample_idx"],
                        iter_idx=log_entry["args"]["iter_idx"],
                        global_sample_idx=log_entry["args"]["global_sample_idx"],
                        keys=log_entry["args"].get("keys", None),
                    )
                elif name in ("SavableDataLoader.StopIteration", "BasicDataLoader.StopIteration"):
                    rank = procs[log_entry["pid"]][0]
                    yield LogLoader.StopIteration(
                        loader=loaders_by_rank_loader_id_iter_id[
                            (rank, log_entry["args"]["loader_id"], log_entry["args"]["iter_id"])
                        ],
                    )
            elif ph == "B":
                if name in (
                    "SavableDatasetWrapper.__iter__",
                    "SimpleSavableDatasetWrapper.__iter__",
                ):
                    rank = procs[log_entry["pid"]][0]
                    # This is not 100% correct, but it's the best mapping we can get right now.
                    loader = loaders_by_rank_loader_id[(rank, log_entry["args"]["loader_id"])]
                    worker = LogLoader.Worker(
                        worker_id=log_entry["args"]["worker_id"],
                        loader=loader,
                    )
                    workers_by_pid_tid[(log_entry["pid"], log_entry["tid"])] = worker
                    workers_by_rank_loader_id_iter_id_worker_id[
                        (rank, loader.loader_id, worker.worker_id)
                    ] = worker
                    yield worker
            elif ph == "b":
                if name in ("SavableDataLoader.__iter__", "BasicDataLoader.__iter__"):
                    rank = procs[log_entry["pid"]][0]
                    loader = loaders_by_rank_loader_id[(rank, log_entry["args"]["loader_id"])]
                    loader.iter_id = log_entry["args"]["iter_id"]
                    loaders_by_rank_loader_id_iter_id[(rank, loader.loader_id, loader.iter_id)] = (
                        loader
                    )
                    yield loader
                elif name in ("SavableDataLoader", "BasicDataLoader"):
                    cfg_rank = log_entry["args"]["worker_config"]["rank"]
                    rank = procs[log_entry["pid"]][0]
                    assert rank == cfg_rank, f"Rank mismatch: {rank} != {cfg_rank}"

                    loader = LogLoader.LoaderIterator(
                        world_size=log_entry["args"]["worker_config"]["world_size"],
                        rank=rank,
                        num_workers=log_entry["args"]["worker_config"]["num_workers"],
                        loader_id=log_entry["args"]["loader_id"],
                        iter_id=-1,
                    )
                    # This is not 100% correct, but it's the best mapping we can get right now.
                    loaders_by_rank_loader_id[(rank, log_entry["args"]["loader_id"])] = loader
                    yield loader


if __name__ == "__main__":
    command()
