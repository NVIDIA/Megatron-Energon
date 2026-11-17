# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import List, Optional

import click
import torch

from megatron.energon.epathlib import EPath
from megatron.energon.savable_loader import (
    PhysicalWorkerCheckpoint,
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    _merge_worker_states,
)
from megatron.energon.worker import WorkerConfig


def natural_sort_key(s):
    """Sort filenames numerically where they contain numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


class CheckpointFiles:
    """Read physical rank states or logical collections, independently of filenames."""

    def __init__(self, state_files: List[EPath]):
        self.is_global_checkpoint = False
        self.is_physical = None
        self.megatron_style = None
        self.rank_states = []
        self.entries = []
        self.physical_states = []
        for state_file in state_files:
            payload = torch.load(str(state_file), weights_only=False)
            wrapped = isinstance(payload, dict) and "dataloader_state_dict" in payload
            if self.megatron_style is not None and wrapped != self.megatron_style:
                raise ValueError("Input files must use the same checkpoint envelope")
            self.megatron_style = wrapped
            if wrapped:
                payload = payload["dataloader_state_dict"]
            if payload is None or isinstance(payload, SavableDataLoaderState):
                self._add_state(payload, state_file.name)
            elif isinstance(payload, (list, tuple)) and all(
                state is None or isinstance(state, SavableDataLoaderState) for state in payload
            ):
                self.is_global_checkpoint = True
                for rank, state in enumerate(payload):
                    self._add_state(state, f"{state_file.name}: rank {rank}")
            elif isinstance(payload, (Mapping, list, tuple)):
                self.is_global_checkpoint = True
                pairs = payload.items() if isinstance(payload, Mapping) else payload
                for logical_id, state in pairs:
                    if not isinstance(state, SavableDataLoaderState) or not isinstance(
                        state.worker_states, dict
                    ):
                        raise ValueError("Expected (logical id, logical checkpoint) pairs")
                    if set(state.worker_states) != {logical_id}:
                        raise ValueError("Logical id must match the checkpoint's worker id")
                    self._add_state(state, f"{state_file.name}: logical worker {logical_id}")
            else:
                raise ValueError(f"Unknown checkpoint type: {type(payload)}")
        if self.is_global_checkpoint and self.is_physical and len(state_files) != 1:
            raise ValueError("Supply a global checkpoint as a single input file")
        self.is_global_checkpoint = self.is_global_checkpoint and len(state_files) == 1
        if not self.rank_states:
            raise ValueError("No initialized checkpoint states found")
        self.state = self.rank_states[0]
        for state in self.rank_states:
            if (
                state.logical_workers != self.state.logical_workers
                or state.micro_batch_size != self.state.micro_batch_size
                or state.iteration_index != self.state.iteration_index
            ):
                raise ValueError(
                    "Checkpoints require matching logical count, batch size and iteration"
                )
        if not self.is_physical:
            self.state = _merge_worker_states(self.rank_states)

    def _add_state(self, state, label):
        self.entries.append((label, state))
        if state is None:
            self.physical_states.append(None)
            return
        physical = isinstance(state.worker_states, list)
        if not physical and not isinstance(state.worker_states, dict):
            raise ValueError("Expected physical or logical worker checkpoints")
        if self.is_physical is not None and physical != self.is_physical:
            raise ValueError("Cannot mix physical and logical checkpoint files")
        self.is_physical = physical
        self.rank_states.append(state)
        if physical:
            self.physical_states.append(state)

    def logical_state(self) -> SavableDataLoaderState:
        """Select each logical group's last contributor from a complete physical save."""
        if not self.is_physical:
            state = self.state
        else:
            workers = len(self.state.worker_states)
            if not workers or any(
                state is not None and len(state.worker_states) != workers
                for state in self.physical_states
            ):
                raise ValueError("Physical ranks must have the same positive number of workers")
            config = WorkerConfig(
                rank=0,
                world_size=len(self.physical_states),
                num_workers=workers,
                logical_workers=self.state.logical_workers,
            )
            checkpoints = {}
            for group in range(self.state.logical_workers):
                contributor = config.logical_worker_last_physical_worker(
                    group, self.state.iteration_index
                )
                rank, worker = divmod(contributor, workers)
                source = self.physical_states[rank]
                if source is None:
                    if self.state.iteration_index:
                        raise ValueError(f"Missing physical checkpoint for rank {rank}")
                    checkpoint = SavableDatasetCheckpoint(state=None, offset=0)
                else:
                    checkpoint = source.worker_states[worker]
                checkpoints[group] = SavableDatasetCheckpoint(
                    state=checkpoint.state, offset=checkpoint.offset
                )
            state = replace(self.state, worker_states=checkpoints)
        if set(state.worker_states) != set(range(state.logical_workers)):
            raise ValueError("Input files must cover every logical worker group")
        return state

    def write(self, path: EPath, state) -> None:
        if self.megatron_style:
            state = {"dataloader_state_dict": state}
        path.local_path().parent.mkdir(exist_ok=True, parents=True)
        torch.save(state, str(path))


@click.command(name="redist")
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=EPath),
    required=True,
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=EPath),
)
@click.option("--new-world-size", type=int, help="Number of destination ranks.")
@click.option(
    "--output-filename",
    default="state_rank{rank}.pt",
    show_default=True,
    help="Filename template for rank outputs; must contain {rank}.",
)
def command_redist(
    input_files: List[EPath],
    output_path: EPath,
    new_world_size: Optional[int] = None,
    output_filename: str = "state_rank{rank}.pt",
):
    """Redistribute INPUT_FILES into OUTPUT_PATH.

    Supply every physical rank file in natural filename order, a global physical
    list, or logical states covering every logical worker. Destination physical
    workers equal logical workers; workers per rank are logical_workers / world_size.
    """
    input_file_list = sorted(input_files, key=lambda path: natural_sort_key(path.name))
    click.echo(f"Processing {len(input_file_list)} checkpoint files")
    try:
        files = CheckpointFiles(input_file_list)
        state = files.logical_state()
        if new_world_size is None:
            click.echo(f"Logical workers: {state.logical_workers}")
            new_world_size = click.prompt("Please enter the new DP world size", type=int)
        if new_world_size <= 0 or state.logical_workers % new_world_size:
            raise ValueError("New world size must be positive and divide the logical worker count")
        workers = state.logical_workers // new_world_size
        new_states = []
        for rank in range(new_world_size):
            groups = range(rank * workers, (rank + 1) * workers)
            if files.is_physical:
                checkpoints = [
                    PhysicalWorkerCheckpoint(
                        state=state.worker_states[group].state,
                        offset=state.worker_states[group].offset,
                        previous_worker_id=group,
                    )
                    for group in groups
                ]
                new_state = replace(state, worker_states=checkpoints)
            else:
                new_state = {
                    group: replace(state, worker_states={group: state.worker_states[group]})
                    for group in groups
                }
            new_states.append(new_state)
        if files.is_global_checkpoint:
            if files.is_physical:
                payload = new_states
            else:
                payload = tuple(
                    (group, replace(state, worker_states={group: checkpoint}))
                    for group, checkpoint in sorted(state.worker_states.items())
                )
            output_file = output_path / input_file_list[0].name
            files.write(output_file, payload)
            click.echo(f"Saved global checkpoint to {output_file}")
        else:
            if "{rank}" not in output_filename:
                raise ValueError("Output filename must contain {rank}")
            names = [output_filename.format(rank=rank) for rank in range(new_world_size)]
            if any(not name or "/" in name or "\\" in name for name in names):
                raise ValueError("Output filename must be a filename without directories")
            if len(set(names)) != len(names):
                raise ValueError("Output filename must produce a different name for every rank")
            for name, rank_state in zip(names, new_states):
                files.write(output_path / name, rank_state)
            click.echo(f"Saved {new_world_size} rank checkpoint files to {output_path}")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command(name="info")
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=EPath),
    required=True,
)
def command_info(input_files: List[EPath]):
    """Display physical or logical checkpoint information from INPUT_FILES."""
    try:
        files = CheckpointFiles(sorted(input_files, key=lambda path: natural_sort_key(path.name)))
    except (ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    kind = "Physical" if files.is_physical else "Logical"
    scope = "global collection" if files.is_global_checkpoint else "rank checkpoint files"
    click.echo(f"Checkpoint type: {kind} {scope}")
    click.echo(f"Number of checkpoint files: {len(input_files)}")
    click.echo(f"Number of logical workers: {files.state.logical_workers}")
    click.echo("\nDetailed information:")
    for label, state in files.entries:
        click.echo(f"  {label}:")
        if state is None:
            click.echo("    Not initialized")
            continue
        click.echo(f"    Iteration index: {state.iteration_index}")
        if not files.is_physical:
            click.echo(f"    Logical worker IDs: {sorted(state.worker_states)}")
        click.echo(f"    Number of worker states: {len(state.worker_states)}")


@click.group(
    name="checkpoint",
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.pass_context
def command(ctx):
    """Tools for energon checkpoints."""
    if ctx.invoked_subcommand is None:
        click.echo(command.get_help(ctx))


command.add_command(command_redist)
command.add_command(command_info)

if __name__ == "__main__":
    command()
