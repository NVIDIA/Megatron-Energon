# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import List, Optional

import click
import torch

from megatron.energon.epathlib import EPath
from megatron.energon.savable_loader import SavableDataLoaderState


class RankStateIterable:
    """Iterates the SavableDatasetCheckpoints of mulitple ranks in a round-robin fashion."""

    def __init__(self, state_files: List[EPath]):
        # First open the first one to figure out if this is a global checkpoint or not
        first_state = torch.load(str(state_files[0]), weights_only=False)

        if isinstance(first_state, SavableDataLoaderState):
            self.rank_states = [first_state] + [
                torch.load(str(state_file), weights_only=False) for state_file in state_files[1:]
            ]
            self.is_global_checkpoint = False
        elif isinstance(first_state, list):
            assert len(state_files) == 1, "Global checkpoint must contain exactly one file"
            assert all(isinstance(state, SavableDataLoaderState) for state in first_state)
            self.rank_states = first_state
            self.is_global_checkpoint = True
        else:
            raise ValueError(f"Unknown checkpoint type: {type(first_state)}")

        self.rank_cur_worker = [0] * len(self.rank_states)
        self.rank_worker_offset = [state.next_worker_id for state in self.rank_states]

        self.rank_num_workers = [len(state.worker_states) for state in self.rank_states]
        assert all(
            self.rank_num_workers[0] == num_workers for num_workers in self.rank_num_workers
        ), "All ranks must have the same number of workers."

    def get_num_ranks(self):
        return len(self.rank_states)

    def get_num_workers(self):
        return self.rank_num_workers[0]

    def __iter__(self):
        """Iterates the SavableDatasetCheckpoints of mulitple ranks in a round-robin fashion."""
        for rank, state in enumerate(self.rank_states):
            for worker_state in state.worker_states:
                yield worker_state


def natural_sort_key(s):
    """
    Function to use for natural sorting of filenames.

    This splits the input string by numbers and non-numbers and ensures
    that numbers are compared as integers, not as strings.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


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
@click.option(
    "--new-world-size", type=int, help="Number of ranks to redistribute to", required=False
)
def command_redist(
    input_files: List[EPath], output_path: EPath, new_world_size: Optional[int] = None
):
    """Redistribute a checkpoint.

    Read checkpoint files from INPUT_FILES and redistribute them for a new
    number of ranks. Write the output to OUTPUT_PATH."""

    # Verify input files
    if not input_files:
        raise click.ClickException("No input files provided")

    input_file_list = sorted(input_files, key=lambda x: natural_sort_key(x.name))
    input_file_list = [input_file.absolute() for input_file in input_file_list]

    click.echo(f"Processing {len(input_file_list)} checkpoint files")

    output_path = output_path.absolute()

    # Determine if we're processing a single global checkpoint or multiple rank files
    rsi = RankStateIterable(input_file_list)

    if not rsi.rank_states:
        raise click.ClickException("No valid checkpoint states found")

    if new_world_size is None:
        click.echo(f"Current world size: {rsi.get_num_ranks()}")
        click.echo(f"Current number of workers per rank: {rsi.get_num_workers()}")
        new_world_size = click.prompt("Please enter the new world size", type=int)
        assert isinstance(new_world_size, int)

    if new_world_size <= 0:
        raise click.ClickException("New world size must be greater than 0")

    total_num_workers = rsi.get_num_workers() * rsi.get_num_ranks()
    assert total_num_workers % new_world_size == 0, (
        "New world size must be a multiple of the current world size"
    )
    new_workers_per_rank = total_num_workers // new_world_size

    # Ensure output directory exists
    output_path.mkdir(exist_ok=True, parents=True)

    new_rank_states = [list() for _ in range(new_world_size)]
    rsi_iter = iter(rsi)
    for rank_idx in range(new_world_size):
        for _ in range(new_workers_per_rank):
            state = next(rsi_iter)
            new_rank_states[rank_idx].append(state)

    assert all(
        len(new_rank_states[0]) == len(new_rank_states[rank]) for rank in range(1, new_world_size)
    ), "All ranks must have the same number of workers, also for the new distribution."

    new_states = [
        SavableDataLoaderState(
            worker_states=new_rank_state,
            next_worker_id=0,  # Reset the next worker ID
        )
        for new_rank_state in new_rank_states
    ]

    # Save the redistributed checkpoint
    if rsi.is_global_checkpoint:
        # Save as a single global checkpoint file
        output_file = output_path / input_file_list[0].name
        torch.save(new_states, str(output_file))
        click.echo(f"Saved global checkpoint to {output_file}")
    else:
        # Save as individual rank files
        for rank_idx, rank_state in enumerate(new_states):
            output_file = output_path / f"state_rank{rank_idx}.pt"
            torch.save(rank_state, str(output_file))
        click.echo(f"Saved {new_world_size} rank checkpoint files to {output_path}")


@click.command(name="info")
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=EPath),
    required=True,
)
def command_info(input_files: List[EPath]):
    """Display information about a checkpoint.

    Read a checkpoint from CHECKPOINT_PATH (either a single file or directory with *.pt files)
    and display information about it.
    """

    # Load the checkpoint(s)
    rsi = RankStateIterable(input_files)

    # Display basic information
    if rsi.is_global_checkpoint:
        click.echo("Checkpoint type: Global checkpoint")
    else:
        click.echo("Checkpoint type: Per-rank checkpoint files")

    click.echo(f"Number of ranks: {rsi.get_num_ranks()}")
    click.echo(f"Number of workers per rank: {rsi.get_num_workers()}")

    # Additional detailed information
    click.echo("\nDetailed information:")
    for rank_idx, state in enumerate(rsi.rank_states):
        if rsi.is_global_checkpoint:
            click.echo(f"  Rank {rank_idx}:")
        else:
            click.echo(f"  Rank {rank_idx} ({input_files[rank_idx].name}):")
        click.echo(f"    Next worker ID: {state.next_worker_id}")
        click.echo(f"    Number of worker states: {len(state.worker_states)}")


@click.group(
    name="checkpoint",
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.pass_context
def command(ctx):
    """Tools for energon checkpoints."""

    # This is needed to show help if no subcommand is provided
    if ctx.invoked_subcommand is None:
        click.echo(command.get_help(ctx))


command.add_command(command_redist)
command.add_command(command_info)

if __name__ == "__main__":
    command()
