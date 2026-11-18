# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional

import click
import torch

from megatron.energon.checkpoint import RankStateIterable, natural_sort_key
from megatron.energon.dataloader.dataloader import RankState, _split_batch_restore_key
from megatron.energon.epathlib import EPath
from megatron.energon.savable_loader import SavableDataLoaderState


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
@click.option("--new-micro-batch-size", type=click.IntRange(min=1))
def command_redist(
    input_files: List[EPath],
    output_path: EPath,
    new_world_size: Optional[int] = None,
    new_micro_batch_size: Optional[int] = None,
):
    """Redistribute a checkpoint.

    Read checkpoint files from INPUT_FILES and redistribute them for a new
    number of ranks. Write the output to OUTPUT_PATH."""

    # Verify input files
    if not input_files:
        raise click.ClickException("No input files provided")

    input_file_list = sorted(input_files, key=lambda x: natural_sort_key(x.name))

    click.echo(f"Processing {len(input_file_list)} checkpoint files")

    # Determine if we're processing a single global checkpoint or multiple rank files
    rsi = RankStateIterable(input_file_list)

    if not rsi.rank_states:
        raise click.ClickException("No valid checkpoint states found")

    if new_world_size is None:
        click.echo(f"Current DP world size: {rsi.get_num_ranks()}")
        click.echo(f"Current number of workers per DP rank: {rsi.get_num_workers()}")
        new_world_size = click.prompt("Please enter the new DP world size", type=int)
        assert isinstance(new_world_size, int)

    if new_world_size <= 0:
        raise click.ClickException("New world size must be greater than 0")

    total_num_workers = rsi.get_num_workers() * rsi.get_num_ranks()
    assert total_num_workers % new_world_size == 0, (
        "New DP world size must be a multiple of the current DP world size"
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

    old_size = rsi.get_micro_batch_size()
    batch_size = old_size if new_micro_batch_size is None else new_micro_batch_size
    if isinstance(rsi.rank_states[0], RankState):
        factor = 1
        if old_size != batch_size:
            if old_size is None or batch_size is None or old_size % batch_size:
                raise click.ClickException("The new micro batch size must divide the original size")
            factor = old_size // batch_size
        new_states = [
            RankState(
                worker_states=[worker for worker, _, _ in rank],
                prefetched_restore_keys=[
                    keys
                    if factor == 1
                    else [piece for key in keys for piece in _split_batch_restore_key(key, factor)]
                    for _, keys, _ in rank
                ],
                exhausted_workers=[exhausted for _, _, exhausted in rank],
                next_worker_id=0,
                micro_batch_size=batch_size,
            )
            for rank in new_rank_states
        ]
    else:
        if batch_size != old_size:
            raise click.ClickException(
                "Micro batch size changes require a custom loader checkpoint"
            )
        new_states = [
            SavableDataLoaderState(worker_states=rank, next_worker_id=0, micro_batch_size=old_size)
            for rank in new_rank_states
        ]

    # Save the redistributed checkpoint
    if rsi.is_global_checkpoint:
        # Save as a single global checkpoint file
        output_file = output_path / input_file_list[0].name
        torch.save(new_states, str(output_file))
        click.echo(f"Saved global checkpoint to {output_file}")
    else:
        rsi.write_new_states_to_folder(output_path, new_states)

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

    click.echo(f"Number of DP ranks: {rsi.get_num_ranks()}")
    click.echo(f"Number of workers per DP rank: {rsi.get_num_workers()}")

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
