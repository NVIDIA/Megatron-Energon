# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for energon CLI tools."""

import click

from megatron.energon.tools.analyze_debug import command as analyze_debug_command
from megatron.energon.tools.checkpoint import command as checkpoint_command
from megatron.energon.tools.info import command as info_command
from megatron.energon.tools.lint import command as lint_command
from megatron.energon.tools.mount import command as mount_command
from megatron.energon.tools.prepare import command as prepare_command
from megatron.energon.tools.preview import command as preview_command


def print_version(ctx, param, value):
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    try:
        from megatron.energon import __version__

        click.echo(f"megatron-energon {__version__}")
    except ImportError:
        click.echo("megatron-energon version unknown")
    ctx.exit()


@click.group(context_settings=dict(help_option_names=["-h", "--help"]), invoke_without_command=True)
@click.option(
    "--version",
    "-V",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=print_version,
    help="Show version and exit.",
)
@click.pass_context
def main(ctx):
    """A set of tools that energon provides.

    Among other things, you can use it to lint or preprocess your dataset.

    See help of commands to learn more."""

    # This is needed to show help if no subcommand is provided
    if ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


main.add_command(analyze_debug_command)
main.add_command(checkpoint_command)
main.add_command(lint_command)
main.add_command(info_command)
main.add_command(mount_command)
main.add_command(prepare_command)
main.add_command(preview_command)

if __name__ == "__main__":
    main()
