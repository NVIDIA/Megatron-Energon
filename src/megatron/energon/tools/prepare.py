# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import json
import re
import typing
from types import FunctionType
from typing import Any, List, Optional, Type

import click
import yaml

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory, CrudeWebdataset
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME


class CrudeSampleDummy:
    pass


def type_str(tp: Type) -> str:
    """Returns a human-readable string for a type."""
    if typing.get_origin(tp) is not None:
        return repr(tp)
    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__qualname__
        return f"{tp.__module__}.{tp.__qualname__}"
    if tp is ...:
        return "..."
    if isinstance(tp, FunctionType):
        return tp.__name__
    return repr(tp)


def sample_loader_template(fields: dict, parts: list):
    """Returns a template for a sample_loader.py file."""

    fields_str = ""
    for field in fields:
        if field.name in ("__key__", "__restore_key__", "__subflavor__", "__subflavors__"):
            continue
        line = f"""        {field.name}=raw["TODO"],  # expected type: {type_str(field.type)}"""
        if field.default is not dataclasses.MISSING:
            line += ", default: " + repr(field.default)
        fields_str += line + "\n"

    return "\n".join(
        [
            "# This file was automatically generated by `energon prepare`.",
            "# TODO: Edit it to return the proper fields",
            "# import torch",
            "",
            "def sample_loader(raw: dict) -> dict:"
            "    # Note: Images are already decoded to tensors",
            "    # TODO: Set the correct values for all (required) fields",
            "    return dict(",
            fields_str,
            "    )",
            "",
            "def part_filter(part: str) -> bool:",
            "    # TODO: Filter for parts required by the sample_loader",
            "    # E.g. if your dataset contains jpeg, txt and json, but you won't use json,",
            "    # remove it from the list, such that it is not decoded. If you need all, keep as is",
            f"    return part in {tuple(parts)!r}",
            "",
        ]
    )


def printify_json(data: Any) -> Any:
    """Shortens json data to a human-readable length."""
    if isinstance(data, dict):
        return {k: printify_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 3:
            return [printify_json(v) for v in data[:3]] + ["..."]
        return [printify_json(v) for v in data]
    elif isinstance(data, str):
        return data[:25] + ("..." if len(data) > 25 else "")
    return data


@click.command(name="prepare")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.option(
    "--progress/--no-progress",
    default=True,
)
@click.option(
    "--split-parts",
    help="Path pattern for parts in the form 'train:train/{000000-009999}.tar'. Will ignore ratio.",
    multiple=True,
    default=None,
)
@click.option(
    "--exclude",
    help="Exclude tar file paths (relative to root) matching that regex (at any position)",
)
@click.option(
    "--num-workers",
    type=int,
    default=16,
    help="Number of workers to use to index tar files",
)
@click.option(
    "--tar-index-only",
    help="Only (re)generate the tar-index",
    is_flag=True,
)
@click.option(
    "--shuffle-tars",
    help="If set, the tar files will be shuffled before splitting.",
    is_flag=True,
)
def command(
    path: EPath,
    progress: bool,
    split_parts: Optional[List[str]],
    exclude: str,
    num_workers: int,
    tar_index_only: bool,
    shuffle_tars: bool,
):
    """Prepare WebDataset for use with energon.

    The PATH should point to the folder with the dataset.
    This tool will add the required metadata yaml files to the dataset. See README.md for more
    details.
    """

    path = path.absolute()

    if tar_index_only:
        assert (path / MAIN_FOLDER_NAME / ".info.yaml").is_file(), "No .info.yaml found"
        with (path / MAIN_FOLDER_NAME / ".info.yaml").open("r") as f:
            info = yaml.safe_load(f)
        all_tars = list(info["shard_counts"].keys())
    else:
        if (path / MAIN_FOLDER_NAME / "dataset.yaml").is_file() or (
            path / MAIN_FOLDER_NAME / ".info.yaml"
        ).is_file():
            if not click.confirm(
                "It seems the dataset had already been prepared. Do you want to continue?"
            ):
                return

        all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
        all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    if exclude:
        all_tars = [p for p in all_tars if not re.search(exclude, p)]

    if len(all_tars) == 0:
        click.echo("Did not find any tar files. Exiting.")
        return

    if not tar_index_only:
        click.echo(f"Found {len(all_tars)} tar files in total. The first and last ones are:")
        click.echo(f"- {all_tars[0]}")
        click.echo(f"- {all_tars[-1]}")
        click.echo(
            "If you want to exclude some of them, cancel with ctrl+c and specify an exclude "
            "filter in the command line."
        )

    if split_parts:
        split_parts_patterns = [x.split(":", 1) for x in split_parts]
        split_parts_ratio = None
    elif not tar_index_only:
        split_input = click.prompt(
            'Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1"', type=str
        )
        # Extract split floats
        try:
            split = [float(x.strip()) for x in split_input.split(",")]
            assert len(split) == 3
        except (ValueError, AssertionError):
            click.echo("Invalid split. Stopping.")
            return
        split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
        split_parts_patterns = None
    else:
        split_parts_ratio = None
        split_parts_patterns = None

    if progress:

        def progress_fn(els, length=None):
            with click.progressbar(
                els,
                label="Indexing shards",
                show_pos=True,
                length=length,
            ) as bar:
                for el in bar:
                    yield el

    else:

        def progress_fn(els, length=None):
            return els

    found_types = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        progress_fn=progress_fn,
        tar_index_only=tar_index_only,
        shuffle_seed=42 if shuffle_tars else None,
        workers=num_workers,
    )
    found_types = list(found_types)
    if tar_index_only:
        return

    # Print json of first two samples
    for sample_idx, data in enumerate(
        BaseWebdatasetFactory.iter_dataset_content(path / all_tars[0], ("json",))
    ):
        print(f"Sample {sample_idx}, keys:")
        for key in data.keys():
            print(f" - {key}")
        if "json" in data:
            print(f"Json content of sample {sample_idx} of {all_tars[0]}:")
            print(json.dumps(printify_json(json.loads(data["json"])), indent=2))
        if sample_idx >= 1:
            break

    if len(found_types) > 10:
        click.echo(
            f"Found the following part types in the dataset: {', '.join(found_types[:10])} and more.."
        )
        allow_interactive_field_map = False
    else:
        click.echo(f"Found the following part types in the dataset: {', '.join(found_types)}")
        allow_interactive_field_map = True

    if click.confirm("Do you want to create a dataset.yaml interactively?", default=True):
        # Get a list of all classes in megatron.energon that are subclasses of WebdatasetBase
        import megatron.energon as data_import

        all_classes = []
        for name, cls in inspect.getmembers(data_import):
            if isinstance(cls, type) and issubclass(cls, Sample):
                all_classes.append(cls)

        all_classes.append(
            ("Crude sample (plain dict for cooking)", CrudeSampleDummy)
        )  # Tuple is (Printed name, resulting class)

        # Print all classes and ask user to pick one
        click.echo("The following sample types are available:")
        for i, cls in enumerate(all_classes):
            if isinstance(cls, tuple):
                click.echo(f"{i}. {cls[0]}")
            else:
                click.echo(f"{i}. {cls.__name__}")
        while True:
            choice = click.prompt("Please enter a number to choose a class", type=int)
            try:
                cls = all_classes[choice]
                break
            except IndexError:
                click.echo("Invalid choice. Please try again.")
                continue

        if isinstance(cls, tuple):
            cls = cls[1]

        # Ask user to enter field_map
        sample_type_source = inspect.getsource(cls)
        click.echo("The sample type you selected:\n")
        click.echo(sample_type_source)

        dataset_definition = {
            "sample_type": {
                "__module__": "megatron.energon",
                "__class__": cls.__name__,
            }
        }

        if cls == CrudeSampleDummy:
            click.echo(
                "CrudeWebdataset does not need a field map. You will need to provide a `Cooker` for your dataset samples in your `TaskEncoder`."
            )
            click.echo(
                "Furthermore, you might want to add `subflavors` in your meta dataset specification."
            )
        else:
            if not allow_interactive_field_map:
                click.echo(
                    "You cannot set a field_map for this dataset. You will need a sample_loader."
                )

            if allow_interactive_field_map and click.confirm(
                "Do you want to set a simple field_map[Y] (or write your own sample_loader [n])?",
                default=True,
            ):
                click.echo(
                    "\nFor each field, please specify the corresponding name in the WebDataset."
                )
                click.echo(f"Available types in WebDataset: {', '.join(found_types)}")
                click.echo(f"Leave empty for skipping optional field")
                click.echo(
                    f"You may also access json fields e.g. by setting the field to: json[field][field]"
                )
                click.echo(f"You may also specify alternative fields e.g. by setting to: jpg,png")

                click.echo(f"Please enter the field_map for {cls.__name__}:")

                dataset_definition["field_map"] = field_map = {}
                for field in dataclasses.fields(cls):
                    if field.name in (
                        "__key__",
                        "__restore_key__",
                        "__subflavor__",
                        "__subflavors__",
                    ):
                        continue
                    while True:
                        if (
                            field.default is dataclasses.MISSING
                            and field.default_factory is dataclasses.MISSING
                        ):
                            default = ""
                        elif field.default is not dataclasses.MISSING:
                            default = f", default: {field.default}"
                        elif field.default_factory is not dataclasses.MISSING:
                            default = f", default: {field.default_factory!r}"
                        else:
                            raise RuntimeError("This should never happen")
                        field_map[field.name] = input(
                            f"Please enter a webdataset field name for '{field.name}' "
                            f"({field.type}{default}): ",
                        )
                        if not field_map[field.name] and default:
                            del field_map[field.name]
                            break
                        type_ok = True
                        for option in field_map[field.name].split(","):
                            field_name = option.split("[", 1)[0]
                            if field_name not in found_types:
                                click.echo(
                                    "That type doesn't exist in the WebDataset. Please try again."
                                )
                                type_ok = False
                        if type_ok:
                            break
            else:
                if not allow_interactive_field_map:
                    template_part_types = set(["TODO"])
                else:
                    template_part_types = found_types

                if not (path / MAIN_FOLDER_NAME / "sample_loader.py").is_file() or click.confirm(
                    "Do you want to override the existing sample loader?"
                ):
                    with (path / MAIN_FOLDER_NAME / "sample_loader.py").open("w") as f:
                        f.write(
                            sample_loader_template(
                                dataclasses.fields(cls),
                                parts=template_part_types,
                            )
                        )
                    click.echo(
                        f"\nCreated {path / MAIN_FOLDER_NAME / 'sample_loader.py'}. Please edit it to "
                        f"return the proper values."
                    )
                dataset_definition["sample_loader"] = "sample_loader.py:sample_loader"
                dataset_definition["part_filter"] = "sample_loader.py:part_filter"

        # Write the dataset.yaml file
        with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
            yaml.dump(dataset_definition, f, sort_keys=False)
    else:
        click.echo("You will have to add a dataset.yaml manually.")

    click.echo("Done")


if __name__ == "__main__":
    command()
