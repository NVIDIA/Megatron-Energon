# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import yaml

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors import (
    BaseCoreDatasetFactory,
    BaseWebdatasetFactory,
    JoinedWebdatasetFactory,
    Sample,
)
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.indexing import JoinIndexWriter
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta
from megatron.energon.flavors.webdataset.structs import WebdatasetInfo
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, DatasetLoaderInterface
from megatron.energon.typed_converter import raw_to_typed
from megatron.energon.worker import WorkerConfig


def join_multiple_indices(
    db_paths: List[EPath],
    shard_maps: List[Dict[str, int]],
    output_join_index_path: EPath,
):
    """
    Joins the 'samples' table of one primary_db with multiple secondary_dbs
    by 'sample_key'. For each secondary DB, we select three columns:
      - tar_file_id
      - byte_offset
      - byte_size
    The result is streamed out row-by-row and written to join index.
    Note that the order of samples is determined by the shard_map of the primary DB.

    Args:
        db_paths: List of paths to primary and secondary databases.
        shard_maps: List of shards dor all datasets. Mapping from shard name to shard index.
        output_join_index_path: Path to the output join index.
    """

    import sqlite3

    primary_db = db_paths[0]
    secondary_dbs = db_paths[1:]

    # 1. Connect to the primary DB in 'main'
    conn = sqlite3.connect(f"file:{primary_db!s}?mode=ro", uri=True)

    # For safety, enable a read-only or big timeouts
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA journal_mode = WAL;")

    # 2. Attach each secondary DB under a unique alias, e.g. db1, db2, ...
    aliases = []
    for i, sec_db in enumerate(secondary_dbs, start=1):
        alias = f"db{i}"
        aliases.append(alias)
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (f"file:{sec_db}?mode=ro",))

    # 3. Load tar_files mappings from primary and secondaries into memory
    # The mapping will map from tar_file_id as used in the sqlite DBs to the
    # index as used in the shard_map from the split.
    tar_files_id_mapping = {}
    # Load for primary
    tar_files_id_mapping["main"] = {
        row[0]: shard_maps[0][row[1]]
        for row in conn.execute("SELECT id, tar_file_name FROM main.tar_files")
        if row[1] in shard_maps[0]
    }

    conn.execute("DROP TABLE IF EXISTS temp_order")
    conn.execute(
        """
        CREATE TEMP TABLE primary_order (
            tar_file_id INTEGER PRIMARY KEY,
            split_index INTEGER
        )
    """
    )
    conn.executemany(
        "INSERT INTO primary_order(tar_file_id, split_index) values (?,?)",
        tar_files_id_mapping["main"].items(),
    )

    # Load for each secondary alias
    for idx, alias in enumerate(aliases):
        tar_files_id_mapping[alias] = {
            row[0]: shard_maps[idx + 1][row[1]]
            for row in conn.execute(f"SELECT id, tar_file_name FROM {alias}.tar_files")
            if row[1] in shard_maps[idx + 1]
        }

    select_cols = [
        "main.samples.tar_file_id AS main_tar_file_id",
        "main.samples.byte_offset AS main_byte_offset",
        "main.samples.byte_size AS main_byte_size",
    ]

    for i, alias in enumerate(aliases, start=1):
        select_cols.append(f"{alias}.samples.tar_file_id AS tar_file_id_{i}")
        select_cols.append(f"{alias}.samples.byte_offset AS byte_offset_{i}")
        select_cols.append(f"{alias}.samples.byte_size AS byte_size_{i}")

    # Build the LEFT JOIN clauses
    join_clauses = ""
    for alias in aliases:
        # LEFT JOIN dbX.samples sX ON main.samples.sample_key = sX.sample_key
        join_clauses += (
            f" LEFT JOIN {alias}.samples ON main.samples.sample_key = {alias}.samples.sample_key"
        )

    # Construct the full SQL query
    # We select three columns for the primary and each secondary DB
    # Those are (tar_file_id, byte_offset, and byte_size)
    # We join the secondary DBs to the primary DB using a LEFT JOIN, i.e.
    # we keep all rows from the primary DB and add columns from the secondary DBs if available
    # Finally, we also join the temporary shard order table to order the shards as in the split config.
    # This join is done using an INNER JOIN, i.e. we only keep rows that have a matching shard index in the primary dataset,
    # so we'll not include shards that come from other split parts
    sql = f"""
        SELECT
            {', '.join(select_cols)}
        FROM main.samples
        {join_clauses}
        INNER JOIN primary_order o
            ON main_tar_file_id = o.tar_file_id
        ORDER BY o.split_index
    """

    # 3. Execute the query; this returns a cursor we can iterate over row by row
    cursor = conn.execute(sql)

    # TODO: We could remove excluded shards and samples here. Or remove exclusion support.

    all_db_aliases = ["main"] + aliases

    # 4. Write the results to a binary file (or any other format) row by row
    with JoinIndexWriter(output_join_index_path) as join_index_writer:
        # Example: We'll just show how to iterate the rows and pseudo-write them
        num_rows = 0
        for row in cursor:
            # 'row' is a tuple of columns in the order of select_cols

            join_tuples = []
            for i in range(len(all_db_aliases)):
                alias = all_db_aliases[i]
                tar_file_id = row[3 * i]

                assert (
                    tar_file_id is not None
                ), f"Left join has encountered a missing sample: Sample key {row[0]} missing in {secondary_dbs[i]}"
                shard_idx = tar_files_id_mapping[alias][tar_file_id]
                byte_offset = row[3 * i + 1]
                byte_size = row[3 * i + 2]
                join_tuples.append((shard_idx, byte_offset, byte_size))

            # Each row contains (shard_idx, byte_offset, byte_size) for each secondary key.
            join_index_writer.append(*join_tuples)
            num_rows += 1

    # Check that num_rows matches the number of samples in the primary DB
    # It might deviate in case of duplicate samples in a secondary DB
    num_samples = conn.execute(
        "SELECT COUNT(*) FROM main.samples INNER JOIN primary_order o ON main.samples.tar_file_id = o.tar_file_id"
    ).fetchone()[0]
    assert (
        num_rows == num_samples
    ), f"Number of rows in join index ({num_rows}) does not match number of samples in primary DB ({num_samples})"

    conn.close()


@dataclass
class JoinDatasetLoader(DatasetLoaderInterface):
    """Loads a joined dataset from a path."""

    datasets: Union[List[DatasetLoader], Dict[str, DatasetLoader]]
    joiner: Union[Type[Sample], Callable[..., Sample]]
    join_method: Literal["inner_match", "inner", "left"] = "inner_match"
    join_index: Optional[EPath] = None

    split_part: Optional[str] = None
    split_config: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1

    def prepare(self, parent_path: EPath, split_part: Optional[str] = None):
        print(f"Preparing joined dataset in {parent_path}")

        # Get list of joinable datasets
        datasets = self.datasets
        if isinstance(datasets, dict):
            datasets = list(datasets.values())

        for dataset in self.datasets:
            print(f" - {dataset}")

        # TODO: Generate sqlite indices for individual join parts if not already present

        db_paths = []
        shard_name_to_idx = []

        for dataset in datasets:
            db_path = EPath(dataset.path) / MAIN_FOLDER_NAME / "index.sqlite"

            # Join split_config may override individual split configs
            cur_split_config = self.split_config or dataset.split_config

            # Precedence for split_part is:
            # 1. Join dataset split part (overrides individual dataset split parts)
            # 2. Individual dataset split part
            # 3. If none of the above is set, use the split part of the surrounding meta dataset
            cur_split_part = self.split_part or dataset.split_part or split_part

            wds_meta = WebdatasetMeta.from_config(
                path=EPath(dataset.path), split_part=cur_split_part, split_config=cur_split_config
            )

            db_paths.append(db_path)
            shard_name_to_idx.append(
                {
                    shard_name: shard_idx
                    for shard_idx, shard_name in enumerate(wds_meta.split_part_files)
                }
            )

        join_index_path = parent_path / f"join_index_{split_part}.bin"

        join_multiple_indices(
            db_paths=db_paths,
            shard_maps=shard_name_to_idx,
            output_join_index_path=join_index_path,
        )

    def get_dataset(
        self,
        *,
        training: bool,
        split_part: Optional[str] = None,
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs: int = 1,
        split_config: Optional[str] = None,
        **kwargs,
    ) -> BaseCoreDatasetFactory:
        """
        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration.
            shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
            subflavor: Subflavor to use, might be overridden by inner datasets.
            subflavors: Subflavors to use, might be overridden by inner datasets.
            shuffle_over_epochs: Shuffle the dataset over this many epochs.
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The loaded dataset
        """
        if self.split_config is not None:
            split_config = self.split_config
        if self.split_part is not None:
            split_part = self.split_part
        if split_part is None:
            raise ValueError("Missing split part")
        if subflavor is None:
            subflavor = self.subflavor
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        if isinstance(self.datasets, list):
            inner_datasets = [
                dataset.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs,
                    split_config=split_config,
                    **kwargs,
                )
                for dataset in self.datasets
            ]
            assert all(
                isinstance(d, BaseWebdatasetFactory) for d in inner_datasets
            ), "Can only merge webdatasets efficiently"
        elif isinstance(self.datasets, dict):
            inner_datasets = {
                key: dataset.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs,
                    split_config=split_config,
                    **kwargs,
                )
                for key, dataset in self.datasets.items()
            }
            assert all(
                isinstance(d, BaseWebdatasetFactory) for d in inner_datasets.values()
            ), "Can only merge webdatasets efficiently"
        else:
            raise ValueError("Invalid join type")
        return JoinedWebdatasetFactory(
            inner_datasets=inner_datasets,
            training=training,
            worker_config=worker_config,
            shuffle_over_epochs=shuffle_over_epochs,
            join_method=self.join_method,
            join_index=self.join_index,
            joiner=self.joiner,
            **kwargs,
        )

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> Tuple[DatasetBlendMode, List[Tuple[BaseCoreDatasetFactory, Union[float, int, None]]]]:
        return DatasetBlendMode.NONE, [
            (
                self.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs_multiplier,
                    **kwargs,
                ),
                None,
            )
        ]
