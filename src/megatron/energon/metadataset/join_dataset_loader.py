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

    Args:
        primary_db: Path to the primary database.
        secondary_dbs: List of paths to secondary databases.
        secondary_shard_maps: List of mappings from tar_file name to tar_file_index
            for each secondary DB. The order should be as in .info.yaml.
        output_join_index_path: Path to the output join index.
    """

    import sqlite3

    primary_db = db_paths[0]
    secondary_dbs = db_paths[1:]

    # 1. Connect to the primary DB in 'main'
    conn = sqlite3.connect(str(primary_db))

    # For safety, enable a read-only or big timeouts
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA journal_mode = WAL;")

    # 2. Attach each secondary DB under a unique alias, e.g. db1, db2, ...
    aliases = []
    for i, sec_db in enumerate(secondary_dbs, start=1):
        alias = f"db{i}"
        aliases.append(alias)
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(sec_db),))

    # 3. Load tar_files mappings from primary and secondaries into memory
    # The mapping will map from tar_file_id as used in the sqlite DBs to the
    # index as used in the .info.yaml files.
    tar_files_id_mapping = {}
    # Load for primary
    tar_files_id_mapping["main"] = {
        row[0]: shard_maps[0][row[1]]
        for row in conn.execute("SELECT id, tar_file_name FROM main.tar_files")
    }
    # Load for each secondary alias
    for idx, alias in enumerate(aliases):
        tar_files_id_mapping[alias] = {
            row[0]: shard_maps[idx + 1][row[1]]
            for row in conn.execute(f"SELECT id, tar_file_name FROM {alias}.tar_files")
        }

    # Build the SELECT list:
    #   - For columns from the primary, let's pick what you want (sample_key, tar_file_id, etc).
    #   - For each attached DB alias, add "alias.samples.tar_file_id AS tar_file_id_i, alias.samples.byte_offset AS byte_offset_i, alias.samples.byte_size AS byte_size_i"
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
    sql = f"""
        SELECT
            {', '.join(select_cols)}
        FROM main.samples
        {join_clauses}
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
    num_samples = conn.execute("SELECT COUNT(*) FROM main.samples").fetchone()[0]
    assert (
        num_rows == num_samples
    ), f"Number of rows in join index ({num_rows}) does not match number of samples in primary DB ({num_samples})"

    # 5. Detach databases and close
    for alias in aliases:
        conn.execute(f"DETACH DATABASE {alias}")

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
        print(f"Preparing {self.__class__.__name__}")

        # Get list of joinable datasets
        datasets = self.datasets
        if isinstance(datasets, dict):
            datasets = list(datasets.values())

        for dataset in self.datasets:
            print(f" - {dataset}")

        # TODO: Generate sqlite indices for individual join parts if not already present

        # TODO: Iterate primary DS in shard order from split config. This is currently not the case!

        db_paths = []
        shard_name_to_idx = []

        for dataset in datasets:
            db_path = dataset.path / MAIN_FOLDER_NAME / "index.sqlite"

            # info = raw_to_typed(
            #     yaml.safe_load(info_path.read_text()),
            #     WebdatasetInfo,
            # )

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

            shard_names = [shard_info.name for shard_info in wds_meta.shards]

            db_paths.append(db_path)
            shard_name_to_idx.append(
                {shard_name: shard_idx for shard_idx, shard_name in enumerate(shard_names)}
            )

        # TODO: We might have multiple joins in one metadataset (e.g. for train and val, or not?)
        # TODO: Keep multiple join indices

        join_index_path = parent_path / "join_index.bin"

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
