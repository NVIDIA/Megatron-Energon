# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from megatron.energon.dataclass_slots import dataclass_slots
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
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


def join_multiple_indices(
    meta_infos: List["JoinedDatasetMetaInfo"],
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
        meta_infos: List of meta infos for all datasets.
        output_join_index_path: Path to the output join index.
    """

    assert meta_infos[0].nonmatch in (
        "skip",
        "error",
    ), "Primary dataset must have nonmatch set to 'skip' or 'error'"

    import sqlite3

    # 1. Connect to the primary DB in 'main'
    conn = sqlite3.connect(f"file:{meta_infos[0].db_path!s}?mode=ro", uri=True)

    # For safety, enable a read-only or big timeouts
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA journal_mode = WAL;")

    # 2. Attach each secondary DB under a unique alias, e.g. db1, db2, ...
    secondary_aliases = []
    for i, sec_mi in enumerate(meta_infos[1:], start=1):
        alias = f"db{i}"
        secondary_aliases.append(alias)
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (f"file:{sec_mi.db_path}?mode=ro",))

    # 3. Load tar_files mappings from primary and secondaries into memory
    # The mapping will map from tar_file_id as used in the sqlite DBs to the
    # index as used in the shard_map from the split.
    tar_files_id_mapping = {}
    # Load for primary
    tar_files_id_mapping["main"] = {
        row[0]: meta_infos[0].shard_name_to_idx[row[1]]
        for row in conn.execute("SELECT id, tar_file_name FROM main.tar_files")
        if row[1] in meta_infos[0].shard_name_to_idx
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
    for alias, mi in zip(secondary_aliases, meta_infos[1:]):
        tar_files_id_mapping[alias] = {
            row[0]: mi.shard_name_to_idx[row[1]]
            for row in conn.execute(f"SELECT id, tar_file_name FROM {alias}.tar_files")
            if row[1] in mi.shard_name_to_idx
        }

    select_cols = [
        "main.samples.tar_file_id AS main_tar_file_id",
        "main.samples.byte_offset AS main_byte_offset",
        "main.samples.byte_size AS main_byte_size",
    ]

    for i, alias in enumerate(secondary_aliases, start=1):
        select_cols.append(f"{alias}.samples.tar_file_id AS tar_file_id_{i}")
        select_cols.append(f"{alias}.samples.byte_offset AS byte_offset_{i}")
        select_cols.append(f"{alias}.samples.byte_size AS byte_size_{i}")

    join_clauses = ""
    where_clauses = ""

    # Build excludes for main
    main_excludes = meta_infos[0].excludes
    if main_excludes and len(main_excludes) > 0:
        # Create temporary table for excludes
        conn.execute("DROP TABLE IF EXISTS temp_excludes_main")
        conn.execute(
            """
                CREATE TEMP TABLE temp_excludes_main (
                    sample_key TEXT PRIMARY KEY
                )
            """
        )
        conn.executemany(
            "INSERT INTO temp_excludes_main(sample_key) values (?)", [(e,) for e in main_excludes]
        )
        # Join with the excludes table
        join_clauses += " LEFT JOIN temp_excludes_main ON main.samples.sample_key = temp_excludes_main.sample_key"
        # And exclude those rows which have an exclude key
        where_clauses += " AND temp_excludes_main.sample_key IS NULL"

    # Build the LEFT JOIN clauses
    for alias, mi in zip(secondary_aliases, meta_infos[1:]):
        if mi.nonmatch == "skip":
            join_type = "INNER JOIN"
        else:
            join_type = "LEFT JOIN"
        # LEFT JOIN dbX.samples sX ON main.samples.sample_key = sX.sample_key
        if mi.excludes:
            # Additionally, for excludes, we need to create a temporary table and join with that, and exclude if found
            # Create temporary table for excludes
            conn.execute(f"DROP TABLE IF EXISTS temp_excludes_{alias}")
            conn.execute(
                f"""
                    CREATE TEMP TABLE temp_excludes_{alias} (
                        sample_key TEXT PRIMARY KEY
                    )
                """
            )
            conn.executemany(
                f"INSERT INTO temp_excludes_{alias}(sample_key) values (?)",
                [(e,) for e in mi.excludes],
            )
            # Join with the excludes table and exclude if found
            join_clauses += f" {join_type} {alias}.samples ON main.samples.sample_key = {alias}.samples.sample_key AND temp_excludes_{alias}.sample_key IS NULL"
            join_clauses += f" LEFT JOIN temp_excludes_{alias} ON {alias}.samples.sample_key = temp_excludes_{alias}.sample_key"
        else:
            join_clauses += f" {join_type} {alias}.samples ON main.samples.sample_key = {alias}.samples.sample_key"

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
        {'WHERE ' + where_clauses if where_clauses else ''}
        ORDER BY o.split_index
    """

    # 3. Execute the query; this returns a cursor we can iterate over row by row
    cursor = conn.execute(sql)

    all_db_aliases = ["main"] + secondary_aliases

    # 4. Write the results to a binary file (or any other format) row by row
    with JoinIndexWriter(output_join_index_path) as join_index_writer:
        # Example: We'll just show how to iterate the rows and pseudo-write them
        num_rows = 0
        num_missing = [0] * len(meta_infos)
        for row in cursor:
            # 'row' is a tuple of columns in the order of select_cols

            join_tuples = []
            for i, (alias, meta_info) in enumerate(zip(all_db_aliases, meta_infos)):
                tar_file_id = row[3 * i]

                if tar_file_id is None:
                    # This is missing from the secondary dataset
                    if meta_info.nonmatch == "none":
                        join_tuples.append((-1, -1, -1))
                        num_missing[i] += 1
                    elif meta_info.nonmatch == "skip":
                        # Skip this row. May only happen for the primary dataset, in which case we skip here.
                        assert i == 0, "Should not happen, should already be excluded by the query"
                        break
                    else:
                        raise ValueError(
                            f"Join has encountered a missing sample: Sample key {row[0]} missing from "
                            f"{meta_info.db_path}, although neither nonmatch_none nor nonmatch_skip are set"
                        )
                else:
                    shard_idx = tar_files_id_mapping[alias][tar_file_id]
                    byte_offset = row[3 * i + 1]
                    byte_size = row[3 * i + 2]
                    join_tuples.append((shard_idx, byte_offset, byte_size))
            else:
                # Each row contains (shard_idx, byte_offset, byte_size) for each secondary key.
                join_index_writer.append(*join_tuples)
                num_rows += 1

    # Check that num_rows matches the number of samples in the primary DB
    # It might deviate in case of duplicate samples in a secondary DB
    num_samples = conn.execute(
        "SELECT COUNT(*) FROM main.samples INNER JOIN primary_order o ON main.samples.tar_file_id = o.tar_file_id"
    ).fetchone()[0]
    assert meta_infos[0].nonmatch == "skip" or (
        num_rows == num_samples
    ), f"Number of rows in join index ({num_rows}) does not match number of samples in primary DB ({num_samples})"
    if num_rows != num_samples:
        print(
            f"Joined {num_rows}/{num_samples} samples, skipped {num_samples - num_rows} samples due to join"
        )
    else:
        print(f"Joined all {num_rows} samples")
    if any(num_missing):
        print(f"Samples missing from joins: {num_missing}")

    conn.close()


@dataclass_slots
class JoinedDatasetInfo:
    """Internal for passing the joined datasets."""

    dataset: DatasetLoader

    nonmatch: Literal["skip", "none", "error"]


@dataclass_slots
class JoinedDatasetMetaInfo:
    """Internal for passing the joined datasets."""

    db_path: EPath
    uuid: str
    excludes: List[str]
    shard_name_to_idx: Dict[str, int]
    nonmatch: Literal["skip", "none", "error"]


@dataclass_slots
class JoinDatasetLoader(DatasetLoaderInterface):
    """Loads a joined dataset from a path."""

    datasets: Union[List[JoinedDatasetInfo], Dict[str, JoinedDatasetInfo]]
    joiner: Union[Type[Sample], Callable[..., Sample]]
    cache_path: Optional[EPath] = None

    split_part: Optional[str] = None
    split_config: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1

    def _get_joined_meta(self, split_part: str) -> Tuple[EPath, List[JoinedDatasetMetaInfo]]:
        """
        Collect the metadata for the joined dataset.

        Returns:
            The hashfile path, and a list of the meta infos.
        """
        # Get list of joinable datasets
        datasets = self.datasets
        if isinstance(datasets, dict):
            datasets = list(datasets.values())

        meta_infos: List[JoinedDatasetMetaInfo] = []

        for dataset in datasets:
            print(f" - {dataset}")

            uuid_path = EPath(dataset.dataset.path) / MAIN_FOLDER_NAME / "index.uuid"
            try:
                uuid = uuid_path.read_text()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing uuid file in {uuid_path}. Did you prepare the dataset?"
                )
            db_path = EPath(dataset.dataset.path) / MAIN_FOLDER_NAME / "index.sqlite"

            # Join split_config may override individual split configs
            cur_split_config = self.split_config or dataset.dataset.split_config

            # Precedence for split_part is:
            # 1. Join dataset split part (overrides individual dataset split parts)
            # 2. Individual dataset split part
            # 3. If none of the above is set, use the split part of the surrounding meta dataset
            cur_split_part = self.split_part or dataset.dataset.split_part or split_part
            assert cur_split_part is not None, "Missing split part"

            wds_meta = WebdatasetMeta.from_config(
                path=EPath(dataset.dataset.path),
                split_part=cur_split_part,
                split_config=cur_split_config,
            )

            meta_infos.append(
                JoinedDatasetMetaInfo(
                    db_path=db_path,
                    uuid=uuid,
                    excludes=list(wds_meta.sample_excludes),
                    shard_name_to_idx={
                        shard_name: shard_idx
                        for shard_idx, shard_name in enumerate(wds_meta.split_part_files)
                    },
                    nonmatch=dataset.nonmatch,
                )
            )

        # Combine the hashes into a single hash by xor
        hash = hashlib.sha256()
        for meta_info in meta_infos:
            hash.update(b"\0uuid=")
            hash.update(meta_info.uuid.encode())
            hash.update(b"\0excludes=")
            for exclude in meta_info.excludes:
                hash.update(exclude.encode())
                hash.update(b"\0")
            hash.update(f"\0nonmatch={meta_info.nonmatch}\0".encode())
        assert self.cache_path is not None
        return self.cache_path / f"join_index_{hash.hexdigest()}.bin", meta_infos

    def post_initialize(self, mds_path: Optional[EPath] = None):
        assert mds_path is not None
        self.cache_path = mds_path.parent / f"{mds_path.name}.cache"

    def prepare(self, split_part: Optional[str] = None):
        assert self.cache_path is not None
        assert split_part is not None
        join_index_path, meta_infos = self._get_joined_meta(split_part)
        if join_index_path.is_file():
            print(f"Joined dataset already prepared at {join_index_path}")
            return

        print(f"Preparing joined dataset in {join_index_path}")
        join_index_path.parent.mkdir(parents=True, exist_ok=True)
        join_multiple_indices(
            meta_infos=meta_infos,
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

        join_index_path, _ = self._get_joined_meta(split_part)

        if isinstance(self.datasets, list):
            inner_datasets = [
                dataset.dataset.get_dataset(
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
                key: dataset.dataset.get_dataset(
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
            join_index=join_index_path,
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
