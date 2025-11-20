# Programmatic Data Preparation

Below, we provide an example function that shows how to prepare a dataset programmatically.

```python
from megatron.energon.flavors import BaseWebdatasetFactory

def prepare_one_dataset(path: Path, num_workers: int, template_dir: Path):

    if (path / ".nv-meta" / "dataset.yaml").exists():
        print(f"Dataset {path} already prepared. Skipping.")
        return

    # Fixed settings
    tar_index_only = False
    split_parts_ratio = [("train", 1), ("val", 0), ("test", 0)]
    split_parts_patterns = None
    
    # Get all tar files
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    if len(all_tars) == 0:
        print("Did not find any tar files. Exiting.")
        return

    print(f"Found {len(all_tars)} tar files in total. The first and last ones are:")
    print(f"- {all_tars[0]}")
    print(f"- {all_tars[-1]}")

    def progress_fn(els, length=None):
        with click.progressbar(
            els,
            label="Indexing shards",
            show_pos=True,
            length=length,
        ) as bar:
            for el in bar:
                yield el

    found_types, duplicates = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        progress_fn=progress_fn,
        tar_index_only=tar_index_only,
        shuffle_seed=None,
        workers=num_workers,
    )

    # Copy sample loader and dataset.yaml templates
    for file in template_dir.glob("*"):
        shutil.copy(file, path / ".nv-meta" / file.name)
```

Example usage:

First, create a template directory with the `dataset.yaml` file, and optionally the `sample_loader.py` file.
Let's call it `template_dir`.

Then, run the script:

```python
if __name__ == "__main__":
    prepare_one_dataset(Path("/path/to/dataset"), 16, Path("/path/to/template_dir"))
```
