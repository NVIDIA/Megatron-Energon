<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(recipe-checkpoint)=

# Recipe Graphs

Recipes are declarative graphs. Runtime datasets are wrapper trees built from the leaves of these graphs.

## Recipe graph construction

The main graph nodes are:

- `DatasetReference`, a leaf path plus split, subset, shuffle, and subflavor settings;
- `RecipeBlend` and `RecipeBlendEpochized`, weighted child graphs;
- `RecipeJoin`, a primary dataset plus joined auxiliary sources;
- `Subset`, which expresses relative or absolute selection;
- `Recipe`, which owns named splits and resolves the full graph.

`Recipe.post_initialize` resolves references relative to the recipe root and validates the resulting graph.
`get_datasets` builds runtime factories. `traverse` visits the same logical leaves without constructing the
runtime pipeline; use it for preparation, diagnostics, and graph inspection.

## Nested option composition

When a reference wraps another recipe or reference, its settings compose with the inner settings:

- positive shuffle-over-epochs multipliers multiply;
- a shuffle value of `-1` dominates positive multipliers;
- `None` disables the composed shuffle and therefore dominates all numeric values;
- relative subsets compose through the nesting;
- absolute subset bounds belong at the concrete leaf, where the absolute sample space is known;
- inner subflavors are retained, with outer values overriding the same keys;
- an outer split selection may choose a different split of a nested recipe.

Keep these rules in one graph-resolution layer. Applying a subset a second time in the reader or moving
shuffle composition into a wrapper changes leaf identity or iteration order.

## Developing graph changes

When adding or changing a recipe node:

1. define how it resolves nested paths, splits, subsets, shuffle values, and subflavors;
2. ensure `traverse` and runtime construction find the same leaves;
3. decide whether child order is semantic;
4. classify any state-layout or order change using {ref}`compatibility`.

The recipe tests, including save/restore, are in `test_recipe.py`.
