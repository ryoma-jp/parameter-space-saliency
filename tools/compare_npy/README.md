# compare_npy

This is a small utility for comparing two `.npy` files and writing the result to an output directory.

## Features

- Load and compare two `.npy` files
- Write `Comparison result: Match` to `result.txt` when the files match
- Write `Comparison result: Mismatch` to `result.txt` and a sparse diff to `diff.csv` when they do not match
- `diff.csv` contains only the mismatched elements, so the number of rows scales with the number of differences rather than the total number of elements
- Allow the output directory to be specified via a command-line argument

The comparison checks the following:

- Array shape
- Dtype
- Array values

For floating-point arrays, `NaN` values are treated as equal to each other.

## Creating Sample `.npy` Files

You can create small sample files with NumPy before running the tool.

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

output_dir = Path("tools/compare_npy/sample_data")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "sample_a.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
np.save(output_dir / "sample_b_match.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
np.save(output_dir / "sample_c_diff.npy", np.array([[1.0, 2.5], [3.0, 4.0]], dtype=np.float32))
PY
```

This creates the following files:

- `tools/compare_npy/sample_data/sample_a.npy`
- `tools/compare_npy/sample_data/sample_b_match.npy`
- `tools/compare_npy/sample_data/sample_c_diff.npy`

You can use `sample_a.npy` and `sample_b_match.npy` for a matching case, and `sample_a.npy` and `sample_c_diff.npy` for a mismatching case.

## Usage

```bash
python3 tools/compare_npy/compare_npy.py <source.npy> <target.npy> --output <output_dir>
```

Example:

```bash
python3 tools/compare_npy/compare_npy.py \
	results/a.npy \
	results/b.npy \
	--output results/compare_report
```

This creates the following files:

- `results/compare_report/result.txt` — comparison result
- `results/compare_report/diff.csv` — mismatched elements only (generated only when shape and dtype match but values differ)

## Output Example

`result.txt` when the arrays match:

```text
source: results/a.npy
target: results/b.npy
source shape: (2, 2)  dtype: float32
target shape: (2, 2)  dtype: float32
Comparison result: Match
```

`result.txt` when the arrays do not match:

```text
source: results/a.npy
target: results/c.npy
source shape: (2, 2)  dtype: float32
target shape: (2, 2)  dtype: float32
Comparison result: Mismatch
```

`diff.csv` (only rows where values differ):

```text
index,source_value,target_value,abs_diff
"(0, 1)",2.0,2.5,0.5
```

## Notes

- `diff.csv` contains only mismatched elements; rows with equal values are omitted
- If shape or dtype differs, element-wise comparison is not performed and `diff.csv` is not generated
- The output directory is created automatically if it does not already exist
