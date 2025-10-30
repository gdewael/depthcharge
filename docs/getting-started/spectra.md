---
jupyter: python3
toc-title: Table of contents
---

# Modeling Mass Spectra

We think that the biggest barriers to building deep learning models for
mass spectra are (1) parsing the data into a reasonable format, (2)
representing the mass spectra in an efficient manner for learning, (3)
constructing models that leverage the structure of the data in a mass
spectrum.

## Parsing Mass Spectra

Depthcharge supports reading mass spectra from a variety of open
formats: mzML, mzXML, and MGF.[^1] Under the hood, Depthcharge uses the
Apache Arrow data format to store mass spectrometry data in a tabular
manner, and likewise Apache Parquet to store individual mass
spectrometry data files on disk. This means that standard data science
tools like Pandas or Polars can be used to interact with our mass
spectra after it is parsed by Depthcharge.

For example, we can read an mzML file into a `polars.DataFrame` using
`spectra_to_df()`:

:::: {.cell execution_count="2"}
``` {.python .cell-code}
import polars as pl
import depthcharge as dc

mzml_file = "../../data/TMT10-Trial-8.mzML"

# Read an mzML into a DataFramoe:
df = dc.data.spectra_to_df(mzml_file, progress=False)
print(df.head())
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 7)
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
    | peak_file    | scan_id      | ms_level | precursor_mz | precursor_ch | mz_array    | intensity_a |
    | ---          | ---          | ---      | ---          | arge         | ---         | rray        |
    | str          | str          | u8       | f64          | ---          | list[f64]   | ---         |
    |              |              |          |              | i16          |             | list[f64]   |
    +==================================================================================================+
    | TMT10-Trial- | controllerTy | 2        | 804.774963   | 3            | [284.917023 | [0.004933,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.005314, … |
    |              | lerNum…      |          |              |              | 312.908997, | 0.00807…    |
    |              |              |          |              |              | … 157…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1001.6693    | 2            | [311.863007 | [0.008258,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.008424, … |
    |              | lerNum…      |          |              |              | 330.019012, | 0.00644…    |
    |              |              |          |              |              | … 173…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1047.6174    | 3            | [338.669006 | [0.011869,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.010293, … |
    |              | lerNum…      |          |              |              | 354.175995, | 0.01242…    |
    |              |              |          |              |              | … 185…      |             |
    | TMT10-Trial- | controllerTy | 2        | 800.4349     | 3            | [247.792007 | [0.005207,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.004437, … |
    |              | lerNum…      |          |              |              | 313.678009, | 0.00915…    |
    |              |              |          |              |              | … 194…      |             |
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
:::
::::

We can write the mass spectra directly to a Parquet file:

:::: {.cell execution_count="3"}
``` {.python .cell-code}
pq_file = dc.data.spectra_to_parquet(mzml_file, progress=False)
print(pl.read_parquet(pq_file).head())
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 7)
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
    | peak_file    | scan_id      | ms_level | precursor_mz | precursor_ch | mz_array    | intensity_a |
    | ---          | ---          | ---      | ---          | arge         | ---         | rray        |
    | str          | str          | u8       | f64          | ---          | list[f64]   | ---         |
    |              |              |          |              | i16          |             | list[f64]   |
    +==================================================================================================+
    | TMT10-Trial- | controllerTy | 2        | 804.774963   | 3            | [284.917023 | [0.004933,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.005314, … |
    |              | lerNum…      |          |              |              | 312.908997, | 0.00807…    |
    |              |              |          |              |              | … 157…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1001.6693    | 2            | [311.863007 | [0.008258,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.008424, … |
    |              | lerNum…      |          |              |              | 330.019012, | 0.00644…    |
    |              |              |          |              |              | … 173…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1047.6174    | 3            | [338.669006 | [0.011869,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.010293, … |
    |              | lerNum…      |          |              |              | 354.175995, | 0.01242…    |
    |              |              |          |              |              | … 185…      |             |
    | TMT10-Trial- | controllerTy | 2        | 800.4349     | 3            | [247.792007 | [0.005207,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.004437, … |
    |              | lerNum…      |          |              |              | 313.678009, | 0.00915…    |
    |              |              |          |              |              | … 194…      |             |
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
:::
::::

Or we can stream them from the original file in batches:

:::: {.cell execution_count="4"}
``` {.python .cell-code}
batch = next(dc.data.spectra_to_stream(mzml_file, progress=False))
print(pl.from_arrow(batch))
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 7)
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
    | peak_file    | scan_id      | ms_level | precursor_mz | precursor_ch | mz_array    | intensity_a |
    | ---          | ---          | ---      | ---          | arge         | ---         | rray        |
    | str          | str          | u8       | f64          | ---          | list[f64]   | ---         |
    |              |              |          |              | i16          |             | list[f64]   |
    +==================================================================================================+
    | TMT10-Trial- | controllerTy | 2        | 804.774963   | 3            | [284.917023 | [0.004933,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.005314, … |
    |              | lerNum…      |          |              |              | 312.908997, | 0.00807…    |
    |              |              |          |              |              | … 157…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1001.6693    | 2            | [311.863007 | [0.008258,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.008424, … |
    |              | lerNum…      |          |              |              | 330.019012, | 0.00644…    |
    |              |              |          |              |              | … 173…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1047.6174    | 3            | [338.669006 | [0.011869,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.010293, … |
    |              | lerNum…      |          |              |              | 354.175995, | 0.01242…    |
    |              |              |          |              |              | … 185…      |             |
    | TMT10-Trial- | controllerTy | 2        | 800.4349     | 3            | [247.792007 | [0.005207,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | 0.004437, … |
    |              | lerNum…      |          |              |              | 313.678009, | 0.00915…    |
    |              |              |          |              |              | … 194…      |             |
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
:::
::::

### Spectrum Preprocessing

Preprocessing steps, such as filtering peaks and transforming
intensities is performed during parsing using and controlled using the
`preprocessing_fn` parameter in all of the above functions. Depthcharge
is closely tied into the [spectrum_utils
package](https://spectrum-utils.readthedocs.io/en/latest/) and any of
the processing methods within spectrum_utils may be applied to spectra
in Depthcharge as well. Additionally, custom functions can be specified
that accept a `depthcharge.MassSpectrum` as input and return a
`depthcharge.MassSpectrum` as output.

The `preprocessing_fn` parameter defines collection of functions to
apply to each mass spectrum in sequence. The default `preprocessing_fn`
is:

::: {.cell execution_count="5"}
``` {.python .cell-code}
[
    dc.data.preprocessing.set_mz_range(min_mz=140),
    dc.data.preprocessing.filter_intensity(max_num_peaks=200),
    dc.data.preprocessing.scale_intensity(scaling="root"),
    dc.data.preprocessing.scale_to_unit_norm,
]
```
:::

However, we can change this process to meet our needs. As an example,
let's create rather useless preprocessing function that sets the
intensity of all the peaks to a value of one:

::: {.cell execution_count="6"}
``` {.python .cell-code}
import numpy as np

def scale_to_one(spec):
    """Set intensities to one.

    Parameters
    ----------
    spec : depthcharge.MassSpectrum
        The mass spectrum to transform.

    Returns
    -------
    depthcharge.MassSpectrum
        The transformed mass spectrum.
    """
    spec.intensity = np.ones_like(spec.intensity)
    return spec
```
:::

We can then use our preprocessing function, either by itself or in
combination with other functions:

:::: {.cell execution_count="7"}
``` {.python .cell-code}
df = dc.data.spectra_to_df(
    mzml_file,
    progress=False,
    preprocessing_fn=[scale_to_one]
)
print(df.head())
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 7)
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
    | peak_file    | scan_id      | ms_level | precursor_mz | precursor_ch | mz_array    | intensity_a |
    | ---          | ---          | ---      | ---          | arge         | ---         | rray        |
    | str          | str          | u8       | f64          | ---          | list[f64]   | ---         |
    |              |              |          |              | i16          |             | list[f64]   |
    +==================================================================================================+
    | TMT10-Trial- | controllerTy | 2        | 804.774963   | 3            | [284.917023 | [1.0, 1.0,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | … 1.0]      |
    |              | lerNum…      |          |              |              | 312.908997, |             |
    |              |              |          |              |              | … 157…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1001.6693    | 2            | [311.863007 | [1.0, 1.0,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | … 1.0]      |
    |              | lerNum…      |          |              |              | 330.019012, |             |
    |              |              |          |              |              | … 173…      |             |
    | TMT10-Trial- | controllerTy | 2        | 1047.6174    | 3            | [338.669006 | [1.0, 1.0,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | … 1.0]      |
    |              | lerNum…      |          |              |              | 354.175995, |             |
    |              |              |          |              |              | … 185…      |             |
    | TMT10-Trial- | controllerTy | 2        | 800.4349     | 3            | [247.792007 | [1.0, 1.0,  |
    | 8.mzML       | pe=0 control |          |              |              | ,           | … 1.0]      |
    |              | lerNum…      |          |              |              | 313.678009, |             |
    |              |              |          |              |              | … 194…      |             |
    +--------------+--------------+----------+--------------+--------------+-------------+-------------+
:::
::::

### Extracting Additional Data

By default, Depthcharge only extracts a minimal amount of information
from a mass spectrometry data file. Additional fields can be retrieved
using the `custom_fields` parameter in the above parsing functions.
However, we have to tell Depthcharge exactly what data we want to
extract and how to extract it.

Currently, all mass spectrometry data file parsing is handled using the
corresponding parser from
[Pyteomics](https://pyteomics.readthedocs.io/en/latest/), which yield a
Python dictionary for each spectrum. The function we define to extract
data must operate on this spectrum dictionary. Below, we define a custom
field to extract the retention time for each spectrum:

:::: {.cell execution_count="8"}
``` {.python .cell-code}
import pyarrow as pa

ce_field = dc.data.CustomField(
    # The new column name:
    name="ret_time",
    # The function to extract the retention time:
    accessor=lambda x: x["scanList"]["scan"][0]["scan start time"],
    # The expected data type:
    dtype=pa.float64(),
)

df = dc.data.spectra_to_df(
    mzml_file,
    progress=False,
    custom_fields=ce_field,
)
print(df.head())
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 8)
    +------------+------------+----------+------------+------------+------------+-----------+----------+
    | peak_file  | scan_id    | ms_level | precursor_ | precursor_ | mz_array   | intensity | ret_time |
    | ---        | ---        | ---      | mz         | charge     | ---        | _array    | ---      |
    | str        | str        | u8       | ---        | ---        | list[f64]  | ---       | f64      |
    |            |            |          | f64        | i16        |            | list[f64] |          |
    +==================================================================================================+
    | TMT10-Tria | controller | 2        | 804.774963 | 3          | [284.91702 | [0.004933 | 1.069915 |
    | l-8.mzML   | Type=0 con |          |            |            | 3, 312.908 | ,         |          |
    |            | trollerNum |          |            |            | 997, …     | 0.005314, |          |
    |            | …          |          |            |            | 157…       | …         |          |
    |            |            |          |            |            |            | 0.00807…  |          |
    | TMT10-Tria | controller | 2        | 1001.6693  | 2          | [311.86300 | [0.008258 | 1.076773 |
    | l-8.mzML   | Type=0 con |          |            |            | 7, 330.019 | ,         |          |
    |            | trollerNum |          |            |            | 012, …     | 0.008424, |          |
    |            | …          |          |            |            | 173…       | …         |          |
    |            |            |          |            |            |            | 0.00644…  |          |
    | TMT10-Tria | controller | 2        | 1047.6174  | 3          | [338.66900 | [0.011869 | 1.083688 |
    | l-8.mzML   | Type=0 con |          |            |            | 6, 354.175 | ,         |          |
    |            | trollerNum |          |            |            | 995, …     | 0.010293, |          |
    |            | …          |          |            |            | 185…       | …         |          |
    |            |            |          |            |            |            | 0.01242…  |          |
    | TMT10-Tria | controller | 2        | 800.4349   | 3          | [247.79200 | [0.005207 | 1.09051  |
    | l-8.mzML   | Type=0 con |          |            |            | 7, 313.678 | ,         |          |
    |            | trollerNum |          |            |            | 009, …     | 0.004437, |          |
    |            | …          |          |            |            | 194…       | …         |          |
    |            |            |          |            |            |            | 0.00915…  |          |
    +------------+------------+----------+------------+------------+------------+-----------+----------+
:::
::::

### Adding Additional Outside Data

As a DataFrame or Parquet file, the parsed mass spectra are relatively
easy to manipulate uses standard data science tools like Polars and
Pandas. However, we can also efficiently add new data to our mass
spectra during parsing by providing a separate metadata dataframe as the
`metadata_df` parameter. We require that this dataframe have a `scan_id`
field and it may optionally have a `peak_file` field that will be used
to join the metadata table with the parsed spectra.

For example, we could use a metadata_df to pair peptide detections with
the spectrum that they were detected from:

::::: {.cell execution_count="9"}
``` {.python .cell-code}
metadata_df = pl.DataFrame({
    "scan_id": [
        f"controllerType=0 controllerNumber=1 scan={x}"
        for x in (501, 507)
    ],
    "peptide": ["LESLIEK", "EDITHR"]
})

df = dc.data.spectra_to_df(
    mzml_file,
    progress=False,
    metadata_df=metadata_df,
)
print(df.head())
```

::: {.cell-output .cell-output-stdout}
    shape: (4, 8)
    +------------+------------+----------+------------+------------+------------+------------+---------+
    | peak_file  | scan_id    | ms_level | precursor_ | precursor_ | mz_array   | intensity_ | peptide |
    | ---        | ---        | ---      | mz         | charge     | ---        | array      | ---     |
    | str        | str        | u8       | ---        | ---        | list[f64]  | ---        | str     |
    |            |            |          | f64        | i16        |            | list[f64]  |         |
    +==================================================================================================+
    | TMT10-Tria | controller | 2        | 804.774963 | 3          | [284.91702 | [0.004933, | LESLIEK |
    | l-8.mzML   | Type=0 con |          |            |            | 3, 312.908 | 0.005314,  |         |
    |            | trollerNum |          |            |            | 997, …     | … 0.00807… |         |
    |            | …          |          |            |            | 157…       |            |         |
    | TMT10-Tria | controller | 2        | 1001.6693  | 2          | [311.86300 | [0.008258, | null    |
    | l-8.mzML   | Type=0 con |          |            |            | 7, 330.019 | 0.008424,  |         |
    |            | trollerNum |          |            |            | 012, …     | … 0.00644… |         |
    |            | …          |          |            |            | 173…       |            |         |
    | TMT10-Tria | controller | 2        | 1047.6174  | 3          | [338.66900 | [0.011869, | EDITHR  |
    | l-8.mzML   | Type=0 con |          |            |            | 6, 354.175 | 0.010293,  |         |
    |            | trollerNum |          |            |            | 995, …     | … 0.01242… |         |
    |            | …          |          |            |            | 185…       |            |         |
    | TMT10-Tria | controller | 2        | 800.4349   | 3          | [247.79200 | [0.005207, | null    |
    | l-8.mzML   | Type=0 con |          |            |            | 7, 313.678 | 0.004437,  |         |
    |            | trollerNum |          |            |            | 009, …     | … 0.00915… |         |
    |            | …          |          |            |            | 194…       |            |         |
    +------------+------------+----------+------------+------------+------------+------------+---------+
:::

::: {.cell-output .cell-output-stderr}
    /Users/wfondrie/src/depthcharge/depthcharge/data/arrow.py:101: PerformanceWarning:

    Determining the column names of a LazyFrame requires resolving its schema, which is a potentially expensive operation. Use `LazyFrame.collect_schema().names()` to get the column names without this warning.
:::
:::::

## Building PyTorch Datasets from Mass Spectra

Although the individual mass spectrometry data file parsing features are
nice, often we will want to train models on more than one file and at a
scale that is unlikely to fit in memory. For this task, Depthcharge
provides three dataset classes for mass spectra:

-   `SpectrumDataset` - Use this class for training on mass spectra.
-   `AnnotatedSpectrumDataset` - Use this class for training on
    annotated mass spectra, such as peptide-spectrum matches.
-   `StreamingSpectrumDataset` - Use this class for running inference on
    mass spectra.

The `SpectrumDataset` and `AnnotatedSpectrumDataset` classes parse
spectra into a [Lance
dataset](https://lancedb.github.io/lance/index.html#) which allows for
efficient on-disk storage and fast random access of the stored spectra.

All of these classes can be created from the same mass spectrometry data
formats as above, or can be created from previously parsed mass spectra
as dataframes or Parquet files. Furthermore, when doing the former, all
of the same features for preprocessing and adding additional data are
available using the `parse_kwargs` parameter.

For example, we can create a dataset for our example file:

::: {.cell execution_count="10"}
``` {.python .cell-code}
from depthcharge.data import SpectrumDataset

parse_kwargs = {"progress": False}

dataset = SpectrumDataset(mzml_file, batch_size=2, parse_kwargs=parse_kwargs)
```
:::

The `SpectrumDataset` and `AnnoatatedSpectrumDataset` use the native
[PyTorch integration in
Lance](https://lancedb.github.io/lance/integrations/pytorch.html) and
all of the corresponding parameters are available as keyword arguments.
Furthermore, all three of these classes are [PyTorch `IterableDataset`
classes](https://pytorch.org/docs/stable/data.html#iterable-style-datasets),
so they are ready to be used directly to train and evaulate deep
learning models with PyTorch.

**A word of caution:** the batch size and any parallelism a best handled
by the dataset class itself rather than the PyTorch `DataLoader`; hence,
we recommend initializing `DataLoaders` with `num_workers` \<= 1 and
`batch_size` = 1, or simply:

:::: {.cell execution_count="11"}
``` {.python .cell-code}
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=None)

for batch in loader:
    print(batch["scan_id"], batch["precursor_mz"])
```

::: {.cell-output .cell-output-stdout}
    ['controllerType=0 controllerNumber=1 scan=501', 'controllerType=0 controllerNumber=1 scan=504'] tensor([ 804.7750, 1001.6693])
    ['controllerType=0 controllerNumber=1 scan=507', 'controllerType=0 controllerNumber=1 scan=510'] tensor([1047.6174,  800.4349])
:::
::::

## Transfomer Models for Mass Spectra

Now that we know how to parse mass spectra, we can now build a model
that uses them. In Depthcharge, we've designed Transformer models
specifically for this task: the `SpectrumTransformerEncoder`. However,
the dataset classes and other modules in Depthcharge are fully
interoperable with any PyTorch module.

:::: {.cell execution_count="12"}
``` {.python .cell-code}
from depthcharge.transformers import SpectrumTransformerEncoder

model = SpectrumTransformerEncoder()

for batch in loader:
    out = model(batch["mz_array"], batch["intensity_array"])
    print(out[0].shape)
```

::: {.cell-output .cell-output-stdout}
    torch.Size([2, 119, 128])
    torch.Size([2, 108, 128])
:::
::::

Note that by default, our Transformer model only considers the spectrum
itself and not any of the precursor information. However, we can add it!

The first element output by each Transformer module in Depthcharge is a
global representation of the sequence, which is a mass spectrum in this
case. By default, it is set to `0`s and ignored. We can change this
behavior by creating a child class of our Transformer module and
overriding the `global_token_hook` method. Let's create a hook that will
add information about the precursor mass and charge to the global
representation:

::: {.cell execution_count="13"}
``` {.python .cell-code}
from depthcharge.encoders import FloatEncoder

class MyModel(SpectrumTransformerEncoder):
    """Our custom model class."""
    def __init__(self, *args, **kwargs):
        """Add parameters for the global token hook."""
        super().__init__(*args, **kwargs)
        self.precursor_mz_encoder = FloatEncoder(self.d_model)
        self.precursor_z_encoder = FloatEncoder(self.d_model, 1, 10)

    def global_token_hook(self, mz_array, intensity_array, *args, **kwargs):
        """Return a simple representation of the precursor."""
        mz_rep = self.precursor_mz_encoder(
            kwargs["precursor_mz"].type_as(mz_array)[None, :],
        )
        charge_rep = self.precursor_z_encoder(
            kwargs["precursor_charge"].type_as(mz_array)[None, :],
        )
        return (mz_rep + charge_rep)[0, :]
```
:::

Now we can use our new `MyModel` class with our mass spectra:

:::: {.cell execution_count="14"}
``` {.python .cell-code}
model = MyModel()

for batch in loader:
    out = model(**batch)
    print(out[0].shape)
```

::: {.cell-output .cell-output-stdout}
    torch.Size([2, 119, 128])
    torch.Size([2, 108, 128])
:::
::::

These Depthcharge modules are merely an accessible starting point for us
to build a model fully customized to our task at hand.

[^1]: We plan to support the Bruker .d format will be supported through
    timsrust.
