"""
Convert Kerchunk references to Zarr
"""

import typer
from pathlib import Path
import xarray as xr
from xarray import Dataset
import zarr
# from zarr.codecs import BloscCodec
from rich import print
from dask.distributed import LocalCluster, Client
from zarr.storage import LocalStore  # for Zarr 3
from typing_extensions import Annotated, Optional, List

from rekx.dask_configuration import auto_configure_for_large_dataset
from .typer_parameters import (
    typer_argument_time_series,
    typer_argument_variable,
    typer_option_verbose,
    typer_option_dry_run,
    typer_option_tolerance,
)
from .constants import VERBOSE_LEVEL_DEFAULT
from .drop import drop_other_data_variables

DASK_COMPUTE = True
ZARR_STORE_BASE_PATH = Path("sis_italia")
ZARR_CONSOLIDATE_DEFAULT = False
ZARR_COMPRESSOR_CODEC = "zstd"
COMPRESSION_FILTER_DEFAULT = ZARR_COMPRESSOR_CODEC
ZARR_COMPRESSOR_LEVEL = 1
COMPRESSION_LEVEL_DEFAULT = ZARR_COMPRESSOR_LEVEL
ZARR_COMPRESSOR_SHUFFLE = "shuffle"
SHUFFLING_DEFAULT = ZARR_COMPRESSOR_SHUFFLE
ZARR_COMPRESSOR = zarr.codecs.BloscCodec(
    cname=ZARR_COMPRESSOR_CODEC,
    clevel=ZARR_COMPRESSOR_LEVEL,
    shuffle=ZARR_COMPRESSOR_SHUFFLE,
)
DATASET_SELECT_TOLERANCE_DEFAULT = 0.1
GREEN_DASH = f"[green]-[/green]"


# Add this before generating the Zarr store
def apply_safe_chunking(dataset, main_variable, time_chunks, lat_chunks, lon_chunks):
    for var in dataset.variables:
        # Skip primary variable (handled separately)
        if var == main_variable:
            continue
            
        # Clear problematic encoding
        if 'chunks' in dataset[var].encoding:
            del dataset[var].encoding['chunks']
        if 'compressor' in dataset[var].encoding:
            del dataset[var].encoding['compressor']
            
        # Apply safe chunking to auxiliary variables
        if var in ['lat', 'lon']:
            dataset[var] = dataset[var].chunk({
                'time': time_chunks,
                'lat': lat_chunks,
                'lon': lon_chunks,
                # 'bnds': 2  # Explicit chunk for bounds dimension
            })
    
    return dataset


def read_parquet_via_zarr(
    time_series: Annotated[Path, typer_argument_time_series],
    variable: Annotated[str, typer_argument_variable],
    # longitude: Annotated[float, typer_argument_longitude_in_degrees],
    # latitude: Annotated[float, typer_argument_latitude_in_degrees],
    # window: Annotated[int, typer_option_spatial_window_in_degrees] = None,
    tolerance: Annotated[
        Optional[float], typer_option_tolerance
    ] = DATASET_SELECT_TOLERANCE_DEFAULT,
) -> None:
    """
    Read a time series data file via Xarray's Zarr engine
    format.

    Parameters
    ----------
    time_series:
        Path to Xarray-supported input file
    variable: str
        Name of the variable to query
    longitude: float
        The longitude of the location to read data
    latitude: float
        The latitude of the location to read data
    # window:
    tolerance: float
        Maximum distance between original and new labels for inexact matches.
        Read Xarray manual on nearest-neighbor-lookups

    Returns
    -------
    data_retrieval_time : str
        The median time of repeated operations it took to retrieve data over
        the requested location

    Notes
    -----
    ``mask_and_scale`` is always set to ``False`` to avoid errors related with
    decoding timestamps. See also ...

    """
    from .models import get_file_format

    file_format = get_file_format(time_series)
    open_dataset_options = file_format.open_dataset_options()  # some Class function !

    # dataset_select_options = file_format.dataset_select_options(tolerance)
    # indexers = set_location_indexers(
    #     data_array=time_series,
    #     longitude=longitude,
    #     latitude=latitude,
    #     verbose=verbose,
    # )

    try:
        # with xr.open_dataset(str(time_series), **open_dataset_options) as dataset:
        #     _ = (
        #         dataset[variable]
        #         .sel(
        #             lon=longitude,
        #             lat=latitude,
        #             method="nearest",
        #             **dataset_select_options,
        #         )
        #         .load()
        #     )
        return xr.open_dataset(
            time_series.as_posix(),
            **open_dataset_options,
        )

    except Exception as exception:
        print(
            f"Cannot open [code]{time_series}[/code] from [code]{time_series}[/code] via Xarray: {exception}"
        )
        raise SystemExit(33)


def read_large_parquet_optimized(parquet_store: Path, variable: str):
    """Read large Parquet store with memory optimization."""
    
    # Use kerchunk engine for Parquet reading with streaming optimizations
    open_options = {
        "engine": "kerchunk",
        "storage_options": {
            "remote_protocol": "file",
            # Disable caching for large files to save memory
            "cache_size": 0,  
        },
        # Don't load everything into memory immediately
        "chunks": None,  # Let Dask decide initial chunking
    }
    
    print(f"Reading large Parquet store: {parquet_store}")
    dataset = xr.open_dataset(
            str(parquet_store),
            **open_options,
            )
    
    # Drop unnecessary variables early to save memory
    dataset = drop_other_data_variables(dataset)
    
    return dataset


def parse_compression_filters(compressing_filters: str) -> List[str]:
    if isinstance(compressing_filters, str):
        return compressing_filters.split(",")
    else:
        raise typer.BadParameter("Compression filters input must be a string")


def generate_zarr_store(
    dataset: Dataset,
    variable: str,
    store: LocalStore, # for Zarr 3
    # latitude_chunks: int,
    # longitude_chunks: int,
    # time_chunks: int = -1,
    compute: bool = DASK_COMPUTE,
    consolidate: bool = ZARR_CONSOLIDATE_DEFAULT,
    compressor = ZARR_COMPRESSOR,
    mode: str = 'w-',
    overwrite_output: bool = False,
):
    """
    Notes
    -----

    Files produced by `ncks` or with legacy compression : often have .encoding
    attributes referencing numcodecs codecs (e.g., numcodecs.shuffle.Shuffle),
    which are not accepted by Zarr v3. In order to avoid encoding related
    errors, we clear the legacy encoding from all variables before writing.

    """
    # Reset legacy encoding
    for var in dataset.variables:
        dataset[var].encoding = {}  # Critical step!
   
    # print(f" {GREEN_DASH} Chunk the dataset")
    # dataset = dataset.chunk({"time": time_chunks, "lat": latitude_chunks, "lon": longitude_chunks})
    # print(f'   > Dataset shape after chunking : {dataset.data_vars}')

    print(f"   Define the store path for the current chunking shape")
    store = LocalStore(store)  # for Zarr 3

    # Define Zarr v3 encoding
    encoding = {
        dataset[variable].name: {"compressors": (compressor,)},
    }
    for coordinate in dataset.coords:
        encoding[coordinate] = {"compressors": (), "filters": ()}
   
    # Write to Zarr
    if compute == False:
        print(f' {GREEN_DASH} Build the Dask task graph')
    print(f' {GREEN_DASH} Generate Zarr store')

    if overwrite_output:
        mode = "w"

    return dataset.to_zarr(
        store=store,
        compute=compute,
        consolidated=consolidate,
        encoding=encoding,
        zarr_format=3,
        mode=mode,
        # safe_chunks=False,
    )


def generate_zarr_store_streaming(
    dataset,
    variable: str,
    data_type: str,
    store: str,
    compute: bool = True,
    consolidate: bool = True,
    compressor = ZARR_COMPRESSOR,
    zarr_format: int = 3
):
    """
    Generate Zarr store with v3 optimizations and streaming approach.
    """
    from zarr.storage import LocalStore
    
    # Clean up encoding for Zarr v3 compatibility
    dataset.drop_encoding()
    
    # LocalStore for Zarr v3
    store_obj = LocalStore(store)

    # Determine the full size of the time dimension
    time_size = dataset.sizes['time']
    
    # Define encoding: time is contiguous, others retain their chunking
    encoding = {}
    for variable in dataset.data_vars:
        dimensions = dataset[variable].dims
        # dtype = dataset[variable].dtype
        chunks = {}

        # if mask_and_scale:
        #     fill_value = dataset[variable].encoding['_FillValue']
        # else:
        #     fill_value = None

        for dimension in dimensions:
        # for i, dimension in enumerate(dimensions):

            # Replace time chunk with full length
            if dimension == "time":
                # if not time:
                chunks.update(time=time_size)
            # elif dimension == "lat":
            #     chunks.update(lat=latitude)

            # elif dimension == "lon":
            #     chunks.update(lon=longitude)

            # else:
            #     chunks.append(dataset.sizes[dimension])
            # else:
                # Use existing chunking or default to full dimension if not chunked
                # chunks.update(chunks[i] if chunks[i] is not None else dataset.dims[dim])

        # Define (Zarr v3) encoding
        encoding = {
            dataset[variable].name: {
                "chunks": chunks,
                "compressors": (compressor,),
                "filters": (),
                # "shards": None,
                # '_FillValue': fill_value,
                # "dtype": data_type,
                },
        }

    # for coordinate in dataset.coords:
    #     encoding[coordinate] = {"compressors": (), "filters": ()}

    # Optionally clear coordinate encodings for Zarr v3 compatibility
    for coord in dataset.coords:
        encoding[coord] = {"compressors": (), "filters": ()}

    # Now pass this encoding to .to_zarr()

    # Stream the conversion to avoid memory buildup
    if compute:
        print("Starting streaming conversion to Zarr...")
        result = dataset.to_zarr(
            store=store_obj,
            mode='w',
            compute=True,
            consolidated=consolidate,
            encoding=encoding,
            zarr_format=zarr_format
        )
        
        if consolidate:
            # Zarr v3 consolidated metadata provides significant performance boost
            print("Consolidating metadata...")
            zarr.consolidate_metadata(store_obj)
        
        return result

    else:
        # Return delayed computation for manual scheduling
        return dataset.to_zarr(
            store=store_obj,
            mode='w',
            compute=False,
            consolidated=consolidate,
            encoding=encoding,
            zarr_format=zarr_format
        )


def convert_parquet_to_zarr_store(
    parquet_store: Annotated[Path, typer_argument_time_series],
    zarr_store: Annotated[Path, typer.Argument(help='Local Zarr store')],
    variable: Annotated[str, typer_argument_variable],
    data_type: Annotated[str, typer.Option(help="Data type")] = 'float32',
    drop_other_variables: bool = True,
    compression: Annotated[
        str, typer.Option(help="Compression filter")
    ] = COMPRESSION_FILTER_DEFAULT,
    compression_level: Annotated[
        int, typer.Option(help="Compression level")
    ] = COMPRESSION_LEVEL_DEFAULT,
    shuffling: Annotated[str, typer.Option(help=f"Shuffle... ")] = SHUFFLING_DEFAULT,
    # backend: Annotated[RechunkingBackend, typer.Option(help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]")] = RechunkingBackend.nccopy,
    dask_scheduler: Annotated[
        str, typer.Option(help="The port:ip of the dask scheduler")
    ] = None,
    workers: Annotated[int, typer.Option(help="Number of workers")] = None,
    threads_per_worker: Annotated[int, typer.Option(help="Threads per worker")] = 2,
    memory_limit: Annotated[int, typer.Option(help="Memory limit for the Dask cluster in GB. [yellow bold]Will override [code]auto-memory[/code][/yellow bold]")] = None,
    auto_memory_limit: Annotated[bool, typer.Option(help="Memory limit per worker")] = True,
    consolidate: Annotated[bool, typer.Option(help="Consolidate Zarr store metadata. [black on yellow] Not part in Zarr 3 [/black on yellow]")]= ZARR_CONSOLIDATE_DEFAULT,
    compute: Annotated[bool, typer.Option(help="Compute immediately [code]True[/code] or build a Dask task graph [code]False[code]")] = DASK_COMPUTE,
    mode: Annotated[ str, typer.Option(help="Writing file mode")] = 'w-',
    overwrite_output: Annotated[bool, typer.Option(help="Overwrite existing output file")] = False,
    dry_run: Annotated[bool, typer_option_dry_run] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Convert a single large Parquet store to Zarr efficiently.

    1. Read Parquet index via the Zarr engine
    2. Rechunk. Again.
    3. Generate Zarr store
    4. Time speed of reading complete time series over a single geographic location

    """
    if not memory_limit:
        memory_limit = auto_memory_limit

    if dry_run:
        config = auto_configure_for_large_dataset(
            memory_limit=memory_limit,
            workers=workers,
            threads_per_worker=threads_per_worker,
        )
        print(f"Would use Dask configuration: {config}")
        print(f"Input: {parquet_store}")
        print(f"Output: {zarr_store}")
        return
    
    # Ensure output directory exists
    if not zarr_store:
        zarr_store = Path(f"{parquet_store.stem}.zarr")
        # zarr_store.parent.mkdir(parents=True, exist_ok=True)
    
   # Auto-configure Dask for single large file processing
    dask_config = auto_configure_for_large_dataset(
        memory_limit=memory_limit,
        workers=workers,
        threads_per_worker=threads_per_worker,
        verbose=verbose,
    )
    
    with LocalCluster(**dask_config) as cluster, Client(cluster) as client:
        print(f"Dashboard: {client.dashboard_link}")

        # Read Parquet Index
        dataset = read_parquet_via_zarr(
        # dataset = read_large_parquet_optimized(
                # time_series=parquet_store,
                parquet_store,
                variable=variable,
        )
        # Drop "other" data variables ?
        if drop_other_variables:
            dataset = drop_other_data_variables(dataset)
    
        compressor = zarr.codecs.BloscCodec(
            cname=compression,
            clevel=compression_level,
            shuffle=shuffling,
        )
        # Generate Zarr store with streaming approach
        generate_zarr_store_streaming(
            dataset=dataset,
            variable=variable,
            data_type=data_type,
            store=str(zarr_store),
            compute=compute,
            consolidate=consolidate,
            compressor=compressor,
            # mode=mode,
            # overwrite_output=overwrite_output,
        )
