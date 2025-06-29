import shlex
import subprocess
from pathlib import Path
from typing import Optional
import netCDF4 as nc
import typer

from .typer_parameters import (
    typer_option_dry_run,
    typer_argument_source_path_with_pattern,
    typer_option_output_directory,
    typer_option_filename_pattern,
)
import xarray as xr

from rich import print

from typing_extensions import Annotated

from rekx.backend import RechunkingBackend
from rekx.constants import VERBOSE_LEVEL_DEFAULT
from rekx.typer_parameters import typer_option_verbose

from .log import logger
from .models import XarrayVariableSet, select_xarray_variable_set_from_dataset
from .typer_parameters import (
    typer_option_dry_run,
)
from dask.distributed import Client
from functools import partial
from pathlib import Path
from typing import Optional
import typer
import dask


from .nccopy_constants import (
    FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    CACHE_SIZE_DEFAULT,
    CACHE_ELEMENTS_DEFAULT,
    CACHE_PREEMPTION_DEFAULT,
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
    RECHUNK_IN_MEMORY_DEFAULT,
    DRY_RUN_DEFAULT,
)

def modify_chunk_size(
    netcdf_file,
    variable,
    chunk_size,
):
    """
    Modify the chunk size of a variable in a NetCDF file.

    Parameters:
    - nc_file: path to the NetCDF file
    - variable_name: name of the variable to modify
    - new_chunk_size: tuple specifying the new chunk size, e.g., (2600, 2600)
    """
    with nc.Dataset(netcdf_file, "r+") as dataset:
        variable = dataset.variables[variable]

        if variable.chunking() != [None]:
            variable.set_auto_chunking(chunk_size)
            print(
                f"Modified chunk size for variable '{variable}' in file '{netcdf_file}' to {chunk_size}."
            )

        else:
            print(
                f"Variable '{variable}' in file '{netcdf_file}' is not chunked. Skipping."
            )


def _rechunk_single_file(
    input_file: Path,
    output_file: Path,
    time: int,
    latitude: int,
    longitude: int,
    # Other parameters
) -> None:
    """Core rechunking logic for a single file"""
    try:
        with xr.open_dataset(input_file, engine="netcdf4") as dataset:
            # Your existing rechunking logic here
            # Example:
            encoding = {}
            for var in dataset.data_vars:
                chunk_sizes = []
                for dim in dataset[var].dims:
                    if dim == 'time': 
                        chunk_size = time if time > 0 else len(dataset[dim])
                    elif dim == 'lat': 
                        chunk_size = latitude
                    elif dim == 'lon': 
                        chunk_size = longitude
                    else:
                        chunk_size = len(dataset[dim])
                    chunk_sizes.append(chunk_size)
                encoding[var] = {'chunksizes': tuple(chunk_sizes)}
            
            dataset.to_netcdf(
                output_file,
                encoding=encoding,
                engine="h5netcdf"
            )
        typer.echo(f"Processed {input_file.name}")
    except Exception as e:
        typer.echo(f"Error processing {input_file.name}: {str(e)}")


def rechunk(
    input_filepath: Annotated[Path, typer.Argument(help="Input NetCDF file.")],
    output_filepath: Annotated[
        Optional[Path], typer.Argument(help="Path to the output NetCDF file.")
    ],
    time: Annotated[int, typer.Option(help="New chunk size for the `time` dimension.")],
    latitude: Annotated[
        int, typer.Option(help="New chunk size for the `lat` dimension.")
    ],
    longitude: Annotated[
        int, typer.Option(help="New chunk size for the `lon` dimension.")
    ],
    fix_unlimited_dimensions: Annotated[
        bool, typer.Option(help="Convert unlimited size input dimensions to fixed size dimensions in output.")
    ] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    variable_set: Annotated[
        list[XarrayVariableSet], typer.Option(help="Set of Xarray variables to diagnose")
    ] = list[XarrayVariableSet.all],
    cache_size: Optional[int] = CACHE_SIZE_DEFAULT,
    cache_elements: Optional[int] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Optional[float] = CACHE_PREEMPTION_DEFAULT,
    compression: str = COMPRESSION_FILTER_DEFAULT,
    compression_level: int = COMPRESSION_LEVEL_DEFAULT,
    shuffling: str = SHUFFLING_DEFAULT,
    memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    mode: Annotated[ str, typer.Option(help="Writing file mode")] = 'w-',
    overwrite_output: Annotated[bool, typer.Option(help="Overwrite existing output file")] = False,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    backend: Annotated[
        RechunkingBackend,
        typer.Option(
            help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]"
        ),
    ] = RechunkingBackend.xarray,
    dask_scheduler: Annotated[
        str, typer.Option(help="The port:ip of the dask scheduler")
    ] = None,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Rechunk a NetCDF4 dataset with options to fine tune the output
    """
    if verbose:
        import time as timer

        rechunking_timer_start = timer.time()

    # if dask_scheduler:
    #     from dask.distributed import Client
    #     client = Client(dask_scheduler)
    #     typer.echo(f"Using Dask scheduler at {dask_scheduler}")

    try:
        with xr.open_dataset(input_filepath, engine="netcdf4") as dataset:
            # with Dataset(input, 'r') as dataset:
            # def validate_variable_set(variable_set_input: list[str]) -> list[XarrayVariableSet]:
            #     if variable_set_input in XarrayVariableSet.__members__:
            #         return XarrayVariableSet[variable_set_input]
            #     else:
            #         raise ValueError(f"Invalid variable set: {variable_set_input}")

            def validate_variable_set(variable_set_input: list[str]) -> list[XarrayVariableSet]:
                if not variable_set_input:
                    # Use a sensible default or raise
                    return [XarrayVariableSet.all]
                validated = []
                for v in variable_set_input:
                    if v in XarrayVariableSet.__members__:
                        validated.append(XarrayVariableSet[v])
                    else:
                        raise ValueError(f"Invalid variable set: {v}")
                return validated

            variable_set = validate_variable_set(variable_set)
            selected_variables = select_xarray_variable_set_from_dataset(
                XarrayVariableSet, variable_set, dataset
            )
            backend_name = backend.name
            backend = backend.get_backend()
            command = backend.rechunk(
                input_filepath=input_filepath,
                variables=list(selected_variables),
                output_filepath=output_filepath,
                time=time,
                latitude=latitude,
                longitude=longitude,
                fix_unlimited_dimensions=fix_unlimited_dimensions,
                cache_size=cache_size,
                cache_elements=cache_elements,
                cache_preemption=cache_preemption,
                compression=compression,
                compression_level=compression_level,
                shuffling=shuffling,
                memory=memory,
                mode=mode,
                overwrite_output=overwrite_output,
                dry_run=dry_run,  # just return the command!
            )

            if dry_run:
                print(
                    f"[bold]Dry run[/bold] the [bold]following command that would be executed[/bold] :",
                    f"    {command}"
                    )

                return  # Exit for a dry run

            else:
                # Only nccopy backend returns executable commands
                
                if backend_name == RechunkingBackend.nccopy.name:
                    subprocess.run(shlex.split(command), check=True)
                    command_arguments = shlex.split(command)
                    try:
                        subprocess.run(command_arguments, check=True)
                        print(f"Command {command} executed successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"An error occurred while executing the command: {e}")

                else:
                    print(f"Rechunking completed: {command}")

            if verbose:
                rechunking_timer_end = timer.time()
                elapsed_time = rechunking_timer_end - rechunking_timer_start
                logger.debug(f"Rechunking via {backend} took {elapsed_time:.2f} seconds")
                print(f"Rechunking took {elapsed_time:.2f} seconds.")

    except Exception as e:
        typer.echo(f"Error processing {input_filepath.name}: {str(e)}")


def rechunk_netcdf_files(
    source_path: Annotated[Path, typer_argument_source_path_with_pattern],
    time: Annotated[int, typer.Option(help="New chunk size for the `time` dimension.")],
    latitude: Annotated[
        int, typer.Option(help="New chunk size for the `lat` dimension.")
    ],
    longitude: Annotated[
        int, typer.Option(help="New chunk size for the `lon` dimension.")
    ],
    pattern: Annotated[str, typer_option_filename_pattern] = "*.nc",
    output_directory: Annotated[Path, typer_option_output_directory] = Path('.'),
    fix_unlimited_dimensions: Annotated[
        bool, typer.Option(help="Convert unlimited size input dimensions to fixed size dimensions in output.")
    ] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    variable_set: Annotated[
        list[XarrayVariableSet], typer.Option(help="Set of Xarray variables to diagnose")
    ] = list[XarrayVariableSet.all],
    cache_size: Optional[int] = CACHE_SIZE_DEFAULT,
    cache_elements: Optional[int] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Optional[float] = CACHE_PREEMPTION_DEFAULT,
    compression: str = COMPRESSION_FILTER_DEFAULT,
    compression_level: int = COMPRESSION_LEVEL_DEFAULT,
    shuffling: str = SHUFFLING_DEFAULT,
    memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    backend: Annotated[
        RechunkingBackend,
        typer.Option(
            help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]"
        ),
    ] = RechunkingBackend.xarray,
    mode: Annotated[ str, typer.Option(help="Writing file mode")] = 'w-',
    overwrite_output: Annotated[bool, typer.Option(help="Overwrite existing output file")] = False,
    workers: Annotated[int, typer.Option(help="Number of worker processes.")] = 4,
    memory_limit: str = "4GB",
    dry_run: Annotated[bool, typer_option_dry_run] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
) -> None:
    """Rechunk multiple NetCDF files in parallel"""
    # Resolve input files
    if source_path.is_file():
        input_file_paths = [source_path]

    elif source_path.is_dir():
        input_file_paths = list(source_path.glob(pattern))

    else:
        raise ValueError(f"Invalid input path: {source_path}")
    
    if not input_file_paths:
        typer.echo("No files found matching pattern")
        return

    if dry_run:
        dry_run_message = (
                f"[bold]Dry running operations that would be performed[/bold] :"
                f"\n"
                f"> Reading files in [code]{source_path}[/code] matching the pattern [code]{pattern}[/code]"
                f"\n"
                f"> Number of files matched : {len(list(input_file_paths))}"
                f"\n"
                f"> Writing rechunked data in [code]{output_directory}[/code]"
                )
        print(dry_run_message)

    if input_file_paths and not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        if verbose > 0:
            print(f"[yellow]Convenience action[/yellow] : creating the requested output directory [code]{output_directory}[/code].")

    # Prepare output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    # output_files = [Path(output_directory) / f"{f.stem}_rechunked{f.suffix}" for f in input_file_paths]
    # output_files = [Path(output_directory) / f.name for f in input_file_paths]

    output_filename_base = f"{time}_{latitude}_{longitude}_{compression}_{compression_level}"
    if shuffling and compression_level > 0:
        output_filename_base += "_shuffled"
    output_files = [
        output_directory / f"{f.stem}_{output_filename_base}{f.suffix}"
        for f in input_file_paths
    ]

    # Initialize parallel client
    client = Client(
        n_workers=workers,
        threads_per_worker=1,  # Better for I/O bound tasks
        memory_limit=memory_limit,
    )
    if verbose:
        typer.echo(f"Processing {len(input_file_paths)} files with {workers} workers")

    # Create processing function with fixed parameters
    # with multiprocessing.Pool(processes=workers) as pool:
    partial_rechunk_command = partial(
        rechunk,
        time=time,
        latitude=latitude,
        longitude=longitude,
        fix_unlimited_dimensions=fix_unlimited_dimensions,
        variable_set=variable_set,
        cache_size=cache_size,
        cache_elements=cache_elements,
        cache_preemption=cache_preemption,
        compression=compression,
        compression_level=compression_level,
        shuffling=shuffling,
        memory=memory,
        mode=mode,
        overwrite_output=overwrite_output,
        dry_run=dry_run,  # just return the command!
        backend=backend,
    )
        # pool.map(partial_rechunk_command, input_file_paths)
    if verbose:
        print(f"[bold green]Done![/bold green]")

    # Process files in parallel
    tasks = [
        dask.delayed(partial_rechunk_command)(in_file, out_file)
        for in_file, out_file in zip(input_file_paths, output_files)
    ]

    dask.compute(*tasks)
    if verbose:
        typer.echo("Parallel rechunking completed")
