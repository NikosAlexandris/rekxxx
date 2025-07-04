from rich import print
from typing_extensions import Annotated
from pathlib import Path
import typer
from .typer_parameters import (
    typer_option_dry_run,
    typer_argument_source_path_with_pattern,
    typer_option_output_directory,
    typer_option_filename_pattern,
)
from rekx.backend import RechunkingBackend
from .models import XarrayVariableSet, select_xarray_variable_set_from_dataset
from rekx.constants import VERBOSE_LEVEL_DEFAULT

from typing import Union
import multiprocessing
from functools import partial
from typing import List
import xarray as xr
from rich import print
from typing_extensions import Annotated

from rekx.messages import NOT_IMPLEMENTED_CLI
from rekx.typer_parameters import typer_option_verbose


from .nccopy_constants import (
    FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    CACHE_SIZE_DEFAULT,
    CACHE_ELEMENTS_DEFAULT,
    CACHE_PREEMPTION_DEFAULT,
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
    RECHUNK_IN_MEMORY_DEFAULT,
    SPATIAL_SYMMETRY_DEFAULT,
)


def parse_chunks(chunks: Union[int, str]) -> List[int]:
    if isinstance(chunks, str):
        return [int(chunk_size) for chunk_size in chunks.split(",")]
    elif isinstance(chunks, int):
        return [chunks]
    else:
        raise typer.BadParameter("Chunks must be a list of integers.")


def parse_compression_filters(compressing_filters: str) -> List[str]:
    if isinstance(compressing_filters, str):
        return compressing_filters.split(",")
    else:
        raise typer.BadParameter("Compression filters input must be a string")


def parse_numerical_option(input: int) -> List[int]:
    if isinstance(input, int):
        return [input]
    elif isinstance(input, str):
        return [int(property) for property in input.split(",")]
    else:
        raise typer.BadParameter(
            "Input must be a either a single integer or float or a string of comma-separated values."
        )


def parse_float_option(input: float) -> List[float]:
    if isinstance(input, str):
        print(f"This input is a string!")
        return [float(string) for string in input.split(",")]
    return [input]


def callback_compression_filters():
    return ["zlib"]


def generate_rechunk_commands(
    input_filepath: Annotated[Path, typer.Argument(help="Input NetCDF file.")],
    output: Annotated[
        Path | None, typer.Argument(help="Path to the output NetCDF file.")
    ],
    time: Annotated[
        int | None,
        typer.Option(
            help="New chunk size for the `time` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    latitude: Annotated[
        int | None,
        typer.Option(
            help="New chunk size for the `lat` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    longitude: Annotated[
        int | None,
        typer.Option(
            help="New chunk size for the `lon` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    fix_unlimited_dimensions: Annotated[
        bool,
        typer.Option(
            help="Convert unlimited size input dimensions to fixed size dimensions in output."
        ),
    ] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    spatial_symmetry: Annotated[
        bool,
        typer.Option(
            help="Add command only for identical latitude and longitude chunk sizes"
        ),
    ] = SPATIAL_SYMMETRY_DEFAULT,
    variable_set: Annotated[
        XarrayVariableSet, typer.Option(help="Set of Xarray variables to diagnose")
    ] = XarrayVariableSet.all,
    cache_size: Annotated[
        int,
        typer.Option(
            help="Cache size", show_default=True, parser=parse_numerical_option
        ),
    ] = CACHE_SIZE_DEFAULT,
    cache_elements: Annotated[
        int,
        typer.Option(help="Number of elements in cache", parser=parse_numerical_option),
    ] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Annotated[
        float,
        typer.Option(
            help=f"Cache preemption strategy {NOT_IMPLEMENTED_CLI}",
            parser=parse_float_option,
        ),
    ] = CACHE_PREEMPTION_DEFAULT,
    compression: Annotated[
        str, typer.Option(help="Compression filter", parser=parse_compression_filters)
    ] = COMPRESSION_FILTER_DEFAULT,
    compression_level: Annotated[
        int, typer.Option(help="Compression level", parser=parse_numerical_option)
    ] = COMPRESSION_LEVEL_DEFAULT,
    shuffling: Annotated[bool, typer.Option(help=f"Shuffle... ")] = SHUFFLING_DEFAULT,
    memory: Annotated[
        bool, typer.Option(help="Use the -w flag to nccopy")
    ] = RECHUNK_IN_MEMORY_DEFAULT,
    # backend: Annotated[RechunkingBackend, typer.Option(help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]")] = RechunkingBackend.nccopy,
    dask_scheduler: Annotated[
        str, typer.Option(help="The port:ip of the dask scheduler")
    ] = None,
    commands_file: Path = "rechunk_commands.txt",
    dry_run: Annotated[bool, typer_option_dry_run] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Generate variations of rechunking commands based on `nccopy`.
    """
    # Shuffling makes sense only along with compression
    if any([level > 0 for level in compression_level]) and shuffling:
        shuffling = [shuffling, False]
    else:
        shuffling = [False]
    with xr.open_dataset(input_filepath, engine="netcdf4") as dataset:
        selected_variables = select_xarray_variable_set_from_dataset(
            XarrayVariableSet, variable_set, dataset
        )
        import itertools

        commands = []
        for (
            chunking_time,
            chunking_latitude,
            chunking_longitude,
            caching_size,
            caching_elements,
            caching_preemption,
            compressing_filter,
            compressing_level,
            shuffling,
        ) in itertools.product(
            time,
            latitude,
            longitude,
            cache_size,
            cache_elements,
            cache_preemption,
            compression,
            compression_level,
            shuffling,
        ):
            backend = RechunkingBackend.nccopy.get_backend()  # hard-coded!
            # Review Me ----------------------------------------------------
            if spatial_symmetry and chunking_latitude != chunking_longitude:
                continue
            else:
                command = backend.rechunk(
                    input_filepath=input_filepath,
                    variables=list(selected_variables),
                    output_directory=output,
                    time=chunking_time,
                    latitude=chunking_latitude,
                    longitude=chunking_longitude,
                    fix_unlimited_dimensions=fix_unlimited_dimensions,
                    cache_size=caching_size,
                    cache_elements=caching_elements,
                    cache_preemption=caching_preemption,
                    compression=compressing_filter,
                    compression_level=compressing_level,
                    shuffling=shuffling,
                    memory=memory,
                    dry_run=True,  # just return the command!
                )
                if not command in commands:
                    commands.append(command)

    commands_file = Path(
        commands_file.stem + "_for_" + Path(input_filepath).stem + commands_file.suffix
    )
    if verbose:
        print(
            f"[bold]Writing generated commands into[/bold] [code]{commands_file}[/code]"
        )
        for command in commands:
            print(f" [green]>[/green] [code dim]{command}[/code dim]")

    if not dry_run:
        with open(commands_file, "w") as f:
            for command in commands:
                f.write(command + "\n")


def generate_rechunk_commands_for_multiple_netcdf(
    source_path: Annotated[Path, typer_argument_source_path_with_pattern],
    time: Annotated[
        int,
        typer.Option(
            help="New chunk size for the `time` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    latitude: Annotated[
        int,
        typer.Option(
            help="New chunk size for the `lat` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    longitude: Annotated[
        int,
        typer.Option(
            help="New chunk size for the `lon` dimension.",
            parser=parse_numerical_option,
        ),
    ],
    fix_unlimited_dimensions: Annotated[
        bool,
        typer.Option(
            help="Convert unlimited size input dimensions to fixed size dimensions in output."
        ),
    ] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    pattern: Annotated[str, typer_option_filename_pattern] = "*.nc",
    output_directory: Annotated[Path, typer_option_output_directory] = Path('.'),
    spatial_symmetry: Annotated[
        bool,
        typer.Option(
            help="Add command only for identical latitude and longitude chunk sizes"
        ),
    ] = SPATIAL_SYMMETRY_DEFAULT,
    variable_set: Annotated[
        list[XarrayVariableSet], typer.Option(help="Set of Xarray variables to diagnose")
    ] = list[XarrayVariableSet.all],
    cache_size: Annotated[
        int,
        typer.Option(
            help="Cache size", show_default=True, parser=parse_numerical_option
        ),
    ] = CACHE_SIZE_DEFAULT,
    cache_elements: Annotated[
        int,
        typer.Option(help="Number of elements in cache", parser=parse_numerical_option),
    ] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Annotated[
        float,
        typer.Option(
            help=f"Cache preemption strategy {NOT_IMPLEMENTED_CLI}",
            parser=parse_float_option,
        ),
    ] = CACHE_PREEMPTION_DEFAULT,
    compression: Annotated[
        str, typer.Option(help="Compression filter", parser=parse_compression_filters)
    ] = COMPRESSION_FILTER_DEFAULT,
    compression_level: Annotated[
        int, typer.Option(help="Compression level", parser=parse_numerical_option)
    ] = COMPRESSION_LEVEL_DEFAULT,
    shuffling: Annotated[
        bool,
        typer.Option(
            help=f"Shuffle... [reverse bold orange] Testing [/reverse bold orange]"
        ),
    ] = SHUFFLING_DEFAULT,
    memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    # backend: Annotated[RechunkingBackend, typer.Option(help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]")] = RechunkingBackend.nccopy,
    dask_scheduler: Annotated[
        str, typer.Option(help="The port:ip of the dask scheduler")
    ] = None,
    commands_file: Path = "rechunk_commands.txt",
    workers: Annotated[int, typer.Option(help="Number of worker processes.")] = 4,
    dry_run: Annotated[bool, typer_option_dry_run] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Generate variations of rechunking commands based on `nccopy`.
    """
    if source_path.is_file():
        input_file_paths = [source_path]
        print(f"[green]Identified the file in question![/green]")

    elif source_path.is_dir():
        input_file_paths = list(str(path) for path in source_path.glob(pattern))

    else:
        raise ValueError(f"Invalid input path: {source_path}")

    if not list(input_file_paths):
        print(
            f"No files found in [code]{source_path}[/code] matching the pattern [code]{pattern}[/code]!"
        )
        return

    if dry_run:
        dry_run_message = (
                f"[bold]Dry running operations that would be performed[/bold] :"
                f"\n"
                f"> Reading files in [code]{source_path}[/code] matching the pattern [code]{pattern}[/code]"
                f"\n"
                f"> Number of files matched : {len(list(input_file_paths))}"
                f"\n"
                f"> Writing rechunking commands in [code]{commands_file}[/code]"
                )
        print(dry_run_message)

    if input_file_paths and not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        if verbose > 0:
            print(f"[yellow]Convenience action[/yellow] : creating the requested output directory [code]{output_directory}[/code].")
    with multiprocessing.Pool(processes=workers) as pool:
        partial_generate_rechunk_commands = partial(
                generate_rechunk_commands,
                output=output_directory,
                time=time,
                latitude=latitude,
                longitude=longitude,
                fix_unlimited_dimensions=fix_unlimited_dimensions,
                spatial_symmetry=spatial_symmetry,
                variable_set=variable_set,
                cache_size=cache_size,
                cache_elements=cache_elements,
                cache_preemption=cache_preemption,
                compression=compression,
                compression_level=compression_level,
                shuffling=shuffling,
                memory=memory,
                dask_scheduler=dask_scheduler,
                commands_file=commands_file,
                dry_run=dry_run,
                verbose=verbose,
        )
        pool.map(partial_generate_rechunk_commands, input_file_paths)
    if verbose:
        print(f"[bold green]Done![/bold green]")
