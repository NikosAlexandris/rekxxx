def configure_dask(
    n_workers: int | None,
    threads_per_worker: int | None,
    memory_limit: str | None,
):
    """
    """
    import psutil, os

    physical_cores = (
        psutil.cpu_count(logical=False) or os.cpu_count() // 2
    )  # fallback[13][19]
    total_ram = psutil.virtual_memory().total  # bytes[10][29]

    # Sensible defaults
    n_workers = n_workers or max(1, physical_cores // 2)
    threads_per_worker = threads_per_worker or 1
    if memory_limit is None:
        per_worker = total_ram // n_workers
        # leave 5 % buffer for OS
        memory_limit = f"{int(per_worker * 0.95 / 1e9)}GB"

    return dict(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=":8787",
    )


def auto_configure_for_large_dataset(
    memory_limit: float| str = "auto", 
    workers: int | None = None,
    threads_per_worker: int = 2
):
    """Configure Dask optimally for single large dataset processing."""
    import psutil
    
    total_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count(logical=False)
    
    if memory_limit == "auto":
        # Use 80% of available memory, distributed across workers
        available_memory = int(total_memory * 0.8)
    else:
        available_memory = int(total_memory * memory_limit)
        
    if workers is None:
        # For large single files, fewer workers with more memory each
        workers = min(8, cpu_count // 2)  # Conservative worker count
    
    memory_per_worker = available_memory // workers

    # Memory management parameters
    import dask.config
    dask.config.set({
        'distributed.worker.memory.target': 0.75,  # Set target memory fraction
        'distributed.worker.memory.spill': 0.85,    # Set spill memory fraction
        'distributed.worker.memory.pause': 0.9,
    })

    # Pass memory configuration parameters directly, not in worker_options
    return {
        'n_workers': workers,
        'threads_per_worker': threads_per_worker,
        'memory_limit': f"{memory_per_worker // (1024**3)}GB",
        'processes': True,
        'dashboard_address': ':8787',
    }
