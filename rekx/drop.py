from xarray import Dataset


def drop_other_data_variables(
    dataset: Dataset,
    other_variables: list[str] = ['lat_bnds', 'lon_bnds', 'record_status'],
    drop_other_variables: bool = True,
    other_dimensions: list[str] = ['bnds'],
    drop_other_dimensions: bool = True,
    other_attributes: list[str] = ['bounds'],
    drop_other_attributes: bool = True,
) -> Dataset:
    """
    Clean up an xarray Dataset by removing:
    - Unwanted variables
    - Unused dimensions
    - Unwanted attributes from coordinates
    
    Parameters
    ----------
    dataset:
        Input Dataset to process
    other_variables:
        Variables to remove
    drop_other_variables:
        Toggle variable removal
    other_dimensions:
        Dimensions to remove if unused
    drop_other_dimensions:
        Toggle dimension removal
    other_attributes:
        Attributes to remove
    drop_other_attributes:
        Toggle attribute removal
    
    Returns
    -------
        Cleaned Dataset

    """
    # Remove specified variables
    if drop_other_variables:
        variables_to_drop = [variable for variable in other_variables if variable in dataset]
        dataset = dataset.drop_vars(variables_to_drop)

    # Remove unused dimensions
    if drop_other_dimensions:
        for dimension in other_dimensions:
            if dimension in dataset.dims:
                # Check if dimension is still used
                if not any(dimension in dataset[variable].dims for variable in dataset.variables):
                    dataset = dataset.drop_dims(dimension)

    # Remove specified attributes
    if drop_other_attributes:
        for coordinate in ['lat', 'lon']:
            if coordinate in dataset.coords:
                for attribute in other_attributes:
                    dataset[coordinate].attrs.pop(attribute, None)  # Safe removal

    return dataset
