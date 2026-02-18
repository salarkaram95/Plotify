import numpy as np
import xarray as xr
import gsw

def calc_mld(sa, ct, depth, delta_rho=0.03, ref_depth=10.0, return_ml_prop=False):
    """
    Calculate mixed layer depth using a density threshold relative to a reference depth, and optionally return mixed layer SA and CT. 
    
    NOTE: This assumes that the bottom grid cell is either fully inside or fully outside of the mixed layer. If the MLD cuts through the middle of a cell, this overestimates volume.

    Parameters
    -------
    sa: xarray.DataArray
        Absolute Salinity [depth, y, x] or [time, depth, y, x]
    ct: xarray.DataArray
        Conservative Temperature [depth, y, x] or [time, depth, y, x]
    depth: xarray.DataArray
        Depths corresponding to rho (deptht)
    delta_rho: float
        Density increase from reference depth defining the MLD (kg/m³) (default 0.03 kg/m³) 
    ref_depth: float, optional
        Reference depth for MLD calculation (default: 10 m)
    return_ml_properties: bool
        If True, also return mixed layer mean SA and CT

    Returns
    --
    mld: xarray.DataArray
        Mixed layer depth
    sa_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Absolute Salinity
    ct_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Conservative Temperature   
    """

    # Calculate density
    sigma_0 = xr.apply_ufunc(
        gsw.sigma0,
        sa,
        ct,
        dask="allowed" 
    )

    # Find reference level, default = 10 m
    k_ref = np.abs(depth - ref_depth).argmin().item()
    sigma_ref = sigma_0.isel(deptht=k_ref)
    has_ref = sigma_ref.notnull()

    # Index thresholds
    mask = sigma_0 >= (sigma_ref + delta_rho)

    
    # First depth index where threshold is met
    k_mld = mask.argmax("deptht")
    
    # Columns where threshold never met
    never_met = ~mask.any("deptht")

    # Convert index to depth
    mld = depth.isel(deptht=k_mld)

    # Bottom depth
    bottom_depth = depth.where(sa.notnull()).max("deptht")

    # If threshold never met but reference valid → mixed to bottom
    mld = xr.where(never_met & has_ref, bottom_depth, mld)

    # If reference invalid → keep NaN
    mld = mld.where(has_ref)

    if not return_ml_prop:
        return mld


    # Compute mixed layer properties (thickness-weighted in case of irregular grids)

#   # Compute layer thickness
    dz = depth.diff("deptht")

#   # Reindex to be the same size
    dz = dz.reindex(deptht=depth.deptht, method="pad")

#   # Fill the first level have the same thickness as the second layer. Just a practical approximation, should not have too large effect on overall results.
    dz = dz.fillna(dz.isel(deptht=1))
    
    # Broadcast MLD for comparison
    mld_broadcast = mld.expand_dims({"deptht": depth.deptht}, axis=-3)

    # Mask for mixed layer
    ml_mask = depth <= mld_broadcast

    # Thickness-weighted means
    weights = dz.where(ml_mask)

    sa_ml = (sa * weights).sum("deptht") / weights.sum("deptht")
    ct_ml = (ct * weights).sum("deptht") / weights.sum("deptht")

    return mld, sa_ml, ct_ml

def calc_mld_dask(sa, ct, depth, delta_rho=0.03, ref_depth=10.0, return_ml_prop=False):
    """
    Calculate mixed layer depth using a density threshold relative to a reference depth, and optionally return mixed layer SA and CT. 
    
    NOTE: This assumes that the bottom grid cell is either fully inside or fully outside of the mixed layer. If the MLD cuts through the middle of a cell, this overestimates volume.

    Parameters
    -------
    sa: xarray.DataArray
        Absolute Salinity [depth, y, x] or [time, depth, y, x]
    ct: xarray.DataArray
        Conservative Temperature [depth, y, x] or [time, depth, y, x]
    depth: xarray.DataArray
        Depths corresponding to rho (deptht)
    delta_rho: float
        Density increase from reference depth defining the MLD (kg/m³) (default 0.03 kg/m³) 
    ref_depth: float, optional
        Reference depth for MLD calculation (default: 10 m)
    return_ml_properties: bool
        If True, also return mixed layer mean SA and CT

    Returns
    --
    mld: xarray.DataArray
        Mixed layer depth
    sa_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Absolute Salinity
    ct_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Conservative Temperature   
    """

    # Calculate density
    sigma_0 = xr.apply_ufunc(
        gsw.sigma0,
        sa,
        ct,
        dask="allowed" 
    )

    # Find reference level, default = 10 m
    k_ref = int(abs(depth - ref_depth).argmin().compute()) #dask compatible
    sigma_ref = sigma_0.isel(deptht=k_ref)
    has_ref = sigma_ref.notnull()

    # Index thresholds
    mask = sigma_0 >= (sigma_ref + delta_rho)

    
    # First depth index where threshold is met
    mld = depth.where(mask).min("deptht")  # <-- Dask-safe mask-based method  
    
    # Columns where threshold never met
    never_met = ~mask.any("deptht")

    # Bottom depth
    bottom_depth = depth.where(sa.notnull()).max("deptht")

    # If threshold never met but reference valid → mixed to bottom
    mld = xr.where(never_met & has_ref, bottom_depth, mld)

    # If reference invalid → keep NaN
    mld = mld.where(has_ref)

    if not return_ml_prop:
        return mld


    # Compute mixed layer properties (thickness-weighted in case of irregular grids)

    # Compute thickness between levels
    dz = depth.diff("deptht", label="upper")  # or "lower", depending on your depth orientation
    dz = xr.concat([dz.isel(deptht=0), dz], dim="deptht")  # approximate first layer thickness
    
    # Broadcast MLD for comparison
    mld_broadcast = mld.expand_dims({"deptht": depth.deptht}, axis=-3)

    # Mask for mixed layer
    ml_mask = depth <= mld_broadcast

    # Thickness-weighted means
    weights = dz.where(ml_mask)

    sa_ml = (sa * weights).sum("deptht") / weights.sum("deptht")
    ct_ml = (ct * weights).sum("deptht") / weights.sum("deptht")



def calc_percentiles_region(var, q=[5, 95], dim=None, mask=None, skipna=True):
    """
    Calculate given percentiles of a variable, optionally within a masked region.

    Useful for setting color limits on maps or axis limits for histograms, 
    especially when focusing on a specific region.

    Parameters
    ------
    var: xarray.DataArray or numpy.ndarray
        Variable for which to compute percentiles.
    q: list of float, optional
        List of percentiles to compute (default [5, 95]).
    dim: str or list of str, optional
        Dimension(s) over which to compute percentiles.
        If None, computes over all data.
    mask: xarray.DataArray or numpy.ndarray of bool, optional
        Boolean mask of same shape as var. Only data where mask==True is considered.
    skipna: bool, optional
        If True, ignore NaN values (only relevant for xarray).

    Returns
    -------
    percentiles : dict
        Dictionary mapping each percentile to its value.
        E.g., {5: 0.1, 95: 10.2}
    """

    # Apply mask if provided
    if mask is not None:
        if isinstance(var, xr.DataArray):
            var_masked = var.where(mask)
        else:
            var_masked = np.where(mask, var, np.nan)
    else:
        var_masked = var

    percentiles = {}
    
    if isinstance(var_masked, xr.DataArray):
        for p in q:
            func = np.nanpercentile if skipna else np.percentile
            percentiles[p] = float(var_masked.reduce(func, q=p, dim=dim))
    else:
        for p in q:
            func = np.nanpercentile if skipna else np.percentile
            percentiles[p] = float(func(var_masked, p))
    
    return percentiles


import numpy as np
import xarray as xr
import gsw

def calc_mld(sa, ct, depth, delta_rho=0.03, ref_depth=10.0, return_ml_prop=False):
    """
    Calculate mixed layer depth using a density threshold relative to a reference depth, and optionally return mixed layer SA and CT. 
    
    NOTE: This assumes that the bottom grid cell is either fully inside or fully outside of the mixed layer. If the MLD cuts through the middle of a cell, this overestimates volume.

    Parameters
    -------
    sa: xarray.DataArray
        Absolute Salinity [depth, y, x] or [time, depth, y, x]
    ct: xarray.DataArray
        Conservative Temperature [depth, y, x] or [time, depth, y, x]
    depth: xarray.DataArray
        Depths corresponding to rho (deptht)
    delta_rho: float
        Density increase from reference depth defining the MLD (kg/m³) (default 0.03 kg/m³) 
    ref_depth: float, optional
        Reference depth for MLD calculation (default: 10 m)
    return_ml_properties: bool
        If True, also return mixed layer mean SA and CT

    Returns
    -----
    mld: xarray.DataArray
        Mixed layer depth
    sa_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Absolute Salinity
    ct_ml: xarray.DataArray (optional)
        Thickness-weighted mixed layer mean Conservative Temperature   
    """

    # Calculate density
    sigma_0 = xr.apply_ufunc(
        gsw.sigma0,
        sa,
        ct,
        dask="allowed" 
    )

    # Find reference level, default = 10 m
    k_ref = np.abs(depth - ref_depth).argmin().item()
    sigma_ref = sigma_0.isel(deptht=k_ref)
    has_ref = sigma_ref.notnull()

    # Index thresholds
    mask = sigma_0 >= (sigma_ref + delta_rho)

    # First depth index where threshold is met
    k_mld = mask.argmax("deptht")

    # Columns where threshold never met
    never_met = ~mask.any("deptht")

    # Convert index to depth
    mld = depth.isel(deptht=k_mld)

    # Bottom depth
    bottom_depth = depth.where(sa.notnull()).max("deptht")

    # If threshold never met but reference valid → mixed to bottom
    mld = xr.where(never_met & has_ref, bottom_depth, mld)

    # If reference invalid → keep NaN
    mld = mld.where(has_ref)

    if not return_ml_prop:
        return mld


    # Compute mixed layer properties (thickness-weighted in case of irregular grids)

    # Compute layer thickness
    dz = depth.diff("deptht")

    # Reindex to be the same size
    dz = dz.reindex(deptht=depth.deptht, method="pad")

    # Fill the first level have the same thickness as the second layer. Just a practical approximation, should not have too large effect on overall results.
    dz = dz.fillna(dz.isel(deptht=1))

    # Broadcast MLD for comparison
    mld_broadcast = mld.expand_dims({"deptht": depth.deptht}, axis=-3)

    # Mask for mixed layer
    ml_mask = depth <= mld_broadcast

    # Thickness-weighted means
    weights = dz.where(ml_mask)

    sa_ml = (sa * weights).sum("deptht") / weights.sum("deptht")
    ct_ml = (ct * weights).sum("deptht") / weights.sum("deptht")

    return mld, sa_ml, ct_ml


def calc_percentiles_region(var, q=[5, 95], dim=None, mask=None, skipna=True):
    """
    Calculate given percentiles of a variable, optionally within a masked region.

    Useful for setting color limits on maps or axis limits for histograms, 
    especially when focusing on a specific region.

    Parameters
    ------
    var: xarray.DataArray or numpy.ndarray
        Variable for which to compute percentiles.
    q: list of float, optional
        List of percentiles to compute (default [5, 95]).
    dim: str or list of str, optional
        Dimension(s) over which to compute percentiles.
        If None, computes over all data.
    mask: xarray.DataArray or numpy.ndarray of bool, optional
        Boolean mask of same shape as var. Only data where mask==True is considered.
    skipna: bool, optional
        If True, ignore NaN values (only relevant for xarray).

    Returns
    -------
    percentiles : dict
        Dictionary mapping each percentile to its value.
        E.g., {5: 0.1, 95: 10.2}
    """

    # Apply mask if provided
    if mask is not None:
        if isinstance(var, xr.DataArray):
            var_masked = var.where(mask)
        else:
            var_masked = np.where(mask, var, np.nan)
    else:
        var_masked = var

    percentiles = {}
    
    if isinstance(var_masked, xr.DataArray):
        for p in q:
            func = np.nanpercentile if skipna else np.percentile
            percentiles[p] = float(var_masked.reduce(func, q=p, dim=dim))
    else:
        for p in q:
            func = np.nanpercentile if skipna else np.percentile
            percentiles[p] = float(func(var_masked, p))
    
    return percentiles


def area_weighted_hist(data, e1t, e2t, mask=None, bins=50, range=None, output='count'):
    """
    Compute an area-weighted histogram on ORCA grids.

    Parameters
    ---
    data: xarray.DataArray (2D: y, x)
        Variable to histogram (e.g., surface MLD, ML-SA, ML-CT)
    e1t, e2t : xarray.DataArray (2D: y, x)
        Horizontal grid spacing in meters
    mask: xarray.DataArray (2D: y, x), optional
        Boolean mask: True = include cell
    bins: int
        Number of bins
    range: tuple
        (min, max) range for histogram
    output  str
        'count', 'percent', 'density', 'cumulative'
    """
    
    # Check shapes
    if data.ndim != 2:
        raise ValueError("Data must be 2D (y x x)")
    if data.shape != e1t.shape or data.shape != e2t.shape:
        raise ValueError("e1t and e2t must match data shape")
    
    # Compute true cell area
    area = e1t * e2t
    
    # Mask invalid points
    valid = data.notnull()
    if mask is not None:
        valid = valid & mask
    
    flat_data = data.where(valid).values.ravel()
    flat_weights = area.where(valid).values.ravel()
    
    keep = ~np.isnan(flat_data)
    flat_data = flat_data[keep]
    flat_weights = flat_weights[keep]
    
    hist, bin_edges = np.histogram(
        flat_data,
        bins=bins,
        range=range,
        weights=flat_weights
    )
    
    # Normalize
    if output == 'percent':
        hist = 100 * hist / hist.sum()
    elif output == 'density':
        bin_width = bin_edges[1] - bin_edges[0]
        hist = hist / (hist.sum() * bin_width)
    elif output == 'cumulative':
        hist = np.cumsum(hist)
        hist = hist / hist[-1]
    
    return hist, bin_edges



def moving_mean_1d(series, window, window_type='center', include_nans=False, edge_fill=True):
    """
    Fully vectorized moving mean for 1D time series.
    
    Parameters:
        series (array-like): 1D input series
        window (int): Moving window size
        window_type (str): 'center', 'backward', or 'forward'. Default = True
        include_nans (bool): If True, NaNs are treated as values. If False, they are ignored. Default = False.
        edge_fill (bool): If True, compute partial windows at edges; else edges are NaN. Default = True
    
    Returns:
        np.ndarray: Moving mean, same size as input
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    
    if window < 1:
        raise ValueError("window must be >= 1")
    
    # Mask and replace NaNs if ignoring them
    if include_nans:
        mask = np.ones_like(series, dtype=bool)
        data = series.copy()
    else:
        mask = ~np.isnan(series)
        data = np.where(mask, series, 0)
    
    # Create window of ones
    w = np.ones(window, dtype=float)
    
    # Convolve data and mask
    if window_type == 'center':
        mode = 'same'
        sum_conv = np.convolve(data, w, mode='same')
        count_conv = np.convolve(mask.astype(float), w, mode='same')
    elif window_type == 'backward':
        sum_conv = np.convolve(data, w[::-1], mode='full')  # flip for backward
        count_conv = np.convolve(mask.astype(float), w[::-1], mode='full')
        sum_conv = sum_conv[window-1:window-1+n]
        count_conv = count_conv[window-1:window-1+n]
    elif window_type == 'forward':
        sum_conv = np.convolve(data, w, mode='full')
        count_conv = np.convolve(mask.astype(float), w, mode='full')
        sum_conv = sum_conv[:n]
        count_conv = count_conv[:n]
    else:
        raise ValueError("window_type must be 'center', 'backward', or 'forward'")
    
    # Compute moving mean
    result = np.where(count_conv > 0, sum_conv / count_conv, np.nan)
    
    # Handle edges if edge_fill=False
    if not edge_fill:
        if window_type == 'center':
            half = window // 2
            result[:half] = np.nan
            result[-half:] = np.nan
        elif window_type == 'backward':
            result[:window-1] = np.nan
        else:  # forward
            result[-window+1:] = np.nan
    
    return result
