import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import numpy as np
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
import pickle  # for loading merged geometries
import shapely.vectorized as sv  # for fast masking


def setup_polar_base(ax, hemisphere='south', 
                   shp_folder="cartopy_data/natural_earth/50m_physical/",
                   lon_lim=(-180, 180), 
                   lat_lim=(-90, -60),                 
                   circular_boundary=True,
                   land_color='lightgrey',
                   ice_shelf_color='white',
                   bathy=False,
                   bathy_resolution="coarse",
                   bathy_levels=[-5000, -4000, -3000, -2000, -1000,-500],
                   bathy_color='black'):
    """
    Configure a matplotlib axis for a polar stereographic projection. Also added option to add bathymetric lines

    Parameters
    -----------
    ax: matplotlib axis
        The axis to configure (must have a polar stereographic projection).
    hemisphere: str
        'north' or 'south' for polar projection. Default = south
    shp_folder: str
        Folder containing Natural Earth shapefiles. 
    lon_lim: tuple
        Longitude extent (min, max) in degrees. Default circumpolar (-180 - 180)
    lat_lim: tuple
        Latitude extent (min, max) in degrees. Default: -90 - -60
    circular_boundary: bool
        If True, sets a circular boundary around the pole. Default: True, circumpolar maps should be round
    land_color: str
        Color for land polygons. Default: Light grey
    ice_shelf_color: str
        Color for ice shelves. Default: White, because ice
    bathy: bool
        If True, adds bathymetry. Default: False
    bathy_resolution: str
        Option to change between coarse, medium, and high resolution for the bathymetry. Default: coarse (for speed, vroom vroom)
    bathy_levels
        Option to change isobaths. Defualt:-5000:1000:-1000,-500
    bathy_color: str
        Option to change the color of the isobaths on your map. Default: black.
    """
    
    # Determine shapefiles based on hemisphere
    if hemisphere.lower() == 'south':
        ice_shp_file = "ne_50m_antarctic_ice_shelves_polys.shp"
        default_lat_lim = (-90, -60)
    elif hemisphere.lower() == 'north':
        ice_shp_file = "ne_50m_antarctic_ice_shelves_polys.shp"  # Could use Arctic-specific shapefile if available
        default_lat_lim = (60, 90)
    else:
        raise ValueError("hemisphere must be 'north' or 'south'")

    # Load shapefiles for land and ice shelves
    land_shp = ShapelyFeature(
        Reader(shp_folder + "ne_50m_land.shp").geometries(),
        ccrs.PlateCarree(),
        facecolor=land_color
    )
    
    ice_shelf_shp = ShapelyFeature(
        Reader(shp_folder + ice_shp_file).geometries(),
        ccrs.PlateCarree(),
        facecolor=ice_shelf_color, edgecolor='none'
    )
            
    # Add shapefiles to axis
    ax.add_feature(land_shp)
    ax.add_feature(ice_shelf_shp)

    # Set extent
    if lat_lim is None:
        lat_lim = default_lat_lim
    ax.set_extent([*lon_lim, *lat_lim], crs=ccrs.PlateCarree())

    # Circular boundary 
    if circular_boundary:
        theta = np.linspace(0, 2*np.pi, 200)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(0.5 * verts + [0.5, 0.5])
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.set_aspect("equal", adjustable="box")


    # Optional bathymetry
    
    if bathy:
        try:
            bathy_files = {
                "coarse": "gebco_antarctic_coarse.nc",
                "medium": "gebco_antarctic_medium.nc",
                "high": "gebco_antarctic_high.nc",
            }
            if bathy_resolution not in bathy_files:
                raise ValueError("Input 'bathy_resolution' must be 'coarse', 'medium', or 'high'")

            
            # Load bathymetry
            gebco_file = f"gebco_2025_sub_ice_topo/{bathy_files[bathy_resolution]}"
            ds = xr.open_dataset(gebco_file)
            bathy_data = ds["elevation"].values
            bathy_lon = ds["lon"].values
            bathy_lat = ds["lat"].values
    
            # Subset to current lon/lat extent
            lon_mask = (bathy_lon >= lon_lim[0]) & (bathy_lon <= lon_lim[1])
            lat_mask = (bathy_lat >= lat_lim[0]) & (bathy_lat <= lat_lim[1])
    
            bathy_sub = bathy_data[np.ix_(lat_mask, lon_mask)]
            lon_sub = bathy_lon[lon_mask]
            lat_sub = bathy_lat[lat_mask]
    
            # Mask bathy under land

            with open(shp_folder + "merged_land.pkl", "rb") as f:
                merged_land = pickle.load(f)
            with open(shp_folder + "merged_ice.pkl", "rb") as f:
                merged_ice = pickle.load(f)

            mask = sv.contains(merged_land, lon_sub[None, :], lat_sub[:, None])
            mask |= sv.contains(merged_ice, lon_sub[None, :], lat_sub[:, None])
            
#            import shapely.vectorized as sv
#            land_shapes = list(Reader(shp_folder + "ne_50m_land.shp").geometries())
#            ice_shapes = list(Reader(shp_folder + "ne_50m_antarctic_ice_shelves_polys.shp").geometries())
#            mask = np.zeros_like(bathy_sub, dtype=bool)
#            # Mask land
#            for geom in land_shapes:
#                mask |= sv.contains(geom, lon_sub[None, :], lat_sub[:, None])
#            # Mask ice shelves
#            for geom in ice_shapes:
#                mask |= sv.contains(geom, lon_sub[None, :], lat_sub[:, None])
#            # Apply mask
            bathy_masked = np.where(mask, np.nan, bathy_sub)
    
            # Plot masked bathymetry
            cs = ax.contour(
                lon_sub, lat_sub, bathy_masked,
                levels=bathy_levels,
                colors=bathy_color,
                linestyles='-',
                linewidths=1,
                transform=ccrs.PlateCarree(),
            )
    
        except Exception as e:
            print("Bathymetry not plotted:", e)

    
    return ax



def setup_polar_grid(ax,
                     hemisphere='south',
                     draw_labels=True,
                     grid_kwargs=None,
                     lon_ticks=np.arange(-180, 181, 60),
                     lat_step=5,
                     label_fontsize=20):
    """
    Configure gridlines and tick labels for a polar stereographic axis.

    Parameters
    ---------
    ax: matplotlib axis
        The axis to configure (must use a polar stereographic projection).
    hemisphere: str
        'north' or 'south' for polar projection. Default = 'south'.
        Determines latitude tick locations.
    draw_labels: bool
        If True, draw longitude and latitude labels. Default = True.
    grid_kwargs: dict or None
        Dictionary of keyword arguments passed to ax.gridlines().
        If None, default dashed black gridlines are used.
    lon_ticks: array-like
        Longitudes at which to draw meridians (degrees).
        Default = every 60 degrees (-180 to 180).
    lat_step: int or float
        Spacing between latitude circles (degrees).
        Default = 5 degrees.
    label_fontsize: int
        Font size of longitude and latitude labels.
        Default = 20.

    Returns
    ----
    gl: cartopy.mpl.gridliner.Gridliner
        The configured Gridliner object.
    """

    hemisphere = hemisphere.lower()

    if grid_kwargs is None:
        grid_kwargs = dict(
            linewidth=1,
            color='black',
            alpha=0.5,
            linestyle='--'
        )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=draw_labels,
        **grid_kwargs
    )

    # Set longitude ticks
    gl.xlocator = mticker.FixedLocator(lon_ticks)

    # Set latitude ticks depending on hemisphere
    if hemisphere == 'south':
        gl.ylocator = mticker.FixedLocator(np.arange(-90, -59, lat_step))
    elif hemisphere == 'north':
        gl.ylocator = mticker.FixedLocator(np.arange(60, 91, lat_step))
    else:
        raise ValueError("hemisphere must be 'north' or 'south'")

    # Format tick labels
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    if draw_labels:
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = True
        gl.right_labels = False

        gl.xlabel_style = {"fontsize": label_fontsize}
        gl.ylabel_style = {"fontsize": label_fontsize}

    return gl


















