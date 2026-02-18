# Plotify
Code and small datasets for plotting polar stereographic maps for the SO

This contains:
- plotify.py, which contains two functions to get nice circumpolar maps of the southern ocean. It pulls data from the two data folders, which are
- cartopy_data, which contains .shp files for land masks and ice shelf masks
- gebco_2025_sub_ice_topo, which contains bathymetric data from GEBCO2025 (only south of 50 S) in three grades of resolution. Idea is to use it for quick plotting of isobaths so all files are much coarser than the full GEBCO dataset. 
