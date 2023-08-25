import datetime
import os
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

import ocsmesh


# Setup paths
dem_dir = '/path/to/dem'
mesh_path = '/path/to/mesh/file.2dm'
out_dir = pathlib.Path('/path/to/out/dir')
# TODO:
# out_name, format, etc


# Setup parameters
mesh_crs = "epsg:4326"
threshold = -0.5
initial_filter_length = 75 # (meters) can be calcuated from mesh?
acceptable_above_threshold_count = 50
max_multiplier = 8


def interpolate_data():

    # Create a list of all files
    raster_files = list(pathlib.Path(dem_dir).glob('*.tiff'))

    if len(raster_files) == 0:
        raise ValueError("No input DEMs found!")


    # Sort: the last item in the list has highest priority in interpolation
    raster_files_sorted = sorted(
        raster_files, key=lambda i: datetime.datetime.fromtimestamp(i.stat().st_mtime)
    )


    out_dir.mkdir(exist_ok=True, parents=True)
    mesh = ocsmesh.Mesh.open(mesh_path, crs=mesh_crs)

    # Invert elevations of the original mesh to match positive up of Geotiffs
    original_inv_values = -mesh.msh_t.value.copy()


    # Read raster files
    rasters = [ocsmesh.Raster(i) for i in raster_files_sorted]

    # Interpolate on mesh with the more recent file having more priority
    mesh.msh_t.value = original_inv_values.copy()

    mesh.interpolate(rasters, method='linear', info_out_path=out_dir/'interp_info.csv')
    value_interp1 = mesh.msh_t.value.copy()


    filter_multiplier = 1
    iteration_raster_paths = raster_files_sorted
    while True:

        try:
            df_vdatum = pd.read_csv(out_dir/'interp_info.csv', header=None, index_col=0)
            if (df_vdatum[5] > threshold).sum() <= acceptable_above_threshold_count:
                break

            # Second interpolation
            rasters = [ocsmesh.Raster(i) for i in iteration_raster_paths]
            for e, rast in enumerate(rasters):
                print(f"Processing {e+1} of {len(rasters)} raster with"
                      f" f(x) = {filter_multiplier}...")
                center = rast.bbox.centroid.coords[0]
                crs = rast.crs
                side = initial_filter_length
                # assume either geographic or meters
                if crs.is_geographic:
                    side = initial_filter_length / 111000 # convert deg to m
                filter_size = filter_multiplier * int(np.ceil(side / np.sqrt(np.abs(
                        rast.src.transform[0] * rast.src.transform[4]))))
                rast.average_filter(
                    size=filter_size, drop_above=threshold)#, apply_on_bands=[1])



            mesh.msh_t.value = value_interp1.copy() #original_inv_values.copy()
            mesh.interpolate(rasters, method='nearest',
                             info_out_path=out_dir/'interp_info_corr.csv',
                             filter_by_shape=True)
            value_interp2 = mesh.msh_t.value.copy()

            # Combine the two interpolation
            mask = value_interp1.ravel() > threshold
            mask = np.logical_and(
                mask, np.logical_not(value_interp1 == value_interp2).ravel()
            )
            mesh.msh_t.value = value_interp1
            mesh.msh_t.value[mask] = value_interp2[mask]

            # Combine vdatum
            df_vdatum1 = pd.read_csv(out_dir/'interp_info.csv', header=None, index_col=0)
            df_vdatum2 = pd.read_csv(out_dir/'interp_info_corr.csv', header=None, index_col=0)

            # metadata might not include the mask indices
            keys = np.argwhere(mask).ravel()
            keys = df_vdatum1.index.intersection(keys)
            keys = df_vdatum2.index.intersection(keys)
            df_vdatum = df_vdatum1
            df_vdatum.loc[keys, :] = df_vdatum2.loc[keys, :]
            df_vdatum.to_csv(out_dir/'interp_info.csv', header=False)
            os.remove(out_dir/'interp_info_corr.csv')

            print(f"Remaining issue nodes after {filter_multiplier}x filter:",
                  (df_vdatum[5] > threshold).sum())

            if (df_vdatum[5] > threshold).sum() > 0:
                df_0pts = df_vdatum[df_vdatum[5] > threshold]
                gs_0pts = gpd.points_from_xy(df_0pts[3], df_0pts[4])
                gdf_0pts = gpd.GeoDataFrame(geometry=gs_0pts, crs=4326)
                gdf_0pts['elev'] = df_0pts[5].array
                gdf_0pts['node_id'] = df_0pts.index
                gdf_0pts['source'] = df_0pts[7].array
                gdf_0pts.to_file(out_dir/f'error_pts_{filter_multiplier}')
                boxes = []
                side_on_mesh = filter_multiplier * initial_filter_length / 111000 # convert deg to m
                for i in gs_0pts:
                    boxes.append(box(
                        i.coords[0][0] - side_on_mesh / 2,
                        i.coords[0][1] - side_on_mesh / 2,
                        i.coords[0][0] + side_on_mesh / 2,
                        i.coords[0][1] + side_on_mesh / 2
                        ))

                gdf_boxes = gpd.GeoDataFrame(geometry=boxes, crs=4326)
                gdf_boxes.to_file(out_dir/f'error_boxes_{filter_multiplier}')

            if (df_vdatum[5] > threshold).sum() <= acceptable_above_threshold_count:
                break

            filter_multiplier = filter_multiplier * 2
            if filter_multiplier > max_multiplier:
                break

            iteration_raster_paths = np.unique(gdf_0pts['source'].array).tolist()
            value_interp1 = mesh.msh_t.value.copy()

        except KeyboardInterrupt:
            print("Aborted this iteration...")
            print("Writing outputs...")
            break


    # Modify nodes that are above threshold to be equal to threshold
    mask = mesh.msh_t.value > threshold
    mesh.msh_t.value[mask.ravel()] = threshold

    # Invert mesh elevation sign to match the input direction convention
    mesh.msh_t.value = -mesh.msh_t.value
    # Write interpolated mesh to the disk
    mesh.write(out_dir / 'interpolated.2dm', format='2dm', overwrite=True)

    # Read metadata file and update nodes that are above threshold
    df_vdatum = pd.read_csv(out_dir/'interp_info.csv', header=None, index_col=0)
    df_vdatum.loc[df_vdatum[5] > threshold, 5] = threshold
    df_vdatum.to_csv(out_dir/'interp_info.csv', header=False)
    idxs = df_vdatum.iloc[:, 0].array

    # Clip the interpolated mesh based on the index of nodes in the metadatafile
    mesh2 = ocsmesh.utils.clip_mesh_by_vertex(mesh.msh_t, idxs, can_use_other_verts=True)

    # Write clipped mesh to the disk
    ocsmesh.Mesh(mesh2).write(out_dir / 'clipped.2dm', format='2dm', overwrite=True)


if __name__ == '__main__':
    interpolate_data()
