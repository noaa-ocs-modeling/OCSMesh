#! /usr/bin/env python

import argparse

from ocsmesh.ops import combine_geometry, combine_hfun
from ocsmesh.cli.cli import CmdCli


class OCSMesh:

    def __init__(self, args, ocsmesh_cli):
        self._args = args
        self._cli = ocsmesh_cli

    def main(self):

        if self._args.command == 'geom':

            nprocs = self._args.nprocs
            if self._args.geom_nprocs:
                nprocs = self._args.geom_nprocs
            nprocs = -1 if nprocs is None else nprocs

            if self._args.geom_cmd == "build":
                arg_dict = dict(
                    dem_files=self._args.dem,
                    out_file=self._args.output,
                    out_format=self._args.output_format,
                    mesh_file=self._args.mesh,
                    ignore_mesh_final_boundary=self._args.ignore_mesh_boundary,
                    zmin=self._args.zmin,
                    zmax=self._args.zmax,
                    chunk_size=self._args.chunk_size,
                    overlap=self._args.overlap,
                    nprocs=nprocs,
                    out_crs=self._args.output_crs,
                    base_crs=self._args.mesh_crs)
                combine_geometry(**arg_dict)

        elif self._args.command == 'hfun':

            nprocs = self._args.nprocs
            if self._args.hfun_nprocs:
                nprocs = self._args.hfun_nprocs
            nprocs = -1 if nprocs is None else nprocs

            if self._args.hfun_cmd == "build":
                arg_dict = dict(
                    dem_files=self._args.dem,
                    out_file=self._args.output,
                    out_format=self._args.output_format,
                    mesh_file=self._args.mesh,
                    hmin=self._args.hmin,
                    hmax=self._args.hmax,
                    contours=self._args.contour,
                    constants=self._args.constants,
                    chunk_size=self._args.chunk_size,
                    overlap=self._args.overlap,
                    method=self._args.method,
                    nprocs=nprocs)
                combine_hfun(**arg_dict)

        elif self._args.command == 'scripts':
            self._cli.execute(self._args)


def create_parser():
    common_parser = argparse.ArgumentParser()

    common_parser.add_argument("--log-level", choices=["info", "debug", "warning"])
    common_parser.add_argument(
        "--nprocs", type=int, help="Number of parallel threads to use when "
        "computing geom and hfun.")
    common_parser.add_argument(
        "--geom-nprocs", type=int, help="Number of processors used when "
        "computing the geom, overrides --nprocs argument.")
    common_parser.add_argument(
        "--hfun-nprocs", type=int, help="Number of processors used when "
        "computing the hfun, overrides --nprocs argument.")
    common_parser.add_argument(
        "--chunk-size",
        help='Size of square window to be used for processing the raster')
    common_parser.add_argument(
        "--overlap",
        help='Size of overlap to be used for between raster windows')

    sub_parse_common = {
            'parents': [common_parser],
            'add_help': False
    }

    parser = argparse.ArgumentParser(**sub_parse_common)
    subp = parser.add_subparsers(dest='command')

    geom_parser = subp.add_parser(
        'geom', **sub_parse_common,
        help="Perform operations related to domain creation and modification.")
    geom_subp = geom_parser.add_subparsers(dest='geom_cmd')
    geom_bld = geom_subp.add_parser(
        'build', **sub_parse_common,
        help="Build command for domain definition")
    geom_bld.add_argument('-o', '--output', required=True)
    geom_bld.add_argument('-f', '--output-format', default="shapefile")
    geom_bld.add_argument('--output-crs', default="EPSG:4326")
    geom_bld.add_argument('--mesh', help='Mesh to extract hull from')
    geom_bld.add_argument(
        '--ignore-mesh-boundary', action='store_true',
        help='Flag to ignore mesh boundary for final boundary union')
    geom_bld.add_argument(
        '--mesh-crs', help='CRS of the input base mesh (overrides)')
    geom_bld.add_argument(
        '--zmin', type=float,
        help='Maximum elevation to consider')
    geom_bld.add_argument(
        '--zmax', type=float,
        help='Maximum elevation to consider')
    geom_bld.add_argument(
        'dem', nargs='+',
        help='Digital elevation model list to be used in geometry creation')

    hfun_parser = subp.add_parser(
        'hfun', **sub_parse_common,
        help="Perform operations related to size function creation and modification.")
    hfun_subp = hfun_parser.add_subparsers(dest='hfun_cmd')
    hfun_bld = hfun_subp.add_parser(
        'build', **sub_parse_common,
        help="Build command for mesh size definition")
    hfun_bld.add_argument('-o', '--output', required=True)
    hfun_bld.add_argument('-f', '--output-format', default="2dm")
    hfun_bld.add_argument('--mesh', help='Base mesh size function')
    hfun_bld.add_argument(
        '--hmax', type=float, help='Maximum element size')
    hfun_bld.add_argument(
        '--hmin', type=float, help='Minimum element size')
    hfun_bld.add_argument(
        '--contour', action='append', nargs='+', type=float, default=[],
        help="Each contour's (level, [expansion, target])"
             " to be applied on all size functions in collector")
    hfun_bld.add_argument(
        '--constant',
        action='append', nargs=2, type=float, dest='constants',
        metavar='CONST_DEFN', default=[],
        help="Specify constant mesh size above a given contour level"
             " by passing (lower_bound, target_size) for each constant")
    hfun_bld.add_argument(
        '--method', type=str, default='exact',
        help='Method used to calculate size function ({exact} |fast)')
    hfun_bld.add_argument(
        'dem', nargs='+',
        help='Digital elevation model list to be used in size function creation')

    # Scripts don't use common arguments as they are standalon code
    scripts_parser = subp.add_parser(
        'scripts',
        help='Access to experimental OCSMesh scripts.')
    cmd_cli = CmdCli(scripts_parser)

    return parser, cmd_cli

def dummy_documentation():
    parser, ocsmesh_cli = create_parser()
    return parser


def main():
    parser, ocsmesh_cli = create_parser()
#    logger.init(args.log_level)
    OCSMesh(parser.parse_args(), ocsmesh_cli).main()


if __name__ == '__main__':
    main()
