import logging
import warnings

from ocsmesh.cli.remesh_by_shape_factor import RemeshByShape
from ocsmesh.cli.remesh import RemeshByDEM
from ocsmesh.cli.mesh_upgrader import MeshUpgrader
from ocsmesh.cli.subset_n_combine import SubsetAndCombine

class CmdCli:

    def __init__(self, parser):

        # TODO: Later add non experimental CLI through this class
        self._script_dict = {}

        parser.add_argument("--loglevel")
        scripts_subp = parser.add_subparsers(dest='scripts_cmd')
        for cls in [RemeshByShape, RemeshByDEM, MeshUpgrader, SubsetAndCombine]:
            item = cls(scripts_subp)
            self._script_dict[item.script_name] = item

    def execute(self, args):

        warnings.warn(
            "Scripts CLI is used for experimental new features"
            " and is subject to change.")

        if args.loglevel is not None:
            logging.getLogger().setLevel(
                    getattr(logging, args.loglevel.upper()))
        self._script_dict[args.scripts_cmd].run(args)
