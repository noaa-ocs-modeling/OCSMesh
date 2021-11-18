import warnings

from ocsmesh.cli.mesh_upgrader import MeshUpgrader
from ocsmesh.cli.remesh import RemeshByDEM
from ocsmesh.cli.remesh_by_shape_factor import RemeshByShape


class CmdCli:

    def __init__(self, parser):

        # TODO: Later add non experimental CLI through this class
        self._script_dict = {}

        scripts_subp = parser.add_subparsers(dest='scripts_cmd')
        for cls in [RemeshByShape, RemeshByDEM, MeshUpgrader]:
            item = cls(scripts_subp)
            self._script_dict[item.script_name] = item

    def execute(self, args):

        warnings.warn(
            "Scripts CLI is used for experimental new features"
            " and is subject to change.")

        self._script_dict[args.scripts_cmd].run(args)
