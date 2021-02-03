class BaseMesh:

    @property
    def msh_t(self):
        return self._msh_t

    @property
    def coord(self):
        if self.msh_t.ndims == 2:
            return self.msh_t.vert2['coord']
        elif self.msh_t.ndims == 3:
            return self.msh_t.vert3['coord']
        else:
            raise ValueError(f'Unhandled mesh dimensions {self.msh_t.ndims}.')
