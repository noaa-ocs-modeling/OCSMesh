import pathlib

from pyproj import CRS


def read(path, crs=None):
    sms2dm = dict()
    with open(pathlib.Path(path), 'r') as f:
        f.readline()
        while 1:
            line = f.readline().split()
            if len(line) == 0:
                break
            if line[0] in ['E3T', 'E4Q']:
                if line[0] not in sms2dm:
                    sms2dm[line[0]] = {}
                sms2dm[line[0]].update({
                    line[1]: line[2:]
                    })
            if line[0] == 'ND':
                if line[0] not in sms2dm:
                    sms2dm[line[0]] = {}
                sms2dm[line[0]].update({
                    line[1]: (
                        list(map(float, line[2:-1])), float(line[-1])
                        )
                    })
    if crs is not None:
        sms2dm['crs'] = CRS.from_user_input(crs)
    return sms2dm


def writer(sms2dm, path, overwrite=False):
    path = pathlib.Path(path)
    if path.is_file() and not overwrite:
        msg = 'File exists, pass overwrite=True to allow overwrite.'
        raise Exception(msg)
    with open(path, 'w') as f:
        f.write(to_string(sms2dm))


def to_string(sms2dm):
    return '\n'.join([
        "MESH2D",
        E3T_string(sms2dm),
        E4Q_string(sms2dm),
        ND_string(sms2dm),
    ])


def ND_string(sms2dm):
    assert all(int(id) > 0 for id in sms2dm['ND'])
    lines = []
    for id, (coords, value) in sms2dm['ND'].items():
        lines.append(' '.join([
            'ND',
            f'{int(id):d}',
            f"{coords[0]:<.16E}",
            f"{coords[1]:<.16E}",
            f"{value:<.16E}"
        ]))
    return '\n'.join(lines)


def geom_string(geom_type, sms2dm):
    assert geom_type in ['E3T', 'E4Q', 'E6T', 'E8Q', 'E9Q']
    assert all(int(id) > 0 for id in sms2dm[geom_type])
    f = []
    for id, geom in sms2dm[geom_type].items():
        line = [
            f'{geom_type}',
            f'{id}',
        ]
        for j in range(len(geom)):
            line.append(f"{geom[j]}")
        f.append(' '.join(line))
    if len(f) > 0:
        return '\n'.join(f)
    else:
        return ''


def E3T_string(sms2dm):
    return geom_string('E3T', sms2dm)


def E4Q_string(sms2dm):
    return geom_string('E4Q', sms2dm)
