import pathlib

from pyproj import CRS


def read(path, crs=None):
    sms2dm = {}
    with open(pathlib.Path(path), 'r') as f:
        lines = list(map(str.split, f.readlines()))
        ind = 1
        while ind < len(lines):
            line = lines[ind]
            ind = ind + 1
            if len(line) == 0:
                break
            if line[0] in ['E3T', 'E4Q']:
                if line[0] not in sms2dm:
                    sms2dm[line[0]] = {}
                v = []
                if line[0] == 'E3T':
                    v = line[2:5]
                elif line[0] == 'E4Q':
                    v = line[2:6]
                sms2dm[line[0]].update({
                    line[1]: v
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
    data = ['MESH2D']
    E3T = E3T_string(sms2dm)
    if E3T is not None:
        data.append(E3T)
    E4Q = E4Q_string(sms2dm)
    if E4Q is not None:
        data.append(E4Q)
    data.append(ND_string(sms2dm))
    return '\n'.join(data)


def ND_string(sms2dm):
    assert all(int(nd_id) > 0 for nd_id in sms2dm['ND'])
    lines = []
    for nd_id, (coords, value) in sms2dm['ND'].items():
        lines.append(' '.join([
            'ND',
            f'{int(nd_id):d}',
            f"{coords[0]:<.16E}",
            f"{coords[1]:<.16E}",
            f"{value:<.16E}"
        ]))
    return '\n'.join(lines)


def geom_string(geom_type, sms2dm):
    assert geom_type in ['E3T', 'E4Q', 'E6T', 'E8Q', 'E9Q']
    assert all(int(elm_id) > 0 for elm_id in sms2dm[geom_type])
    f = []
    for elm_id, geom in sms2dm[geom_type].items():
        line = [
            f'{geom_type}',
            f'{elm_id}',
        ]
        for j, _ in enumerate(geom):
            line.append(f"{geom[j]}")
        f.append(' '.join(line))
    if len(f) > 0:
        return '\n'.join(f)

    return None


def E3T_string(sms2dm):
    return geom_string('E3T', sms2dm)


def E4Q_string(sms2dm):
    return geom_string('E4Q', sms2dm)
