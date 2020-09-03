import pathlib
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.event import listen
from sqlalchemy.sql import select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, Float, String
from geoalchemy2 import Geometry
from geoalchemy2 import Raster as _Raster


Base = declarative_base()


class Geom(Base):
    __tablename__ = 'geom'
    id = Column(String, primary_key=True, nullable=False)
    tag = Column(String)
    geom = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True
            ),
        nullable=False
        )


class GeomCollection(Base):
    __tablename__ = "geom_collection"
    id = Column(String, primary_key=True, nullable=False)
    source = Column(String)
    zmin = Column(Float)
    zmax = Column(Float)
    driver = Column(String, nullable=False)
    geom = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True
            ),
        nullable=False
        )


class Hfun(Base):
    __tablename__ = 'hfun'
    id = Column(String, primary_key=True, nullable=False)
    hfun = Column(
        Geometry(
            geometry_type='MULTIPOLYGON',
            management=True,
            dimension=3,
            ),
        nullable=False
        )


class HfunCollection(Base):
    __tablename__ = 'hfun_collection'
    id = Column(String, primary_key=True, nullable=False)


class Raster(Base):
    __tablename__ = "rasters"
    raster = Column(_Raster, nullable=True)
    md5 = Column(String, primary_key=True)


class TileIndexRasters(Base):
    __tablename__ = 'tile_index_rasters'
    geom = Column(
        Geometry(
            'POLYGON',
            management=True,
            srid=4326
            ),
        nullable=False)
    # raster = Column(Raster(srid=4326, spatial_index=False))
    url = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=False)
    md5 = Column(String, nullable=False)


def spatialite_session(path, echo=False):

    def _engine(path, echo=False):
        path = pathlib.Path(path)
        _new_db = not path.is_file()
        engine = create_engine(f'sqlite:///{str(path)}', echo=echo)

        def load_spatialite(dbapi_conn, connection_record):
            dbapi_conn.enable_load_extension(True)
            dbapi_conn.load_extension('mod_spatialite')

        listen(engine, 'connect', load_spatialite)
        if _new_db:
            conn = engine.connect()
            conn.execute(select([func.InitSpatialMetaData()]))
            conn.close()
            Geom.__table__.create(engine)
            GeomCollection.__table__.create(engine)
            Hfun.__table__.create(engine)
            TileIndexRasters.__table__.create(engine)
            HfunCollection.__table__.create(engine)

        return engine

    def _session(engine):
        return sessionmaker(bind=engine)()

    return _session(_engine(path, echo))


def postgis_session():
    raise NotImplementedError
