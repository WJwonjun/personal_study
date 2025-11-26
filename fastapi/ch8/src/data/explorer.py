from .init import (conn,curs,IntegrityError)
from model.explorer import Explorer
from error import Missing, Duplicate

curs.execute("""create table explorer(
    name text primary key,
    description text,
    country text,
    area text,
    aka text)""")

def row_to_model(row:tuple)->Explorer:
    name,description,country,area,aka = row

    return Explorer(name=name,description = description, country = country,area= area, aka=aka)

def model_to_dict(explorer:Explorer)->dict:
    return explorer.model_dump() if explorer else None

def get_one(name:str)->Explorer:
    qry = "select * from explorer where name:=name"
    params = {"name":name}
    curs.execute(qry,params)
    row = curs.fetchone()
    if row:
        return row_to_model(row)
    else:
        raise Missing(msg=f"Explorer {name} not found")

def get_all(name:str)->list[Explorer]:
    qry = "select * from explorer"
    curs.execute(qry)
    rows = list(curs.fetchall())
    return [row_to_model(row) for row in rows]

def create(explorer:Explorer):
    if not explorer:
        return None
    qry = "insert into explorer values (:name,:description,:country,:area,:aka)"""
    params = model_to_dict(explorer)
    try:
        curs.execute(qry,params)
    except IntegrityError:
        raise Duplicate(msg = 
                        f"Explorer {explorer.name} already exists")
    return get_one(explorer.name)

def modify(name:str,explorer:Explorer):
    qry = """update explorer
    set country=:country,
    name=:name,
    description=:description,
    area=:area,
    aka=:aka
    where name=:name_orig"""
    params = model_to_dict(explorer)
    params["name_orig"] = explorer.name
    curs.execute(qry,params)
    if curs.rowcount==1:
        return get_one(explorer.name)
    else:
        raise Missing(msg=f"Explorer {name} not found")
    
def replace(explorer:Explorer):
    return explorer

def delete(name:str):
    if not name:
        return False
    qry = "delete from explorer where name = :name"
    params = {"name":name}
    curs.execute(qry,params)
    if curs.rowcount==1:
        raise Missing(msg=f"Explorer {name} not found")

