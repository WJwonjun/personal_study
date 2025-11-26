from model.user import PublicUser, SignInUser, PrivateUser
from .init import (conn,curs, get_db, IntegrityError)
from error import Missing, Duplicate 

curs.execute("""create table if not exists
             user(
             name text primary key,
             hast text
             )""")
cur.execute("""create table if not exists
            xuser(
            name text primary key,
             hast text
             )""")

def row_to_model(row:tuple, is_public:bool=True)-> PublicUser|PrivateUser:
    name,hash = row
    if is_public:
        return PublicUser(name=name)
    else:
        return PrivateUser(name=name, hash = hash)

def model_to_dict(user: PrivateUser) -> dict:
    return user.model_dump()

def get_one(name:str, is_public:bool=True) -> PublicUser|PrivateUser:
    qry = "select * from user where name=:name"
    params = {"name":name}
    curs.execute(qry,params)
    row = curs.fetchone()
    if row:
        return row_to_model(row, is_public=is_public)
    else:
        raise Missing(msg=f"User {name} not found")

def get_all() -> list[PublicUser]:
    qry = "select * from user"
    curs.execute(qry)
    return [row_to_model(row) for row in curs.fetchall()]

def create(user:PrivateUser, table:str = "user") -> PublicUser:
    qry = f"""insert into {table}
    (name,hash)
    values
    (:name,:hash)"""
    params = model_to_dict(user)
    try:
        curs.execute(qry,params)
    except IntegrityError:
        raise Duplicate(msg=
                        f"{table}: user {user.name} already exists")
    return PublicUser(name =user.name)

def modify(name:str,user:PublicUser)->PublicUser:
    qry=  """update user set
            name=:name
            where name=:name0"""
    params = {
        "name":user.name,
        "name0":name
    }
    curs.execute(qry, params)
    if curs.rowcount==1:
        return get_one(user.name)
    else:
        raise Missing(msg=f"User {name} not found")
    
def delete(name:str)->None:
    user = get_one(name,is_public=False)
    qry = "delete from user where name=:name"
    params = {"name":name}
    curs.execute(qry,params)
    if curs.rowcount!=1:
        raise Missing(msg=f"User {name} not found")
    create(user, table="xuser")