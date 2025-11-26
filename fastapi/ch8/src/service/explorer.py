from model.explorer import Explorer
import data.explorer as data

def get_all() -> list[Explorer]:
    return data.get_all()

def get_one(name) -> Explorer|None:
    return data.get_one(name)

def create(explorer:Explorer)->Explorer:
    return data.create(explorer)

def modify(name:str,explorer:Explorer)->Explorer:
    return data.modify(name,explorer)

def replace(name:str,explorer:Explorer)-> Explorer:
    return data.replace(name,explorer)

def delete(name:str)->bool:
    return data.delte(name)