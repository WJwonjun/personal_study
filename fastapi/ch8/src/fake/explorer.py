from model.explorer import Explorer

_explorers = [
    Explorer(name="C",
             country = "FR",
             description = "first one"),
    Explorer(name="N",
             country = "De",
             description = "Second one")

]

def get_all() -> list[Explorer]:
    return _explorers

def get_one(name) -> Explorer:
    for _explorer in _explorers:
        if _explorer.name==name:
            return _explorer
    return None

def create(explorer:Explorer)->Explorer:
    return Explorer

def modify(name:str,explorer:Explorer)->Explorer:
    return Explorer

def replace(name:str,explorer:Explorer)-> Explorer:
    return Explorer

def delete(name:str)->bool:
    for _explorer in _explorers:
        if _explorer.name ==name:
            print(_explorer.name)
            return True
    return None
