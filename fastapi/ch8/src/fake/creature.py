from model.creature import Creature

_creatures = [
    Creature(name="Yeti",
             aka="snowman",
             country="CN",
             area ="Himal",
             description="Hirsute"),
    Creature(name="Bigfoot",
             aka="Cousin",
             country ="US",
             area ="*",
             description = "Sas")
]

def get_all() -> list[Creature]:
    return _creatures

def get_one(name) -> Creature:
    for _creature in _creatures:
        if _creature.name==name:
            return _creature
    return None

def create(explorer:Creature)->Creature:
    return Creature

def modify(name:str,explorer:Creature)->Creature:
    return Creature

def replace(name:str,explorer:Creature)-> Creature:
    return Creature

def delete(name:str)->bool:
    for _creature in _creatures:
        if _creature.name ==name:
            return True
    return None
