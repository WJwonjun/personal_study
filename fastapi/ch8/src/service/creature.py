from model.creature import Creature
import data.creature as data

def get_all() -> list[Creature]:
    return data.get_all()

def get_one(name) -> Creature:
    return data.get_one(name)

def create(creature:Creature)->Creature:
    return data.create(Creature)

def modify(name:str,creature:Creature)->Creature:
    return data.modify(name,creature)

def replace(name:str,creature:Creature)-> Creature:
    return data.replace(name,creature)

def delete(name:str)->bool:
    return data.delte(name)