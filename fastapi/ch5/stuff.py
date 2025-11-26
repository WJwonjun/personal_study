from collections import namedtuple
from dataclasses import dataclass
tuple_thing = ("1","2","3")
list_thing = [1,2,3]
CreatureNamedTuple = namedtuple("CreatureNamedTuple","name,country,area,description,aka")
namedtuple_thing = CreatureNamedTuple("yeti",
                                      "CN","Himalaya","hihi","snowman")


@dataclass
class CreatureDataClass():
    name:str
    country:str
    area:str
    description:str
    aka:str

dataclass_thing = CreatureDataClass("yeti",
                                      "CN","Himalaya","hihi","snowman")





print("name is",dataclass_thing.name)