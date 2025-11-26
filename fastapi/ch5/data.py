from model import Creature

_creatures: list[Creature] = [
    Creature(name="yeti",
             country = "CN",
             area = "Himalayas",
             description = "Hirsute",
             aka = "snowman"
             ),
    Creature(name = "sasquatch",
             country = "US",
             area = "*",
             description = "yeti's cousin",
             aka= "bigfoot")
]

def get_creatures() -> list[Creature]:
    return _creatures