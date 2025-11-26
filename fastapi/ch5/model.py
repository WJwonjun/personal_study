from pydantic import BaseModel,constr,Field

class Creature(BaseModel):
    name:str = Field(...,min_length=2)
    country:str
    area:str
    description:str|list
    aka:str

thing = Creature(
    name="yeti",
    country = "CN",
    area = "Himalayas",
    description = "Hirusute",
    aka = "snowman"
)

dragon = Creature(
    name="Dragon",
    description = ["incorrect","string","list"],
    country = "*",
    area="*",
    aka="firedrake"
)