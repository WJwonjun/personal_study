from model.creature import Creature
from service import creature as code

sample = Creature(
    name = "Yeti",
    country = "CN",
    area = "him",
    description="himalaya",
    aka = "snowman"
)

def test_create():
    resp = code.create(sample)
    assert resp==sample

def test_get_exists():
    resp = code.get_one("Yeti")
    assert resp ==sample
    
def test_get_missing():
    resp = code.get_one("boxturtle")
    assert resp in None