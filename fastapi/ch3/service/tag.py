from datetime import datetime
from model.tag import Tag

def create(tag: Tag) ->Tag:
    return tag

def get(tag_str: str) -> Tag:
    return Tag(tag=tag_str, created=datetime.utcnow(),secret="")