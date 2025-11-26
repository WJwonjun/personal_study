from datetime import timedelta, datetime
import os
from jose import jwt
import bcrypt
from model.user import PublicUser,PrivateUser,SignInUser
import time

if os.getenv("CRYPTID_UNIT_TEST"):
    from fake import user as data
else:
    from data import user as data

SECRET_KEY =""
ALGORITHM = "HS256"

def verify_password(plain:str, hash:str) -> bool:
    password_bytes = plain.encode('utf-8')
    hash_bytes = hash.encode('utf-8')
    is_valid = bcrypt.checkpw(password_bytes,hash_bytes) 
    return is_valid

def get_hash(plain:str)->str:
    password_bytes = plain.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes,salt)
    return hashed_password.decode('etf-8')

def get_jwt_username(token:str)-> str|None:
    try:
        payload = jwt.decode(token,SECRET_KEY, algorithms=[ALGORITHM])
        if not (username:=payload.get("sub")):
            return None
    except jwt.JWTError:
        return None
    return username

def get_current_user(token:str) ->PublicUser|None:
    if not (username:=get_jwt_username(token)):
        return None
    if (user:=lookup_user(username)):
        return user
    return None

def lookup_user(username:str,is_public=True) -> PublicUser|PrivateUser|None:
    if (user:=data.get_one(username,is_public=is_public)):
        return user
    return None

def auth_user(name:str,plain:str) -> PublicUser|PrivateUser|None:
    if not (user:=lookup_user(name,is_public=False)):
        return None
    if not verify_password(plain,user.hash):
        return None
    return user

def create_access_token(data:dict, expires:timedelta|None=None):
    src = data.copy()
    now = time.time() 
    if not expires:
        expires = timedelta(minutes=15)
    src.update({"exp":now+expires})
    encoded_jwt = jwt.encode(src,SECRET_KEY,algorithm=ALGORITHM)
    return encoded_jwt

def get_all()-> list[PublicUser]:
    return data.get_all()

def get_one(name:str)->PublicUser:
    return data.get_one(name)

def create(sign_in_user:SignInUser)->PublicUser:
    user = PrivateUser(name=sign_in_user.name,hash = get_hash(sign_in_user.password))
    return data.create(user)

def modify(name:str,user:PublicUser)->PublicUser:
    return data.modify(name,user)

def delete(name:str)->None:
    return data.delete(name)
