from fastapi import FastAPI, Depends, Query

app = FastAPI()

# def user_dep(name:str = Query(...),gender:str = Query(...)):
#     return {"name":name,"valid":True}

# @app.get("/user")
# def get_user(user:dict = Depends(user_dep)) -> dict:
#     return user

# @app.method(url,dependencies=[Depends])

def check_dep(name:str=Query(...),gender:str = Query(...)):
    if not name:
        raise

@app.get("/check_user",dependencies = [Depends(check_dep)])
def check_user()->bool:
    return True