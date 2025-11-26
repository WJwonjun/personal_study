from starlette.applications import Starlette
from Starlette.responses import JSONResponse
from starlette.routing import Route

async def greeting(request):
    return JSONResponse('hello? world?')

app = Starlette(debug=True, routes = [Route('/hi',greeting)])