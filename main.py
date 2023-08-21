from fastapi import FastAPI
app = FastAPI()

from typing import Union

@app.get("/", description= "This is my first route path. I am excited to be learning about all this here. Kudos to me")
async def root():
    return{"Message": "Hello world, welcome user"}

@app.post("/")
async def post():
    return{"Message": "Hello world from the post route"}

@app.put("/")
async def put():
    return{"Message": "Hello world from the put route"}