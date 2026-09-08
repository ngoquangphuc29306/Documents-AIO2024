from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.post("/itmes/", response_model=Item)
async def create_item(item: Item):
    return item

@app.get("/items/", response_model=list[Item])
async def read_items():
    return [
        Item(name="Banh trung thi thap cam", price=45.0),
        Item(name="Banh trung thi dau xanh", price=40.0),
    ]