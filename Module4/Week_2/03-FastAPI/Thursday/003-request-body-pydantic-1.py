from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class UserDetails(BaseModel):
    full_name: str
    age: int | None = None


class User(BaseModel):
    id: int | None = None
    username: str
    details: UserDetails


fake_db = [
    {
        "id": 0,
        "username": "Phuc",
        "details": {
            "full_name": "Quang Phuc",
            "age": 21,
        },
    },
    {
        "id": 1,
        "username": "Ngoc",
        "details": {
            "full_name": "Thanh Ngoc",
            "age": 20,
        },
    },
]


@app.get('/get_user/{user_id}', response_model=User)
def get_user(user_id: int):
    return fake_db[user_id]


@app.post('/create_user/', response_model=User)
def post_user(user: User):
    user.id = len(fake_db)
    fake_db.append(user.model_dump())
    return user


@app.put('/update_user/{user_id}', response_model=User)
def update_user(user_id: int, user: User):
    user.id = user_id
    fake_db[user_id] = user.model_dump()
    return fake_db[user_id]


@app.delete('/delete_user/{user_id}')
def delete_user(user_id: int):
    del fake_db[user_id]
    return {"message": "User deleted successfully"}