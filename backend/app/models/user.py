from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.database import Base

class User(Base) :
    __tablename__ = "users"

    # Not using String UUID for ID for low scale project
    # id = Column(String, primary_key = True, default = lambda : str(uuid.uuid4()))
    id = Column(Integer, primary_key = True)
    
    email = Column(String(255), unique = True, index = True, nullable = False)
    google_id = Column(String(255), unique = True, nullable = False)
    username = Column(String(255), nullable = True)

    # "default" parameter runs in Python
    # server_default runs in database
    # As a result, database always sets timestamp
    created_at = Column(DateTime(timezone = True), server_default = func.now())
    last_login = Column(DateTime(timezone = True), onupdate = func.now())