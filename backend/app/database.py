from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# SQLite needs this specific check_same_thread=False for FastAPI multithreading
# SQLite allows only one thread to use the connection
# FastAPI runs multiple threads for requests
# So check_same_thread = False makes allows multiple threads to use SQLite
engine = create_engine(
    settings.DATABASE_URL,
    connect_args = {"check_same_thread" : False}
)

SessionLocal = sessionmaker(
    autocommit = False, # must manually call db.commit()
    autoflush = False, # SQLAlchemy won't auto push changes
    bind = engine # database to use
)

# Parent class for all database tables
# class User(Base)
Base = declarative_base()

# Dependency Injection Helper (FastAPI system)
def get_db() :
    db = SessionLocal()
    try :
        yield db
    finally :
        db.close()