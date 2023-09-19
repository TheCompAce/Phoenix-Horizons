import bcrypt
from sqlalchemy.inspection import inspect
from sqlalchemy import ForeignKey, create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, object_session, joinedload, Load, defer
from sqlalchemy.exc import SQLAlchemyError
import logging

Base = declarative_base()
class BaseModel(Base):
    __abstract__ = True

    def to_dict(self, session, depth=1, backref_name=None, load_relationships=True):
        try:
            self = session.merge(self)
            serialized_data = {column.key: getattr(self, column.key)
                            for column in inspect(self).mapper.column_attrs}
            serialized_data["model_type"] = type(self).__name__

            if depth > 0 and load_relationships:
                for relation in inspect(self).mapper.relationships:
                    if backref_name and relation.back_populates == backref_name:
                        continue
                    value = getattr(self, relation.key)
                    if value is None:
                        serialized_data[relation.key] = None
                    elif isinstance(value, list):
                        serialized_data[relation.key] = [item.to_dict(session, depth=depth-1, backref_name=relation.back_populates) for item in value]
                    else:
                        serialized_data[relation.key] = value.to_dict(session, depth=depth-1, backref_name=relation.back_populates)
            return serialized_data
        except Exception as e:
            logging.error(f"Error serializing object to dict: {e}")
            return None

    @staticmethod
    def to_delta_dict(dict_before: dict, dict_after: dict, depth=1) -> dict:
        try:
            delta = {}

            for key in dict_before.keys() & dict_after.keys():
                value_before = dict_before[key]
                value_after = dict_after[key]

                if key in ("id", "model_type"):
                    delta[key] = value_after
                    # continue

                if value_before == value_after:
                    continue

                if isinstance(value_before, dict) and isinstance(value_after, dict) and depth > 0:
                    delta[key] = BaseModel.to_delta_dict(value_before, value_after, depth-1)
                else:
                    delta[key] = value_after

            return delta
        except Exception as e:
            logging.error(f"Error finding delta of dictionaries: {e}")
            return None
        
    @classmethod
    def query(cls, session, filter_by=None, first=False, load_all_relationships=True, exclude=None):
        query = session.query(cls)
        if filter_by:
            query = query.filter_by(**filter_by)
        if load_all_relationships:
            relationships = inspect(cls).relationships
            for relation in relationships:
                load_option = joinedload(getattr(cls, relation.key))
                if exclude and f"{relation.key}" in exclude:
                    for sub_attr in exclude[f"{relation.key}"]:
                        load_option = load_option.defer(sub_attr)
                query = query.options(load_option)
        if first:
            return query.first()
        return query.all()

        
    def save(self, session):
        session = object_session(self)
        if not session:
            session = SessionLocal()
        
        try:
            self = session.merge(self)

            if isinstance(self, User) and 'password' in self.__dict__ and self.__dict__['password']:
                self.set_password(self.__dict__['password'])
            
            obj_in_db = session.query(self.__class__).get(self.id)
            if obj_in_db:
                for attr, value in self.__dict__.items():
                    if attr != "_sa_instance_state":
                        setattr(obj_in_db, attr, value)
            else:
                session.add(self)
            
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error (Model: {type(self).__name__}, ID: {self.id}): {e}")
        except Exception as e:
            session.rollback()
            logging.error(f"An unexpected error occurred (Model: {type(self).__name__}, ID: {self.id}): {e}")
        finally:
            session.close()



class User(BaseModel):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, index=True)
    password = Column(String)
    actor = relationship("Actor", back_populates="user", uselist=False)

    def set_password(self, plain_password):
        self.password = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


    def verify_password(self, plain_password):
        try:
            if self.password:
                return bcrypt.checkpw(plain_password.encode('utf-8'), self.password.encode('utf-8'))
        except ValueError as e:
            logging.error(f"Invalid salt format: {e}")
        return False

class Entity(BaseModel):
    __tablename__ = 'entities'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class InstancedEntity(BaseModel):
    __tablename__ = 'instanced_entities'
    id = Column(Integer, primary_key=True, index=True)  # Adding a primary key for the InstancedEntity
    x = Column(Float, index=True)
    y = Column(Float, index=True)
    entity_id = Column(Integer, ForeignKey('entities.id', ondelete='CASCADE'))  # Foreign key referencing Entity's id
    entity = relationship("Entity")  # ORM relationship to allow access to Entity objects
    actor = relationship("Actor", back_populates="instanced_entity", uselist=False)

class Actor(BaseModel):
    __tablename__ = 'actors'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    instanced_entity_id = Column(Integer, ForeignKey('instanced_entities.id', ondelete='CASCADE'), nullable=False)
    avatar_id = Column(Integer, default=0)

    user = relationship("User", back_populates="actor")
    instanced_entity = relationship("InstancedEntity", back_populates="actor")

engine = create_engine('sqlite:///./server.db')

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init()
