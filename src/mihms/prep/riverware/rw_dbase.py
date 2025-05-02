from pathlib import Path


## functions need to be able to create standard DB and manipulate it (add, update, remove entries)

class RiverWareDb():

    def __init__(self, db_path):
        self.db_location = Path(db_path)
        self.engine =