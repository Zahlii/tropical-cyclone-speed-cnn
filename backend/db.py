from pymongo import MongoClient

DB = MongoClient().get_database('tropical-cyclone')

TBL_STORMS = DB.get_collection('storms')