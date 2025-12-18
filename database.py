from pymongo import MongoClient
from config import MONGO_URI

# MongoDB connection
mongo = MongoClient(MONGO_URI)
db = mongo.get_database()

# Collections
cars_col = db.get_collection("cars")
users_col = db.get_collection("users")
convos_col = db.get_collection("conversations")
summaries_col = db.get_collection("conversation_summaries")
orders_col = db.get_collection("orders")
failed_writes_col = db.get_collection("failed_writes")
