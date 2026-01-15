import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """
    Returns a MySQL connection.
    Adjust credentials if needed.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",          # change if different
            password="",
            database="detection",
            port=3306,
        )
        return conn
    except Error as e:
        print("❌ MySQL connection error:", e)
        raise