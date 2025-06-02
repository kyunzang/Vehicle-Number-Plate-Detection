import mysql.connector

def get_user_details(number_plate):
    # Set up database connection
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Use your MySQL username
        password="",  # Use your MySQL password
        database="car"
    )
    cursor = conn.cursor(dictionary=True)

    # Query to fetch user details based on the number plate
    query = f"SELECT * FROM car_owners WHERE number_plate = %s"
    cursor.execute(query, (number_plate,))
    
    # Fetch results
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result
