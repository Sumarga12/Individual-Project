import csv
import random
from datetime import datetime, timedelta

header = [
    "Gender", "Age", "NS1", "IgG", "IgM", "District", "Area", "AreaType", "HouseType", "Outcome", "AgeGroup",
    "Date", "Rainfall_mm", "Temperature_C", "Humidity_%", "Mosquito_Breeding_Sites", "Vector_Control_Activity",
    "Awareness_Campaign", "Hospital_Visits", "Response_Time_Days", "Population_Density", "Travel_History", "Standing_Water_Nearby"
]

districts = ["Rupandehi", "Kaski", "Saptari", "Bara", "Dhanusha", "Chitwan", "Banke", "Jhapa", "Makwanpur", "Kailali", "Surkhet", "Kapilvastu", "Morang", "Dang", "Lalitpur", "Bhaktapur", "Parsa", "Siraha", "Sunsari", "Kathmandu"]
areas = [f"Ward-{i}" for i in range(1, 16)] + ["Patan", "Chabahil", "Baneshwor", "Jawalakhel", "Kalanki", "Maharajgunj", "Koteshwor", "Pulchowk", "Dattatraya", "Kamalbinayak", "Suryabinayak"]
area_types = ["Developed", "Undeveloped"]
house_types = ["Building", "Other"]
outcomes = [0, 1]
age_groups = ["<15", "15-29", "30-44", "45-59", "60+"]
genders = ["Male", "Female"]
yn = ["Yes", "No"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-12-31", "%Y-%m-%d")

with open("data/dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for _ in range(10000):
        age = random.randint(1, 90)
        age_group = (
            "<15" if age < 15 else
            "15-29" if age < 30 else
            "30-44" if age < 45 else
            "45-59" if age < 60 else
            "60+"
        )
        row = [
            random.choice(genders),
            age,
            random.randint(0, 1),  # NS1
            random.randint(0, 1),  # IgG
            random.randint(0, 1),  # IgM
            random.choice(districts),
            random.choice(areas),
            random.choice(area_types),
            random.choice(house_types),
            random.choice(outcomes),
            age_group,
            random_date(start_date, end_date).strftime("%Y-%m-%d"),
            random.randint(50, 200),  # Rainfall_mm
            random.randint(20, 35),   # Temperature_C
            random.randint(60, 95),   # Humidity_%
            random.randint(0, 15),    # Mosquito_Breeding_Sites
            random.choice(yn),        # Vector_Control_Activity
            random.choice(yn),        # Awareness_Campaign
            random.randint(1, 30),    # Hospital_Visits
            random.randint(1, 7),     # Response_Time_Days
            random.randint(500, 5000),# Population_Density
            random.choice(yn),        # Travel_History
            random.choice(yn),        # Standing_Water_Nearby
        ]
        writer.writerow(row)

print("Generated 10,000 rows in data/dataset.csv") 