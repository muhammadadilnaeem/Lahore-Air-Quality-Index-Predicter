import requests
import pandas as pd

# Set API key and parameters
api_key = "your api key"
start_date = "2023-06-06"
end_date = "2024-08-05"
city = "Lahore"
country = "PK"

# Set API endpoint and parameters
url = f"https://api.openaq.org/v2/measurements?date_from={start_date}&date_to={end_date}&city={city}&country={country}&limit=10000&api_key={api_key}"

# Send API request and get response
response = requests.get(url)

# Check if response was successful
if response.status_code == 200:
    # Parse JSON response
    data = response.json()

    # Extract measurements from response
    measurements = data["results"]

    # Convert measurements to Pandas DataFrame
    df = pd.DataFrame(measurements)

    # Print or save the data
    print(df)
    df.to_csv("lahore_air_quality_data.csv", index=False)
else:
    print("Error:", response.status_code)