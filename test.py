import gspread
from google.oauth2.service_account import Credentials

SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials.json"
SHEET_NAME = "ticketing_ai"

try:
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
    print("Credentials loaded successfully.")
    client = gspread.authorize(creds)
    print("Google Sheets client authorized successfully.")
    print("Attempting to open the Google Sheet..."+SHEET_NAME)
    spreadsheet = client.open(SHEET_NAME)
    print(spreadsheet.id)
    print("Successfully connected to Google Sheets!")
    print(f"Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
except Exception as e:
    print(f"Error: {e}")