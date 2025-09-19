def google_sheet_tool(complaint_details):
    try:
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
        client = gspread.authorize(creds)
        
        try:
            # Try to open existing sheet
            spreadsheet = client.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            # Create new sheet if it doesn't exist
            print(f"Creating new spreadsheet: {SHEET_NAME}")
            spreadsheet = client.create(SHEET_NAME)
            # Share with yourself (replace with your email)
            spreadsheet.share('your-email@gmail.com', perm_type='user', role='writer')
        
        sheet = spreadsheet.sheet1
        
        # Check if headers exist
        if not sheet.get_all_values():
            headers = ["Ticket ID", "User Name", "Complaint Text", "Severity", "Assigned Agent", "Timestamp"]
            sheet.append_row(headers)
        
        # Prepare the row data
        row_data = [
            complaint_details["ticket_id"],
            complaint_details["user_name"],
            complaint_details["complaint_text"],
            complaint_details["severity"],
            complaint_details["assigned_agent"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        
        sheet.append_row(row_data)
        print(f"Successfully logged ticket {complaint_details['ticket_id']}")
        return True
        
    except Exception as e:
        print(f"Error logging to Google Sheet: {e}")
        return False