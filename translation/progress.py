from email.mime.text import MIMEText
from tqdm import tqdm
import smtplib
import time

# Special class file to only write to a particular line
class OverwriteFile:
    def __init__(self, filename, idx):
        self.filename = filename
        self.idx = idx

    def write(self, data):
        # Read all lines first
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        # Update the specific line
        if self.idx < len(lines):
            lines[self.idx] = data + "\n"
        else:
            raise IndexError("Index out of range for updating the file")
        # Write back the modified lines
        with open(self.filename, 'w') as f:
            f.writelines(lines)

    def flush(self):
        pass  # No action needed for flush in this context


# Function to send progress mail 
def send_email(subject, body, recipient_email, verbose = False):
    sender_email = ""
    sender_password = ""

    # Create the email
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        if verbose: print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

