"""
A simple email sending utility function.

This script provides a straightforward way to send emails via Gmail's SMTP server,
with support for plaintext content and multiple recipients.

Key Features:
- Send emails through Gmail SMTP
- Support for multiple recipients
- Simple interface with clear parameters
- Basic error handling for email delivery issues
"""

from email.message import EmailMessage
import smtplib


def send_email(obj: str, body: str, receivers_emails: list[str] | str, smt_pass: str):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "airflow.automation@fiscozen.it"

    msg = EmailMessage()
    msg["Subject"] = obj
    msg["From"] = sender_email
    msg["To"] = receivers_emails
    msg.set_content(body)

    server = smtplib.SMTP(smtp_server, smtp_port)
    try:
        server.ehlo()
        server.starttls()
        server.login(sender_email, smt_pass)
        server.send_message(msg)
    except Exception as e:
        print(f"Error sending email {e}")
    finally:
        server.quit()


if __name__ == "__main__":
    send_email(
        "prova",
        "PROVAAAAA",
        ["michele.avella@fiscozen.it", "michele.avella.98@gmail.com"],
        "pass",
    )
