from email.message import EmailMessage
import smtplib


def send_email(obj: str, body: str, receivers_emails: list[str], smt_pass: str):
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
