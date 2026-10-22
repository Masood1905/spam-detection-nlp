# email_bot.py  -  Step 5: a Gmail bot that classifies new emails with the trained model
#
# Credentials are read from a .env file (never written in the code):
#   EMAIL_ADDR=yourbot@gmail.com
#   EMAIL_PASS=your-16-character-app-password
# The Gmail account needs 2-Step Verification and an App Password (myaccount.google.com/apppasswords).
import email
import email.utils
import imaplib
import os
import smtplib
import sys
import time
from email.message import EmailMessage

from dotenv import load_dotenv

from predict import predict

load_dotenv()
EMAIL_ADDR = os.getenv("EMAIL_ADDR")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
CHECK_EVERY_SECONDS = 30

if not EMAIL_ADDR or not EMAIL_PASS:
    sys.exit("EMAIL_ADDR and EMAIL_PASS are missing. Copy .env.example to .env and fill them in.")


def send_reply(to_addr, subject, body):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDR
    msg["To"] = str(to_addr).replace("\n", " ").replace("\r", " ").strip()
    msg["Subject"] = "Re: " + str(subject).replace("\n", " ").replace("\r", " ").strip()
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, 587) as s:
        s.starttls()
        s.login(EMAIL_ADDR, EMAIL_PASS)
        s.send_message(msg)


def extract_body(msg):
    """Plain-text body of an email (empty string if there is none)."""
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload is not None:
                return payload.decode(errors="ignore")
    return ""


def check_inbox():
    imap = imaplib.IMAP4_SSL(IMAP_HOST)
    imap.login(EMAIL_ADDR, EMAIL_PASS)
    imap.select("INBOX")

    _, ids = imap.search(None, "UNSEEN")  # only unread emails
    for num in ids[0].split():
        _, data = imap.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(data[0][1])
        sender = email.utils.parseaddr(msg["From"])[1]
        subject = str(msg["Subject"] or "(no subject)")
        body = extract_body(msg)

        label, confidence = predict(f"{subject} {body}")
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] {sender}: {label.upper()} ({confidence:.1%})")

        # Only reply to messages classified as legitimate. Replying to spam would confirm to
        # spammers that the address is active, and we never reply to ourselves or no-reply senders.
        if label == "ham" and sender and sender.lower() != EMAIL_ADDR.lower() and "noreply" not in sender.lower().replace("-", ""):
            send_reply(sender, subject,
                       f"Hello!\n\nYour message was classified as NOT SPAM (confidence {confidence:.1%}).\n\n"
                       "— Spam Classifier Bot")
            print(f"[{now}] replied to {sender}")
        imap.store(num, "+FLAGS", "\\Seen")  # mark as read

    imap.logout()


if __name__ == "__main__":
    print(f"Bot running for {EMAIL_ADDR}. Checking for new emails every {CHECK_EVERY_SECONDS} seconds (Ctrl + C to stop).")
    while True:
        try:
            check_inbox()
        except Exception as e:  # keep running after temporary network or login errors
            print("error:", e)
        time.sleep(CHECK_EVERY_SECONDS)
