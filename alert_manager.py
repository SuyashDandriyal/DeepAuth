import os
import requests
import smtplib
from email.message import EmailMessage

class AlertManager:
    def __init__(self):
        # In a real application, these would come from environment variables
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", 587))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_pass = os.environ.get("SMTP_PASS", "")

    def dispatch_alert(self, prediction_result, confidence, video_name, destination_email=None, destination_webhook=None):
        """
        Dispatches alerts when a Deepfake is detected.
        In this mock/base integration, it prints to console unless real credentials are provided.
        """
        if prediction_result != "FAKE":
            return
            
        message = f"⚠ *Deepfake Detected*\nConfidence: {confidence}%\nVideo: {video_name}"
        
        # 1. Slack Notification
        self._send_slack(message)
        
        # 2. Email Notification
        if destination_email:
            self._send_email(destination_email, "🚨 Security Alert: Deepfake Detected", message)
            
        # 3. Custom Webhook
        if destination_webhook:
            self._send_webhook(destination_webhook, {
                "event": "deepfake_detected",
                "confidence": confidence,
                "video": video_name
            })
            
    def _send_slack(self, text):
        if self.slack_webhook_url:
            try:
                requests.post(self.slack_webhook_url, json={"text": text})
            except Exception as e:
                print(f"Failed to send Slack alert: {e}")
        else:
            print(f"[MOCK SLACK ALERT] {text}")
            
    def _send_email(self, to_email, subject, body):
        if self.smtp_user and self.smtp_pass:
            try:
                msg = EmailMessage()
                msg.set_content(body)
                msg['Subject'] = subject
                msg['From'] = self.smtp_user
                msg['To'] = to_email
                
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)
            except Exception as e:
                print(f"Failed to send Email alert: {e}")
        else:
            print(f"[MOCK EMAIL ALERT to {to_email}] {subject} - {body}")
            
    def _send_webhook(self, url, payload):
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Failed to trigger webhook: {e}")
        else:
            print(f"[MOCK WEBHOOK to {url}] Payload: {payload}")
