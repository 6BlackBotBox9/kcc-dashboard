"""
email_service.py — Multi-channel Email Notification Sender for AgroRunner6
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_ticket_email(to_email, farmer_name, ticket_details):
    """
    Automated Email Notification Sender for AgroRunner6 Grievance & Query Portal.
    Includes farmer query, expert answer (if found in dataset), or KCC toll-free expert helpline.
    """
    SENDER_EMAIL = "ts0151253@gmail.com"
    SENDER_PASSWORD = "rgvi dxfq tyry wamy"  # App Password
    
    ticket_id = ticket_details.get('ticket_id', 'KCC-TICKET')
    subject = f"🎫 [AgroRunner6] Kisan Grievance / Query Confirmation - {ticket_id}"
    
    query_text = ticket_details.get('query_text', '')
    query_found = ticket_details.get('query_found', False)
    matched_q = ticket_details.get('matched_question', '')
    expert_ans = ticket_details.get('expert_answer', '')
    kcc_expert_no = ticket_details.get('kcc_expert_no', '1800-180-1551')
    confidence_score = ticket_details.get('confidence_score', 0.0)
    department = ticket_details.get('department', 'Department of Agriculture & Farmers Welfare')
    priority = ticket_details.get('priority', 'Medium')
    
    # Build Answer or KCC Help section
    if query_found and expert_ans:
        solution_section = f"""
        <div style="background-color: #1e3a1e; padding: 12px; border-left: 4px solid #4caf50; margin: 15px 0; border-radius: 4px;">
            <h4 style="color: #81c784; margin-top: 0;">✅ Verified Solution from KCC Dataset:</h4>
            <p><b>Matched Query:</b> {matched_q}</p>
            <p><b>Expert Recommendation:</b></p>
            <p style="background: #142614; padding: 10px; border-radius: 4px; line-height: 1.5;">{expert_ans}</p>
        </div>
        """
    else:
        solution_section = f"""
        <div style="background-color: #3e2723; padding: 12px; border-left: 4px solid #ff9800; margin: 15px 0; border-radius: 4px;">
            <h4 style="color: #ffb74d; margin-top: 0;">ℹ️ Query Under Review by Agricultural Scientists:</h4>
            <p>Aapki query hamare instant dataset me direct match nahi hui hai. Krishi Vigyan Kendra (KVK) aur Agri-Experts ko escalate kar diya gaya hai.</p>
            <p><b>Direct Kisan Call Center (KCC) Expert Helpline:</b> <span style="font-size: 1.2em; color: #ffeb3b; font-weight: bold;">📞 {kcc_expert_no}</span> (Toll Free, 6:00 AM to 10:00 PM, All Days)</p>
        </div>
        """
        
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0e1117; color: #e0e0e0; padding: 20px;">
        <div style="max-width: 650px; margin: auto; background-color: #1a1d24; border: 1px solid #333; border-radius: 8px; padding: 25px;">
            <div style="background: linear-gradient(135deg, #1b5e20, #2e7d32); padding: 16px; color: white; border-radius: 6px;">
                <h2 style="margin: 0;">🌾 AgroRunner6 - Farmer Grievance & Solution Portal</h2>
            </div>
            
            <p style="margin-top: 20px;">Namaste <b>{farmer_name if farmer_name else 'Kisan Bhai / Citizen'}</b>,</p>
            <p>Aapki query / grievance safaltapoorvak receive aur auto-route kar di gayi hai.</p>
            
            <div style="background-color: #262a33; padding: 12px; border-radius: 5px; margin-bottom: 15px;">
                <b>Your Submitted Query:</b>
                <p style="font-style: italic; color: #cfd8dc;">"{query_text}"</p>
            </div>
            
            <h3>Ticket Details:</h3>
            <ul style="line-height: 1.8;">
                <li><b>Ticket Reference ID:</b> {ticket_id}</li>
                <li><b>Assigned Department:</b> {department}</li>
                <li><b>Priority Level:</b> {priority}</li>
                <li><b>Confidence Score:</b> {confidence_score}%</li>
            </ul>
            
            {solution_section}
            
            <hr style="border: none; border-top: 1px solid #444; margin: 20px 0;">
            <p style="font-size: 0.85em; color: #9e9e9e;">
                <i>This is an automated notification from AgroRunner6 NLP Routing System. For immediate assistance, dial <b>{kcc_expert_no}</b>.</i>
            </p>
        </div>
    </body>
    </html>
    """
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)
