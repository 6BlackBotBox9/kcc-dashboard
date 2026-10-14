"""
sms_service.py — Real SMS Notification Service using Textbee API Key
"""
import requests

def send_ticket_sms(to_phone_number, farmer_name, ticket_details):
    """
    Sends automated SMS.
    Includes the farmer's query summary, expert answer (if found in dataset),
    or KCC expert helpline number (if not found in dataset).
    """
    TEXTBEE_API_KEY = "txb_Uh1HhGUidfLouXSrb8I2dYVRyUvx4q1g"
    
    clean_number = str(to_phone_number).strip().replace("+", "")
    if not clean_number.startswith("91"):
        clean_number = "91" + clean_number
    
    formatted_phone = "+" + clean_number
    
    ticket_id = ticket_details.get('ticket_id', 'KCC-TICKET')
    query_text = ticket_details.get('query_text', '')
    query_found = ticket_details.get('query_found', False)
    expert_ans = ticket_details.get('expert_answer', '')
    kcc_expert_no = ticket_details.get('kcc_expert_no', '1800-180-1551')
    
    # Trim query for SMS limits
    short_query = (query_text[:40] + '..') if len(query_text) > 40 else query_text
    
    if query_found and expert_ans:
        # Solution found in dataset
        short_ans = (expert_ans[:70] + '..') if len(expert_ans) > 70 else expert_ans
        sms_text = (
            f"Kisan Seva ({ticket_id}): Namaste {farmer_name if farmer_name else 'Kisan Bhai'}, "
            f"Q: '{short_query}'. "
            f"Hal: {short_ans} "
            f"Dept: {ticket_details.get('department', 'Agri')} - AgroRunner6"
        )
    else:
        # Solution not in dataset -> Provide KCC Expert No
        sms_text = (
            f"Kisan Seva ({ticket_id}): Namaste {farmer_name if farmer_name else 'Kisan Bhai'}, "
            f"Q: '{short_query}'. Query KVK ko escalate ho gayi hai. "
            f"Turant KCC Expert se baat karein Toll-Free: {kcc_expert_no}. - AgroRunner6"
        )

    url = "https://api.textbee.dev/api/v1/gateway/send-sms"
    
    headers = {
        "x-api-key": TEXTBEE_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "recipients": [formatted_phone],
        "message": sms_text
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            return True, "SMS Dispatched successfully via Mobile Gateway!"
        else:
            return False, f"Textbee Error: {response.status_code} - {response.text}"

    except Exception as e:
        return False, f"Gateway Network Error: {str(e)}"
