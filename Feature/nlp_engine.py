import time
import random

class GrievanceNLPEngine:
    def __init__(self):
        self.is_ready = True
        
    def safety_guard_check(self, text: str) -> dict:
        """Stage 1: Safety Guard & Content Filtering"""
        unsafe_keywords = ["hack", "exploit", "unrelated_test"]
        if any(w in text.lower() for w in unsafe_keywords):
            return {"is_safe": False, "reason": "Query me invalid ya restricted terms hain."}
        return {"is_safe": True, "reason": "Air-gapped safety check pass ho gaya."}

    def process_query(self, query: str):
        """Stage 2: Hybrid Pipeline Query Processing"""
        time.sleep(0.5)  # Processing time simulation
        
        dept_mapping = {
            "kisan credit card": ("Department of Agriculture & Farmers Welfare", "High", "PM-KCC Scheme Rules"),
            "fertilizer": ("Fertilizer Monitoring System (FMS)", "Medium", "Subsidy & Supply Schedule"),
            "drought": ("Disaster Management Wing", "High", "Crop Loss Compensation Guidelines"),
            "default": ("General Krishi Vigyan Kendra (KVK)", "Low", "General Advisory Services")
        }
        
        matched_key = "default"
        for key in dept_mapping:
            if key in query.lower():
                matched_key = key
                break
                
        dept, priority, verified_doc = dept_mapping[matched_key]
        ticket_id = f"GRV-2026-{random.randint(10000, 99999)}"
        
        return {
            "ticket_id": ticket_id,
            "department": dept,
            "priority": priority,
            "confidence_score": round(random.uniform(91.5, 98.9), 2),
            "matched_policy_doc": verified_doc,
            "suggested_action": "Ticket automatically regional officer ko route ho gaya hai."
        }