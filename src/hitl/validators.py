# src/hitl/validators.py

import re
from typing import List, Tuple, Set

class RiskValidator:
    def __init__(self, blocked_keywords: List[str], sensitive_topics: List[str]):
        self.blocked_keywords = [kw.lower() for kw in blocked_keywords]
        self.sensitive_topics = [st.lower() for st in sensitive_topics]
        
        # Regex to detect emails and IPv4/IPv6 addresses
        self.email_regex = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.ip_regex = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

    def analyze(self, question: str, answer: str) -> Tuple[str, List[str]]:
        """
        Runs rules-based scans across proposed edits.
        Returns:
            Tuple[str, List[str]]: (Assigned Risk Level, List of flags raised)
        """
        flags = []
        combined_text = (question + " " + answer).lower()

        # 1. Check for blocked/malicious system prompt injection keywords
        for word in self.blocked_keywords:
            if word in combined_text:
                flags.append(f"BLOCKED_KEYWORD: '{word}'")

        # 2. Scan for flagged security vectors
        for topic in self.sensitive_topics:
            if topic in combined_text:
                flags.append(f"SENSITIVE_TOPIC_RISK: '{topic}'")

        # 3. Detect PII Leakage (Emails, IPs)
        if self.email_regex.search(combined_text):
            flags.append("PII_DETECTED: EMAIL")
        if self.ip_regex.search(combined_text):
            flags.append("PII_DETECTED: IP_ADDRESS")

        # Determine structural severity
        if any("BLOCKED_KEYWORD" in f for f in flags):
            risk_level = "HIGH"
        elif len(flags) > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return risk_level, flags
