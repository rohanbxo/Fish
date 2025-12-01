"""
Streamlit web interface for phishing detection
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
import logging

from src.models.detector import PhishingDetector

# Page config
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-high {
        background-color: #ff4444;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffaa00;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .risk-low {
        background-color: #ffdd00;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-safe {
        background-color: #44ff44;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛡️ Phishing Email Detector")
st.markdown("""
Analyze emails for phishing indicators using advanced AI and NLP techniques.
This tool combines email text analysis with URL risk assessment to provide comprehensive phishing detection.
""")

# Initialize detector with caching to avoid reloading on every interaction
@st.cache_resource
def load_detector():
    """Load and cache the phishing detector"""
    try:
        email_model_path = os.getenv("EMAIL_MODEL_PATH", "models/email_classifier/best_model")
        url_model_path = os.getenv("URL_MODEL_PATH")

        # Check if model exists
        if os.path.exists(email_model_path):
            logging.info(f"Loading model from: {email_model_path}")
        else:
            logging.warning(f"Model not found at {email_model_path}, using base model")
            email_model_path = None

        detector = PhishingDetector(
            email_model_path=email_model_path,
            url_model_path=url_model_path
        )
        return detector, True, None
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        return None, False, str(e)

# Load detector
detector, detector_loaded, error_message = load_detector()

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool uses advanced Natural Language Processing (NLP) and machine learning
    to detect phishing attempts in emails. It analyzes:

    - Email content and language patterns
    - Sender information
    - Embedded URLs and links
    - Urgency indicators
    - Sensitive information requests
    """)

    st.header("Example Phishing Indicators")
    st.markdown("""
    - Urgent language and threats
    - Requests for personal information
    - Suspicious sender domains
    - Mismatched or shortened URLs
    - Poor grammar and spelling
    - Generic greetings ("Dear Customer")
    - Unexpected attachments
    """)

    st.header("Model Status")
    if detector_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error(f"❌ Model Error: {error_message}")

# Main content
tab1, tab2, tab3 = st.tabs(["📧 Analyze Email", "📊 Examples", "ℹ️ Help"])

with tab1:
    st.header("Email Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sender = st.text_input(
            "Sender Email",
            value="",
            placeholder="sender@example.com",
            help="Enter the email address of the sender"
        )
    
    with col2:
        st.write("")  # Spacing
    
    subject = st.text_input(
        "Subject Line",
        value="",
        placeholder="Enter email subject",
        help="The subject line of the email"
    )
    
    body = st.text_area(
        "Email Body",
        value="",
        height=200,
        placeholder="Paste the email content here...",
        help="The full content of the email"
    )
    
    urls = st.text_area(
        "URLs (Optional - one per line)",
        value="",
        height=100,
        placeholder="http://example.com\nhttp://another-url.com",
        help="URLs will be automatically extracted if not provided"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Analysis
    if analyze_button:
        if not body:
            st.error("⚠️ Please enter email body")
        elif not sender:
            st.error("⚠️ Please enter sender email")
        elif not detector_loaded:
            st.error("⚠️ Model not loaded. Please check the error in the sidebar.")
        else:
            with st.spinner("Analyzing email..."):
                try:
                    # Parse URLs
                    url_list = [u.strip() for u in urls.split('\n') if u.strip()]

                    # Use detector directly
                    result = detector.detect(
                        subject=subject,
                        body=body,
                        sender=sender,
                        urls=url_list if url_list else None
                    )

                    st.success("✅ Analysis Complete")
                    st.markdown("---")

                    # Display risk level with color
                    risk_level = result['risk_level']
                    risk_score = result['risk_score']

                    if "HIGH" in risk_level:
                        st.markdown(f'<div class="risk-high">🚨 {risk_level}</div>', unsafe_allow_html=True)
                    elif "MEDIUM" in risk_level:
                        st.markdown(f'<div class="risk-medium">⚠️ {risk_level}</div>', unsafe_allow_html=True)
                    elif "LOW" in risk_level:
                        st.markdown(f'<div class="risk-low">⚡ {risk_level}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-safe">✅ {risk_level}</div>', unsafe_allow_html=True)

                    st.markdown("")

                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Overall Risk Score", f"{risk_score:.2%}")

                    with col2:
                        st.metric("Email Score", f"{result['email_score']:.2%}")

                    with col3:
                        if result['url_scores']:
                            max_url_score = max(result['url_scores'])
                            st.metric("Max URL Score", f"{max_url_score:.2%}")
                        else:
                            st.metric("URLs Detected", "0")

                    # Progress bar for risk
                    st.progress(risk_score)

                    st.markdown("---")

                    # Indicators
                    if result.get('indicators'):
                        st.subheader("🔍 Detected Indicators")
                        for indicator in result['indicators']:
                            st.warning(f"• {indicator}")

                    # Detailed analysis
                    st.subheader("📊 Detailed Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Email Analysis:**")
                        email_analysis = result['explanation']['email_analysis']
                        st.json(email_analysis)

                    with col2:
                        st.write("**URL Analysis:**")
                        url_analysis = result['explanation']['url_analysis']
                        st.json(url_analysis)

                    # Recommendations
                    st.subheader("💡 Recommendations")
                    for rec in result['explanation']['recommendations']:
                        st.info(f"• {rec}")

                    # Raw response
                    with st.expander("📋 View Full Response"):
                        st.json(result)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logging.error(f"Detection error: {e}", exc_info=True)

with tab2:
    st.header("Example Emails")
    
    st.subheader("🚨 Phishing Example")
    
    phishing_example = {
        "sender": "security@paypa1-verify.com",
        "subject": "URGENT: Your Account Will Be Suspended",
        "body": """Dear Valued Customer,

We have detected unusual activity on your PayPal account. Your account will be permanently suspended within 24 hours unless you verify your information immediately.

Click here to verify your account: http://paypal-secure-verify.tk/login

Failure to verify will result in:
- Account suspension
- Loss of funds
- Legal action

This is an automated message. Do not reply.

PayPal Security Team
"""
    }
    
    if st.button("Load Phishing Example"):
        st.code(f"Sender: {phishing_example['sender']}\nSubject: {phishing_example['subject']}\n\n{phishing_example['body']}")
    
    st.markdown("---")
    
    st.subheader("✅ Legitimate Example")
    
    legitimate_example = {
        "sender": "noreply@github.com",
        "subject": "Your GitHub Weekly Digest",
        "body": """Hi there,

Here's what happened in your repositories this week:

- 5 new stars on your project
- 2 pull requests merged
- 1 new contributor

View your full activity: https://github.com/notifications

Thanks for being part of GitHub!

The GitHub Team
"""
    }
    
    if st.button("Load Legitimate Example"):
        st.code(f"Sender: {legitimate_example['sender']}\nSubject: {legitimate_example['subject']}\n\n{legitimate_example['body']}")

with tab3:
    st.header("Help & Information")
    
    st.subheader("How It Works")
    st.markdown("""
    1. **Enter Email Details**: Provide the sender, subject, and body of the email
    2. **Add URLs** (optional): Include any URLs found in the email
    3. **Click Analyze**: The system will process the email using AI models
    4. **Review Results**: Get a risk score and detailed analysis
    
    The system uses:
    - **DistilBERT** transformer model for email text analysis
    - **Random Forest** classifier for URL risk assessment
    - Combined scoring algorithm for overall risk evaluation
    """)
    
    st.subheader("Understanding Risk Scores")
    st.markdown("""
    - **HIGH (0.8 - 1.0)**: Very likely phishing - Do not interact
    - **MEDIUM (0.5 - 0.8)**: Suspicious - Verify sender before acting
    - **LOW (0.3 - 0.5)**: Potentially suspicious - Exercise caution
    - **SAFE (0.0 - 0.3)**: Likely legitimate - Still verify important requests
    """)

    st.subheader("Technical Details")
    st.markdown("""
    This application uses:
    - **DistilBERT** transformer model for email text classification
    - **Random Forest** classifier for URL risk assessment
    - **Feature extraction** for urgency indicators and sensitive information requests
    - All processing is done locally without external API calls
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Phishing Detection System v1.0.0 | Built with ❤️ using Streamlit and FastAPI
</div>
""", unsafe_allow_html=True)
