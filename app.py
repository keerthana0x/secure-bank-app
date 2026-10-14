import streamlit as st
import pyotp
import bcrypt
import time
import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import json

st.set_page_config(page_title="SecureBank Portal", page_icon="💳", layout="centered")

DB_FILE = "users_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "auth_stage" not in st.session_state:
    st.session_state.auth_stage = "login"
if "temp_username" not in st.session_state:
    st.session_state.temp_username = ""

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

st.title("💳 SecureBank India - Portal")
st.markdown("---")

db = load_db()

if st.session_state.logged_in_user and st.session_state.logged_in_user in db:
    st.session_state.auth_stage = "dashboard"

# --- SIGN UP FLOW ---
if st.session_state.auth_stage == "signup":
    st.subheader("📝 Create a New Account")
    
    with st.form("signup_form"):
        new_user = st.text_input("Choose Username").strip()
        new_pass = st.text_input("Choose Password", type="password")
        chosen_gesture = st.selectbox("Choose Security Gesture", ["Open Palm", "Fist (Closed Hand)", "Thumbs Up"])
        
        signup_submit = st.form_submit_button("Register Account")
        
        if signup_submit:
            if not new_user or not new_pass:
                st.error("Please fill in all fields.")
            elif new_user in db:
                st.error("Username already exists. Choose another or Sign In.")
            else:
                hashed_pw = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                totp_secret = pyotp.random_base32()
                
                db[new_user] = {
                    "password_hash": hashed_pw,
                    "totp_secret": totp_secret,
                    "gesture": chosen_gesture
                }
                save_db(db)
                st.success("🎉 Account created successfully! Please click below to Sign In.")

    if st.button("← Back to Sign In"):
        st.session_state.auth_stage = "login"
        st.rerun()

# --- LOGIN FLOW: STEP 1 (PASSWORD) ---
elif st.session_state.auth_stage == "login":
    st.subheader("🔐 Sign In")
    
    with st.form("login_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Next: Verify Password")
        
        if login_submit:
            if username in db and bcrypt.checkpw(password.encode('utf-8'), db[username]["password_hash"].encode('utf-8')):
                st.session_state.temp_username = username
                st.session_state.auth_stage = "otp"
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
                
    st.markdown("---")
    st.write("Don't have an account?")
    if st.button("Create an Account (Sign Up)"):
        st.session_state.auth_stage = "signup"
        st.rerun()

# --- LOGIN FLOW: STEP 2 (OTP) ---
elif st.session_state.auth_stage == "otp":
    user = st.session_state.temp_username
    totp = pyotp.TOTP(db[user]["totp_secret"])
    
    st.subheader("🔑 Step 2: OTP Verification")
    st.info(f"💡 Demo OTP Code for **{user}**: **{totp.now()}**")
    
    entered_otp = st.text_input("Enter 6-digit OTP", max_chars=6)
    
    if st.button("Verify OTP"):
        if totp.verify(entered_otp):
            st.session_state.auth_stage = "gesture"
            st.rerun()
        else:
            st.error("❌ Invalid or expired OTP.")

# --- LOGIN FLOW: STEP 3 (GESTURE VERIFICATION) ---
elif st.session_state.auth_stage == "gesture":
    user = st.session_state.temp_username
    target_gesture = db[user]["gesture"]
    
    st.subheader("✋ Step 3: Gesture Biometric Verification")
    st.write("Take a snapshot showing your registered security gesture to complete login.")
    
    picture = st.camera_input("Take photo")
    
    if picture is not None:
        bytes_data = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=1
            )

            with HandLandmarker.create_from_options(options) as landmarker:
                result = landmarker.detect(mp_image)
                verified = False

                if result.hand_landmarks:
                    for hl in result.hand_landmarks:
                        is_open = (hl[8].y < hl[6].y and hl[12].y < hl[10].y and hl[16].y < hl[14].y and hl[20].y < hl[18].y)
                        is_fist = (hl[8].y > hl[6].y and hl[12].y > hl[10].y and hl[16].y > hl[14].y and hl[20].y > hl[18].y)
                        is_thumbs_up = (hl[4].y < hl[3].y and hl[8].y > hl[6].y)

                        if target_gesture == "Open Palm" and is_open:
                            verified = True
                        elif target_gesture == "Fist (Closed Hand)" and is_fist:
                            verified = True
                        elif target_gesture == "Thumbs Up" and is_thumbs_up:
                            verified = True
                        break

                if verified:
                    st.session_state.logged_in_user = user
                    st.session_state.auth_stage = "dashboard"
                    st.success("🎉 Authentication Successful!")
                    st.rerun()
                else:
                    st.error("❌ Gesture mismatch. Please try again with your correct security gesture.")

# --- DASHBOARD ---
elif st.session_state.auth_stage == "dashboard":
    user = st.session_state.logged_in_user
    st.success(f"Welcome back, {user}!")
    st.metric(label="Account Balance", value="₹1,45,250.00", delta="+₹12,400 this month")
    
    st.markdown("### Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send Money via UPI"):
            st.info("UPI Transfer gateway opened securely.")
    with col2:
        if st.button("Download Statement"):
            st.info("Statement generated successfully.")
            
    st.markdown("---")
    if st.button("Log Out"):
        st.session_state.logged_in_user = None
        st.session_state.auth_stage = "login"
        st.rerun()
