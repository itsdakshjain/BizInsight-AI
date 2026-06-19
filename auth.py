import re
import streamlit as st
from database import get_user_by_username, verify_password, create_user


def is_logged_in():
    return "current_user" in st.session_state and st.session_state["current_user"] is not None


def get_current_user():
    return st.session_state.get("current_user", None)


def logout():
    st.session_state.pop("current_user", None)
    st.session_state.pop("last_upload_hash", None)


def login(username, password):
    user = get_user_by_username(username)
    if not user:
        return None, "No account found with that username."
    if not verify_password(password, user["password_hash"]):
        return None, "Incorrect password."
    return user, None


# ─── Password Complexity Validator ────────────────────────────────────────────
def validate_password_strength(password):
    """
    Validates password strength requirements for new account registrations.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[@#$%&*!]", password):
        return False, "Password must contain at least one special character (e.g., @, #, $, %, &, *, !)."
    return True, None


def register(
    username,
    email,
    password,
    confirm_password,
    workspace_type,
    workspace_id
):
    username = username.strip()
    email = email.strip()
    if not username:
        return False, "Username cannot be empty."

    if not email:
        return False, "Email address cannot be empty."

    email_pattern = r"^[^@]+@[^@]+\.[^@]+$"
    if not re.match(email_pattern, email):
        return False, "Please enter a valid email address."

    is_strong, error_msg = validate_password_strength(password)
    if not is_strong:
        return False, error_msg

    if password != confirm_password:
        return False, "Passwords do not match."
    
    if workspace_type == "corporate":

        if not workspace_id:
            return False, "Corporate ID is required."

    result = create_user(
    username=username,
    email=email,
    password=password,
    workspace_type=workspace_type,
    workspace_id=workspace_id
    )

    if result == "USERNAME_EXISTS":
        return False, "Username already exists."
    if result == "EMAIL_EXISTS":
        return False, "Email already registered."
    if not result:
        return False, "Registration failed."

    return True, None


def show_auth_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown("", unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
        st.title("BizInsight AI")
        st.caption(
            "AI-powered customer intelligence platform for business growth")
        st.markdown("---")

        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input(
                "Username", placeholder="Enter your username", key="login_username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password", key="login_password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    user, error = login(username, password)
                    if error:
                        st.error(error)
                    else:
                        st.session_state["current_user"] = user
                        st.rerun()

        with tab_register:
            new_username = st.text_input(
                "Username",
                placeholder="Choose a username",
                key="reg_username"
            )

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email",
                key="reg_email"
            )

            workspace_type = st.radio(
                "Workspace Type",
                ["Personal", "Corporate"]
            )

            workspace_id = None

            if workspace_type == "Corporate":
                workspace_id = st.text_input(
                    "Corporate Workspace ID",
                    placeholder="Enter company workspace ID"
                )

            new_password = st.text_input(
                "Password",
                type="password",
                placeholder="Min. 8 characters",
                key="reg_password",
                help="Requirements:\n- At least 8 characters\n- One uppercase & one lowercase letter\n- One number\n- One special character (@, #, $, %, &, *, !)"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="reg_confirm"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", use_container_width=True, type="primary"):
                if not new_username or not email or not new_password or not confirm_password:
                    st.error("Please fill in all fields.")
                else:
                    success, error = register(
                        new_username,
                        email,
                        new_password,
                        confirm_password,
                        workspace_type.lower(),
                        workspace_id
                    )

                    if error:
                        st.error(error)
                    else:
                        st.success("Account created successfully! Please log in.")
                        st.session_state["login_username"] = new_username
                        st.rerun()


def show_setup_wizard():
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
        st.title("First Time Setup")
        st.info("No accounts exist yet. Create your admin account to get started.")
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input(
            "Admin Username", placeholder="Choose an admin username")
        email = st.text_input(
            "Admin Email", placeholder="Enter an admin email")
        
        password = st.text_input(
            "Password", 
            type="password", 
            placeholder="Min. 8 characters",
            help="Requirements:\n- At least 8 characters\n- One uppercase & one lowercase letter\n- One number\n- One special character (@, #, $, %, &, *, !)"
        )
        confirm = st.text_input(
            "Confirm Password", type="password", placeholder="Repeat your password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Create Admin Account", use_container_width=True, type="primary"):
            username_clean = username.strip()
            email_clean = email.strip()
            if not username_clean or not email_clean or not password or not confirm:
                st.error("Please fill in all fields.")
            elif not re.match(r"^[^@]+@[^@]+\.[^@]+$", email_clean):
                st.error("Please enter a valid email address.")
            else:
                # Validates complexity, then confirms matching string arrays
                is_strong, error_msg = validate_password_strength(password)
                if not is_strong:
                    st.error(error_msg)
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    success = create_user(username_clean, email_clean, password, role="admin")
                    if success:
                        st.success("Admin account created. You can now log in.")
                        st.rerun()
                    else:
                        st.error("Something went wrong. Try again.")