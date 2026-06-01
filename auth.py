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


def register(username, password, confirm_password):
    if not username.strip():
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    success = create_user(username, password)
    if not success:
        return False, "Username already taken. Choose a different one."
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
            st.markdown("<br>", unsafe_allow_html=True)
            new_username = st.text_input(
                "Username", placeholder="Choose a username", key="reg_username")
            new_password = st.text_input(
                "Password", type="password", placeholder="Min. 6 characters", key="reg_password")
            confirm_password = st.text_input(
                "Confirm Password", type="password", placeholder="Repeat your password", key="reg_confirm")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", use_container_width=True, type="primary"):
                if not new_username or not new_password or not confirm_password:
                    st.error("Please fill in all fields.")
                else:
                    success, error = register(
                        new_username, new_password, confirm_password)
                    if error:
                        st.error(error)
                    else:
                        st.success("Account created! You can now log in.")


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
        password = st.text_input(
            "Password", type="password", placeholder="Min. 6 characters")
        confirm = st.text_input(
            "Confirm Password", type="password", placeholder="Repeat your password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Create Admin Account", use_container_width=True, type="primary"):
            if not username or not password or not confirm:
                st.error("Please fill in all fields.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                success = create_user(username, password, role="admin")
                if success:
                    st.success("Admin account created. You can now log in.")
                    st.rerun()
                else:
                    st.error("Something went wrong. Try again.")
