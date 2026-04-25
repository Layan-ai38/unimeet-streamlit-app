import streamlit as st
import pandas as pd
import sqlite3
import joblib
import re

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="UniMeet",
    page_icon="📅",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F7FBFF;
    color: #0F2747;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #0F2747 !important;
}

p, label, div {
    color: #0F2747;
}

.stTabs [data-baseweb="tab"] {
    background-color: #EAF6FF;
    color: #0F2747;
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    margin-right: 4px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #4DA8FF !important;
    color: white !important;
}

.stButton > button {
    background-color: #0F2747 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6em 1.2em !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 8px rgba(15, 39, 71, 0.25) !important;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #4DA8FF !important;
    color: #FFFFFF !important;
    border: none !important;
}

.stButton > button:hover p,
.stButton > button:hover span,
.stButton > button:hover div {
    color: #FFFFFF !important;
}

.stButton > button:focus,
.stButton > button:active {
    background-color: #0F2747 !important;
    color: #FFFFFF !important;
    border: none !important;
    outline: none !important;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #B9DFFF !important;
    background-color: white !important;
    color: #0F2747 !important;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px !important;
    border: 1px solid #B9DFFF !important;
    background-color: white !important;
    color: #0F2747 !important;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
    padding: 8px;
}

.stAlert {
    border-radius: 10px;
}

.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    border-left: 8px solid #4DA8FF;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #DDF1FF;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    text-align: center;
}

.metric-title {
    color: #4DA8FF;
    font-size: 15px;
    font-weight: 600;
}

.metric-value {
    color: #0F2747;
    font-size: 22px;
    font-weight: bold;
}

.workflow-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #DDF1FF;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    margin-top: 15px;
    margin-bottom: 15px;
}

.example-card {
    background-color: #EAF6FF;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #B9DFFF;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load AI models
# -----------------------------
@st.cache_resource
def load_models():
    intent_model = joblib.load("final_intent_model.pkl")
    urgency_model = joblib.load("final_urgency_model.pkl")
    return intent_model, urgency_model

intent_model, urgency_model = load_models()

# -----------------------------
# Database connection
# -----------------------------
def get_connection():
    return sqlite3.connect("unimeet.db", check_same_thread=False)

conn = get_connection()
cursor = conn.cursor()

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# -----------------------------
# Limited rule-based correction
# -----------------------------
def correct_intent(text, predicted_intent):
    text = clean_text(text)

    cancel_keywords = [
        "cancel",
        "delete my booking",
        "remove my booking",
        "cannot attend",
        "can't attend",
        "cant attend",
        "i cannot come",
        "i can't come",
        "i cant come",
        "no longer need",
        "can't show up",
        "cant show up",
        "cannot show up",
        "i can't show up",
        "i cant show up",
        "i cannot show up",
        "not showing up",
        "will not show up",
        "can't make it",
        "cant make it",
        "cannot make it",
        "won't make it",
        "wont make it"
    ]

    reschedule_keywords = [
        "reschedule",
        "change my appointment",
        "change my meeting",
        "move my appointment",
        "move my meeting",
        "move my appointment to",
        "move my meeting to",
        "another time",
        "another day",
        "different time",
        "different day",
        "different slot",
        "shift my meeting",
        "shift my appointment",
        "postpone"
    ]

    availability_keywords = [
        "available",
        "availability",
        "office hours",
        "free time",
        "open slots",
        "available slots",
        "what times",
        "when is",
        "earliest available"
    ]

    booking_keywords = [
        "book",
        "schedule",
        "appointment",
        "meet",
        "meeting",
        "see dr",
        "see doctor",
        "see my instructor",
        "need to see",
        "want to see",
        "talk to",
        "speak with",
        "discuss"
    ]

    # Priority order matters:
    # cancel/reschedule must be checked before booking because they may contain "appointment".
    if any(keyword in text for keyword in cancel_keywords):
        return "cancel_appointment"

    if any(keyword in text for keyword in reschedule_keywords):
        return "reschedule_appointment"

    if any(keyword in text for keyword in availability_keywords):
        return "check_availability"

    if any(keyword in text for keyword in booking_keywords):
        return "book_appointment"

    return predicted_intent


def correct_urgency_for_booking(text, predicted_urgency):
    text = clean_text(text)

    low_urgency_keywords = [
        "not urgent",
        "no rush",
        "whenever possible",
        "when you have time",
        "later",
        "sometime",
        "when available",
        "just asking"
    ]

    high_urgency_keywords = [
        "urgent",
        "urgently",
        "emergency",
        "asap",
        "as soon as possible",
        "immediate",
        "immediately",
        "now",
        "right now",
        "today",
        "very important",
        "important",
        "critical",
        "serious issue",
        "deadline",
        "final project",
        "final exam",
        "must meet",
        "cannot wait"
    ]

    normal_urgency_keywords = [
        "tomorrow",
        "this week",
        "next week",
        "assignment",
        "project",
        "lecture",
        "course material",
        "need help"
    ]

    # Low first because "not urgent" contains the word "urgent".
    if any(keyword in text for keyword in low_urgency_keywords):
        return "low"

    if any(keyword in text for keyword in high_urgency_keywords):
        return "high"

    if any(keyword in text for keyword in normal_urgency_keywords):
        if predicted_urgency != "high":
            return "normal"

    return predicted_urgency

# -----------------------------
# AI prediction functions
# -----------------------------
def predict_intent(request_text):
    cleaned_text = clean_text(request_text)
    predicted_intent = intent_model.predict([cleaned_text])[0]
    corrected_intent = correct_intent(cleaned_text, predicted_intent)
    return corrected_intent


def predict_urgency_for_booking(request_text):
    cleaned_text = clean_text(request_text)
    predicted_urgency = urgency_model.predict([cleaned_text])[0]
    corrected_urgency = correct_urgency_for_booking(cleaned_text, predicted_urgency)
    return corrected_urgency

# -----------------------------
# Urgency ranking helper
# -----------------------------
def get_urgency_rank(urgency):
    urgency_rank = {
        "low": 1,
        "normal": 2,
        "high": 3
    }
    return urgency_rank.get(str(urgency).lower(), 0)

# -----------------------------
# Database helper functions
# -----------------------------
def get_instructors():
    query = "SELECT instructor_id, name, department FROM instructors"
    return pd.read_sql_query(query, conn)


def get_available_slots(instructor_id):
    query = """
    SELECT 
        availability.slot_id,
        availability.day,
        availability.time
    FROM availability
    WHERE availability.instructor_id = ?
    AND NOT EXISTS (
        SELECT 1
        FROM appointments
        WHERE appointments.instructor_id = availability.instructor_id
        AND appointments.day = availability.day
        AND appointments.time = availability.time
        AND appointments.status = 'booked'
    )
    ORDER BY availability.day, availability.time
    """
    return pd.read_sql_query(query, conn, params=(instructor_id,))


def get_slots_with_status(instructor_id):
    query = """
    SELECT 
        availability.slot_id,
        availability.day,
        availability.time,
        CASE
            WHEN appointments.appointment_id IS NULL THEN 'Available'
            ELSE 'Booked'
        END AS slot_status,
        appointments.student_name AS existing_student,
        appointments.predicted_intent AS existing_intent,
        appointments.predicted_urgency AS existing_urgency
    FROM availability
    LEFT JOIN appointments
    ON availability.instructor_id = appointments.instructor_id
    AND availability.day = appointments.day
    AND availability.time = appointments.time
    AND appointments.status = 'booked'
    WHERE availability.instructor_id = ?
    ORDER BY availability.day, availability.time
    """
    return pd.read_sql_query(query, conn, params=(instructor_id,))


def get_booked_appointments():
    query = """
    SELECT 
        appointments.appointment_id,
        appointments.student_name,
        instructors.name AS instructor_name,
        appointments.day,
        appointments.time,
        appointments.request_text,
        appointments.predicted_intent,
        appointments.predicted_urgency,
        appointments.status
    FROM appointments
    JOIN instructors
    ON appointments.instructor_id = instructors.instructor_id
    ORDER BY appointments.appointment_id
    """
    return pd.read_sql_query(query, conn)


def get_active_appointments_by_student(student_name):
    query = """
    SELECT 
        appointments.appointment_id,
        appointments.student_name,
        instructors.name AS instructor_name,
        appointments.day,
        appointments.time,
        appointments.request_text,
        appointments.predicted_intent,
        appointments.predicted_urgency,
        appointments.status
    FROM appointments
    JOIN instructors
    ON appointments.instructor_id = instructors.instructor_id
    WHERE appointments.status = 'booked'
    AND LOWER(TRIM(appointments.student_name)) = LOWER(TRIM(?))
    ORDER BY appointments.appointment_id
    """
    return pd.read_sql_query(query, conn, params=(student_name,))

# -----------------------------
# Scheduling functions
# -----------------------------
def book_appointment(student_name, instructor_id, day, time, request_text, predicted_urgency):
    predicted_intent = "book_appointment"

    # Rule 1: The selected time must be within the instructor's office hours.
    cursor.execute("""
        SELECT * FROM availability
        WHERE instructor_id = ? AND day = ? AND time = ?
    """, (instructor_id, day, time))

    available_slot = cursor.fetchone()

    if available_slot is None:
        return (
            False,
            "This time is not within the instructor's office hours. Please select a valid office-hour slot."
        )

    # Rule 2: Check whether this instructor slot is already booked.
    cursor.execute("""
        SELECT student_name, predicted_urgency
        FROM appointments
        WHERE instructor_id = ? AND day = ? AND time = ? AND status = 'booked'
    """, (instructor_id, day, time))

    instructor_conflict = cursor.fetchone()

    if instructor_conflict is not None:
        existing_student, existing_urgency = instructor_conflict

        new_priority = get_urgency_rank(predicted_urgency)
        existing_priority = get_urgency_rank(existing_urgency)

        if new_priority > existing_priority:
            message = (
                f"This slot is already booked by another student with a {existing_urgency}-urgency appointment. "
                f"Your request was classified as {predicted_urgency}, which has higher priority. "
                "The system cannot automatically cancel another student's appointment, but it recommends contacting the instructor for priority review or choosing another available slot."
            )

        elif new_priority < existing_priority:
            message = (
                f"This slot is already booked by a {existing_urgency}-urgency appointment, "
                f"which has higher priority than your {predicted_urgency}-urgency request. "
                "The system recommends choosing another available slot."
            )

        else:
            if predicted_urgency == "high":
                message = (
                    "This slot is already booked by another high-urgency appointment. "
                    "The system cannot book two students in the same slot. "
                    "If your case needs immediate attention, please contact the instructor for review or choose another available slot."
                )
            else:
                message = (
                    f"This slot is already booked by another {existing_urgency}-urgency appointment. "
                    "Please choose another available slot."
                )

        return False, message

    # Rule 3: The student cannot have two active appointments at the same time.
    cursor.execute("""
        SELECT * FROM appointments
        WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
        AND day = ? AND time = ? AND status = 'booked'
    """, (student_name, day, time))

    student_conflict = cursor.fetchone()

    if student_conflict is not None:
        return (
            False,
            "You already have another appointment at this time. Please choose a different slot."
        )

    # Rule 4: Save the appointment if all checks pass.
    cursor.execute("""
        INSERT INTO appointments (
            student_name,
            instructor_id,
            day,
            time,
            request_text,
            predicted_intent,
            predicted_urgency,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        instructor_id,
        day,
        time,
        request_text,
        predicted_intent,
        predicted_urgency,
        "booked"
    ))

    conn.commit()

    return True, "Appointment booked successfully."


def cancel_appointment(appointment_id, student_name):
    cursor.execute("""
        UPDATE appointments
        SET status = 'cancelled'
        WHERE appointment_id = ?
        AND status = 'booked'
        AND LOWER(TRIM(student_name)) = LOWER(TRIM(?))
    """, (appointment_id, student_name))

    conn.commit()

    if cursor.rowcount == 0:
        return False, "Appointment not found for this student name or already cancelled."

    return True, "Appointment cancelled successfully."


def reschedule_appointment(appointment_id, student_name, new_day, new_time):
    cursor.execute("""
        SELECT student_name, instructor_id
        FROM appointments
        WHERE appointment_id = ?
        AND status = 'booked'
        AND LOWER(TRIM(student_name)) = LOWER(TRIM(?))
    """, (appointment_id, student_name))

    appointment = cursor.fetchone()

    if appointment is None:
        return False, "Appointment not found for this student name or not active."

    current_student_name, instructor_id = appointment

    cursor.execute("""
        SELECT * FROM availability
        WHERE instructor_id = ? AND day = ? AND time = ?
    """, (instructor_id, new_day, new_time))

    available_slot = cursor.fetchone()

    if available_slot is None:
        return False, "New time is not within the instructor's office hours."

    cursor.execute("""
        SELECT * FROM appointments
        WHERE instructor_id = ? AND day = ? AND time = ? AND status = 'booked'
        AND appointment_id != ?
    """, (instructor_id, new_day, new_time, appointment_id))

    instructor_conflict = cursor.fetchone()

    if instructor_conflict is not None:
        return False, "The new slot is already booked with this instructor. Please choose another available slot."

    cursor.execute("""
        SELECT * FROM appointments
        WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
        AND day = ? AND time = ? AND status = 'booked'
        AND appointment_id != ?
    """, (student_name, new_day, new_time, appointment_id))

    student_conflict = cursor.fetchone()

    if student_conflict is not None:
        return False, "You already have another appointment at the new time."

    cursor.execute("""
        UPDATE appointments
        SET day = ?, time = ?, status = 'booked'
        WHERE appointment_id = ?
        AND LOWER(TRIM(student_name)) = LOWER(TRIM(?))
    """, (new_day, new_time, appointment_id, student_name))

    conn.commit()

    return True, "Appointment rescheduled successfully."

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="info-card">
    <h1 style="margin-bottom: 8px;">📅 UniMeet</h1>
    <h4 style="color:#4DA8FF; margin-top: 0;">
        AI-Powered Smart Campus Appointment Scheduling System
    </h4>
    <p style="font-size:16px;">
        UniMeet helps students manage appointment requests with instructors using two trained machine learning models:
        an <b>Intent Classification Model</b> and an <b>Urgency Classification Model</b>.
        The intent model routes the request to the correct workflow, while the urgency model is used only for booking requests
        to support priority-aware scheduling decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Summary metrics
# -----------------------------
appointments_df_all = get_booked_appointments()
active_appointments_df = appointments_df_all[appointments_df_all["status"] == "booked"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">AI Models</div>
        <div class="metric-value">2</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Appointments</div>
        <div class="metric-value">{len(active_appointments_df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Database</div>
        <div class="metric-value">SQLite</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Smart Request Assistant",
    "Appointments Dashboard",
    "About AI Models"
])

# -----------------------------
# Tab 1: Smart Request Assistant
# -----------------------------
with tab1:
    st.header("Smart Request Assistant")

    st.markdown("""
    <div class="example-card">
        <h3>How to use UniMeet</h3>
        <p>
            Write an appointment-related request in one sentence. The AI intent model will understand your request
            and automatically show the correct workflow.
        </p>
        <ul>
            <li><b>Booking:</b> "I want to book an appointment with Dr. Eman about my project."</li>
            <li><b>Cancellation:</b> "I can't attend my appointment."</li>
            <li><b>Rescheduling:</b> "Move my appointment to another day."</li>
            <li><b>Availability:</b> "When is Dr. Eman available this week?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    student_name = st.text_input("Student Name", placeholder="Enter your name")

    request_text = st.text_area(
        "Your Request",
        placeholder="Example: I want to book an appointment with Dr. Eman about my project."
    )

    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    if st.button("Analyze Request"):
        if not student_name.strip():
            st.error("Please enter the student name.")
            st.session_state.analyzed = False

        elif not request_text.strip():
            st.error("Please enter your request.")
            st.session_state.analyzed = False

        else:
            predicted_intent = predict_intent(request_text.strip())

            st.session_state.analyzed = True
            st.session_state.student_name = student_name.strip()
            st.session_state.request_text = request_text.strip()
            st.session_state.predicted_intent = predicted_intent

    if st.session_state.analyzed:
        predicted_intent = st.session_state.predicted_intent

        st.markdown("### AI Request Analysis")
        st.info(f"Predicted Intent: {predicted_intent}")

        # -------------------------
        # Booking workflow
        # -------------------------
        if predicted_intent == "book_appointment":
            predicted_urgency = predict_urgency_for_booking(st.session_state.request_text)

            st.markdown("""
            <div class="workflow-card">
                <h3>Booking Workflow</h3>
                <p>The request was classified as a booking request. The urgency model is now used to support scheduling priority.</p>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"Predicted Urgency: {predicted_urgency}")

            instructors_df = get_instructors()

            instructor_options = {
                f"{row['name']} - {row['department']}": row["instructor_id"]
                for _, row in instructors_df.iterrows()
            }

            selected_instructor_label = st.selectbox(
                "Select Instructor",
                list(instructor_options.keys()),
                key="booking_instructor"
            )

            selected_instructor_id = instructor_options[selected_instructor_label]
            slots_df = get_slots_with_status(selected_instructor_id)

            if slots_df.empty:
                st.warning("No office-hour slots found for this instructor.")
            else:
                slot_options = {}

                for _, row in slots_df.iterrows():
                    if row["slot_status"] == "Available":
                        label = f"{row['day']} at {row['time']} — Available"
                    else:
                        label = f"{row['day']} at {row['time']} — Booked ({row['existing_urgency']} urgency)"

                    slot_options[label] = (row["day"], row["time"])

                selected_slot_label = st.selectbox(
                    "Select Office-Hour Slot",
                    list(slot_options.keys()),
                    key="booking_slot"
                )

                selected_day, selected_time = slot_options[selected_slot_label]

                if st.button("Confirm Booking"):
                    success, message = book_appointment(
                        student_name=st.session_state.student_name,
                        instructor_id=selected_instructor_id,
                        day=selected_day,
                        time=selected_time,
                        request_text=st.session_state.request_text,
                        predicted_urgency=predicted_urgency
                    )

                    if success:
                        st.success(message)
                    else:
                        st.warning(message)

        # -------------------------
        # Cancel workflow
        # -------------------------
        elif predicted_intent == "cancel_appointment":
            st.markdown("""
            <div class="workflow-card">
                <h3>Cancellation Workflow</h3>
                <p>The request was classified as a cancellation request. The system will only show active appointments that match the entered student name.</p>
            </div>
            """, unsafe_allow_html=True)

            active_appointments = get_active_appointments_by_student(
                st.session_state.student_name
            )

            if active_appointments.empty:
                st.info(
                    "No active appointment was found for this student name. "
                    "Please check the name or book an appointment first."
                )
            else:
                appointment_options = {
                    f"ID {row['appointment_id']} | {row['student_name']} with {row['instructor_name']} on {row['day']} at {row['time']}": row["appointment_id"]
                    for _, row in active_appointments.iterrows()
                }

                selected_cancel_label = st.selectbox(
                    "Select Appointment to Cancel",
                    list(appointment_options.keys()),
                    key="smart_cancel"
                )

                selected_appointment_id = appointment_options[selected_cancel_label]

                if st.button("Confirm Cancellation"):
                    success, message = cancel_appointment(
                        selected_appointment_id,
                        st.session_state.student_name
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        # -------------------------
        # Reschedule workflow
        # -------------------------
        elif predicted_intent == "reschedule_appointment":
            st.markdown("""
            <div class="workflow-card">
                <h3>Rescheduling Workflow</h3>
                <p>The request was classified as a rescheduling request. The system will only show active appointments that match the entered student name.</p>
            </div>
            """, unsafe_allow_html=True)

            active_appointments = get_active_appointments_by_student(
                st.session_state.student_name
            )

            if active_appointments.empty:
                st.info(
                    "No active appointment was found for this student name. "
                    "Please check the name or book an appointment first."
                )
            else:
                appointment_options = {
                    f"ID {row['appointment_id']} | {row['student_name']} with {row['instructor_name']} on {row['day']} at {row['time']}": row["appointment_id"]
                    for _, row in active_appointments.iterrows()
                }

                selected_reschedule_label = st.selectbox(
                    "Select Appointment to Reschedule",
                    list(appointment_options.keys()),
                    key="smart_reschedule"
                )

                selected_appointment_id = appointment_options[selected_reschedule_label]

                appointment_row = active_appointments[
                    active_appointments["appointment_id"] == selected_appointment_id
                ].iloc[0]

                instructor_name = appointment_row["instructor_name"]

                instructor_id_query = """
                SELECT instructor_id FROM instructors
                WHERE name = ?
                """

                instructor_id_df = pd.read_sql_query(
                    instructor_id_query,
                    conn,
                    params=(instructor_name,)
                )

                instructor_id = int(instructor_id_df.iloc[0]["instructor_id"])

                available_slots = get_available_slots(instructor_id)

                if available_slots.empty:
                    st.warning("No available slots for rescheduling.")
                else:
                    slot_options = {
                        f"{row['day']} at {row['time']}": (row["day"], row["time"])
                        for _, row in available_slots.iterrows()
                    }

                    selected_new_slot_label = st.selectbox(
                        "Select New Available Slot",
                        list(slot_options.keys()),
                        key="smart_new_slot"
                    )

                    new_day, new_time = slot_options[selected_new_slot_label]

                    if st.button("Confirm Reschedule"):
                        success, message = reschedule_appointment(
                            selected_appointment_id,
                            st.session_state.student_name,
                            new_day,
                            new_time
                        )

                        if success:
                            st.success(message)
                        else:
                            st.error(message)

        # -------------------------
        # Availability workflow
        # -------------------------
        elif predicted_intent == "check_availability":
            st.markdown("""
            <div class="workflow-card">
                <h3>Availability Workflow</h3>
                <p>The request was classified as an availability inquiry. You can view instructor office-hour slots below.</p>
            </div>
            """, unsafe_allow_html=True)

            instructors_df = get_instructors()

            instructor_options = {
                f"{row['name']} - {row['department']}": row["instructor_id"]
                for _, row in instructors_df.iterrows()
            }

            selected_instructor_label = st.selectbox(
                "Select Instructor to View Slots",
                list(instructor_options.keys()),
                key="availability_instructor"
            )

            selected_instructor_id = instructor_options[selected_instructor_label]
            slots_df = get_slots_with_status(selected_instructor_id)

            if slots_df.empty:
                st.info("No office-hour slots found for this instructor.")
            else:
                st.dataframe(slots_df, width="stretch")

            st.info(
                "If you want to book one of these slots, write a clear booking request such as: "
                "'I want to book an appointment with my instructor.'"
            )

        # -------------------------
        # Unsupported request workflow
        # -------------------------
        elif predicted_intent == "general_inquiry":
            st.markdown("""
            <div class="workflow-card">
                <h3>Unsupported Request</h3>
                <p>
                    The request was classified as outside the supported appointment workflows.
                    No appointment action will be performed.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.warning(
                "This service is currently not available. "
                "UniMeet only supports appointment-related requests such as booking, cancellation, "
                "rescheduling, and availability checking."
            )

            st.info(
                "If your request is related to appointments, please try rephrasing it. "
                "For example: 'I want to book an appointment', 'I can't attend my appointment', "
                "'Move my appointment to another day', or 'When is Dr. Eman available?'"
            )

        else:
            st.warning(
                "The request could not be clearly routed. Please rewrite the request using an appointment-related action."
            )

# -----------------------------
# Tab 2: Dashboard
# -----------------------------
with tab2:
    st.header("Appointments Dashboard")

    st.write(
        "This dashboard displays all appointments stored in the SQLite database. "
        "Booked appointments include the AI-predicted urgency level."
    )

    appointments_df = get_booked_appointments()

    if appointments_df.empty:
        st.info("No appointments found.")
    else:
        st.dataframe(appointments_df, width="stretch")

# -----------------------------
# Tab 3: About AI Models
# -----------------------------
with tab3:
    st.header("About the AI Components")

    st.markdown("""
    UniMeet is a domain-specific AI application for appointment scheduling. It is not a general chatbot.
    It supports appointment-related requests only.

    ### 1. Intent Classification Model
    This model predicts the purpose of the student request.

    Possible outputs:
    - book_appointment
    - cancel_appointment
    - reschedule_appointment
    - check_availability
    - general_inquiry

    The intent model is used as an AI request router. Instead of asking users to manually choose an action,
    the system lets the user write a natural-language request, then automatically shows the correct workflow.

    The general_inquiry class is kept as a safety category. If the user writes something outside the supported appointment workflows,
    the system does not perform any appointment action and asks the user to write an appointment-related request.

    ### 2. Urgency Classification Model
    This model predicts the urgency level of booking requests only.

    Possible outputs:
    - low
    - normal
    - high

    The urgency model is used only after the system confirms that the request is a booking request.

    ### Hybrid Inference Approach
    The trained machine learning models generate the initial predictions.
    A limited rule-based correction layer is applied only for clear cases such as cancellation,
    rescheduling, availability inquiries, and urgent phrases. This improves reliability while keeping
    the trained ML models as the main prediction component.

    ### ML Pipeline
    The machine learning pipeline includes:
    - dataset loading
    - text cleaning
    - TF-IDF vectorization
    - model comparison
    - final Linear SVM training
    - performance evaluation
    - model saving

    ### Non-ML Component
    The scheduling engine uses deterministic rules to:
    - route requests to the correct workflow
    - check instructor office hours
    - prevent instructor time conflicts
    - prevent student time conflicts
    - restrict cancellation and rescheduling to the entered student name
    - compare urgency levels when a booking conflict occurs
    - provide priority-aware recommendations
    - save, cancel, and reschedule appointments

    ### Urgency-Aware Conflict Handling
    If a student selects a slot that is already booked, the system compares the urgency level of the new booking request with the urgency level of the existing appointment.

    - If the new request has higher urgency, the system recommends contacting the instructor for priority review.
    - If the existing appointment has higher urgency, the system recommends choosing another available slot.
    - If both requests have the same urgency, the system does not double-book the slot and recommends choosing another slot or contacting the instructor if needed.

    The system does not automatically cancel another student's appointment. This keeps the scheduling process fair while still using AI predictions to support decision-making.

    ### Model Performance
    The final models were trained using an expanded dataset of 1120 samples.
    - Intent Classification Accuracy: 99.11%
    - Urgency Classification Accuracy: 94.64%
    """)

    st.success("Final selected model: TF-IDF + Linear SVM")
    st.info("Intent Classification Accuracy: 99.11%")
    st.info("Urgency Classification Accuracy: 94.64%")