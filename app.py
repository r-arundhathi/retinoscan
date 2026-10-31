import tempfile
import os
import io
import base64
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template, send_file
from fpdf import FPDF

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(r"C:\retina_project")

MODEL_PATH = ROOT_DIR / "models_grouped" / "best_model_finetuned.keras"
VALIDATOR_MODEL_PATH = ROOT_DIR / "image_validation_dataset" / "models" / "best_validator.keras"

VALIDATOR_IMG_SIZE = 160
VALIDATOR_THRESHOLD = 0.5
CLASS_NAMES_PATH = ROOT_DIR / "outputs_grouped" / "class_names.json"

IMG_SIZE = 224
UNCERTAIN_THRESHOLD = 55.0
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DISEASE_INFO = {
    "anemia": {
        "label": "Anemia",
        "icon": "Blood",
        "color": "#e74c3c",
    },
    "diabeties": {
        "label": "Diabetes",
        "icon": "Medical",
        "color": "#e67e22",
    },
    "heart_disease": {
        "label": "Heart Disease",
        "icon": "Heart",
        "color": "#c0392b",
    },
    "hypertension": {
        "label": "Hypertension",
        "icon": "Alert",
        "color": "#9b59b6",
    },
    "stroke": {
        "label": "Stroke",
        "icon": "Brain",
        "color": "#2980b9",
    },
    "normal": {
        "label": "Normal",
        "icon": "Check",
        "color": "#2ecc71",
    },
}

app = Flask(__name__)
@app.template_filter("ist_datetime")
def ist_datetime(value):
    if value is None:
        return "-"

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    ist = timezone(timedelta(hours=5, minutes=30))
    return value.astimezone(ist).strftime("%d-%m-%Y %H:%M")

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["SECRET_KEY"] = os.environ.get(
    "RETINA_SECRET_KEY",
    "change-this-to-a-long-random-secret"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///retina_portal.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

csrf = CSRFProtect(app)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

model = tf.keras.models.load_model(str(MODEL_PATH))
if not VALIDATOR_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Validator model not found: {VALIDATOR_MODEL_PATH}"
    )

validator_model = tf.keras.models.load_model(
    str(VALIDATOR_MODEL_PATH)
)

print("Retinal image validator loaded successfully.")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

print("Model loaded successfully.")
print("Classes:", CLASS_NAMES)

# --------------------------------------------------
# USER DATABASE MODEL
# --------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(120), nullable=False)
    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = db.Column(db.String(255), nullable=False)

    role = db.Column(db.String(20), nullable=False)
    active = db.Column(db.Boolean, default=True, nullable=False)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_active(self):
        return self.active


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
# --------------------------------------------------
# PATIENT PROFILE
# --------------------------------------------------

class PatientProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        unique=True,
        nullable=False
    )

    patient_code = db.Column(
        db.String(30),
        unique=True,
        nullable=False,
        index=True
    )

    phone = db.Column(db.String(20))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(30))

    user = db.relationship("User", backref="patient_profile")


# --------------------------------------------------
# DOCTOR PROFILE
# --------------------------------------------------

class DoctorProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        unique=True,
        nullable=False
    )

    phone = db.Column(db.String(20))
    specialization = db.Column(db.String(120))
    qualification = db.Column(db.String(120))
    license_number = db.Column(db.String(100))
    clinic_name = db.Column(db.String(150))
    address = db.Column(db.Text)

    user = db.relationship("User", backref="doctor_profile")
    
    # --------------------------------------------------
# PATIENT REPORT MODEL
# --------------------------------------------------

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("patient_profile.id"),
        nullable=False,
        index=True
    )

    doctor_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    report_pdf = db.Column(db.LargeBinary, nullable=False)
    original_image = db.Column(db.LargeBinary, nullable=True)
    doctor_notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    patient = db.relationship(
        "PatientProfile",
        backref=db.backref("reports", lazy=True)
    )

    doctor = db.relationship(
        "User",
        foreign_keys=[doctor_id],
        backref="created_reports"
    )
    # --------------------------------------------------
# UNIQUE PATIENT ID GENERATOR
# --------------------------------------------------

import secrets
import string


def generate_patient_code():
    while True:
        code = "PAT-" + "".join(
            secrets.choice(string.ascii_uppercase + string.digits)
            for _ in range(8)
        )

        existing = PatientProfile.query.filter_by(
            patient_code=code
        ).first()

        if not existing:
            return code
LAST_RESULT = {}

# --------------------------------------------------
# UPLOAD VALIDATION
# --------------------------------------------------

def validate_upload(file):
    """
    Accept only JPG/JPEG/PNG files.
    Verify both the extension and actual image content.
    """

    filename = file.filename or ""

    if not filename.strip():
        return None, "Please select an image to upload."

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return None, (
            "Unsupported file type. Please upload only "
            "JPG, JPEG, or PNG retinal images."
        )

    try:
        file_bytes = file.read()

        if not file_bytes:
            return None, "The uploaded file is empty."

        if len(file_bytes) > MAX_FILE_SIZE:
            return None, "Image size must be 10 MB or less."

        image_stream = io.BytesIO(file_bytes)

        # Verify the file is a genuine supported image.
        with Image.open(image_stream) as img:
            if img.format not in {"JPEG", "PNG"}:
                return None, (
                    "Invalid file content. Please upload a genuine "
                    "JPG or PNG image."
                )

            img.verify()

        # Reopen the image after verification.
        image_stream.seek(0)

        with Image.open(image_stream) as img:
            img.load()
            pil_image = img.convert("RGB")

        return pil_image, None

    except (UnidentifiedImageError, OSError, ValueError):
        return None, (
            "The file is not a valid image. "
            "Please upload a JPG or PNG image."
        )
    except Exception:
        return None, (
            "Unable to read the uploaded file. "
            "Please choose another JPG or PNG image."
        )


def looks_like_retina_image(pil_image):
    """
    Basic image validation only.
    This does not confirm that an image is a retinal fundus image.
    """
    if pil_image is None:
        return False

    width, height = pil_image.size
    return width >= 128 and height >= 128

def validate_retinal_classifier(pil_image):
    image = pil_image.convert("RGB").resize(
        (VALIDATOR_IMG_SIZE, VALIDATOR_IMG_SIZE)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(
        validator_model.predict(image_array, verbose=0)[0][0]
    )

    is_retinal = probability >= VALIDATOR_THRESHOLD

    return is_retinal, probability


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File is too large. Maximum allowed size is 10 MB."
    }), 413


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(pil_image):
    """
    Apply CLAHE and resize the image.
    ResNet50 preprocessing is included in the model.
    """

    img = np.array(pil_image.convert("RGB"))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    img = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB,
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    return arr


# --------------------------------------------------
# IMAGE TO BASE64
# --------------------------------------------------

def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")

    return base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")
# --------------------------------------------------
# LOGIN AND LOGOUT
# --------------------------------------------------

from flask import redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, Optional, Regexp

    
class LoginForm(FlaskForm):
    email = StringField(
        "Email or Patient ID",
        validators=[DataRequired(), Length(max=150)]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()]
    )
    submit = SubmitField("Login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    portal_role = request.args.get("role", "").strip().lower()

    if not portal_role:
        return redirect(url_for("index"))

    if portal_role not in ("doctor", "patient"):
        flash("Invalid login portal.", "error")
        return redirect(url_for("index"))

    form = LoginForm()

    portal_titles = {
        "admin": "Admin Login",
        "doctor": "Doctor Login",
        "patient": "Patient Login",
    }

    login_labels = {
        "admin": "Admin Email",
        "doctor": "Doctor Email",
        "patient": "Patient ID",
    }

    login_placeholders = {
        "admin": "Enter admin email",
        "doctor": "Enter doctor email",
        "patient": "Enter your Patient ID (e.g. PAT-XXXXXXXX)",
    }

    if form.validate_on_submit():
        identifier = form.email.data.strip()
        user = None

        if portal_role == "patient":
            patient = PatientProfile.query.filter_by(
                patient_code=identifier.upper()
            ).first()

            if patient:
                user = patient.user
        else:
            user = User.query.filter_by(
                email=identifier.lower()
            ).first()

        if (
            user
            and user.active
            and (user.role or "").strip().lower() == portal_role
            and user.check_password(form.password.data)
        ):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash(
            f"Invalid {portal_role} credentials, or account is inactive.",
            "error"
        )

    return render_template(
        "login.html",
        form=form,
        portal_role=portal_role,
        portal_title=portal_titles[portal_role],
        login_label=login_labels[portal_role],
        login_placeholder=login_placeholders[portal_role],
    )



@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("index"))





@app.route("/dashboard")
@login_required
def dashboard():
    role = (current_user.role or "").strip().lower()

    if role == "doctor":
        return redirect(url_for("doctor_dashboard"))

    if role == "patient":
        return redirect(url_for("patient_dashboard"))

    logout_user()
    return redirect(url_for("login"))
# --------------------------------------------------
# PATIENT REGISTRATION
# --------------------------------------------------


@app.route("/register", methods=["GET", "POST"])
def register():
    flash(
        "Patient accounts can only be created by an authorized doctor.",
        "error"
    )
    return redirect(url_for("index"))

# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------

from functools import wraps


def role_required(role):
    def decorator(view_function):
        @wraps(view_function)
        @login_required
        def wrapped_view(*args, **kwargs):
            if current_user.role != role:
                flash("You are not authorized to access this page.", "error")
                return redirect(url_for("dashboard"))

            return view_function(*args, **kwargs)

        return wrapped_view

    return decorator


@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    doctors = (
        User.query
        .filter_by(role="doctor")
        .order_by(User.name)
        .all()
    )

    return render_template(
        "admin.html",
        doctors=doctors
    )

# --------------------------------------------------
# ADMIN: CREATE DOCTOR ACCOUNT
# --------------------------------------------------

class DoctorForm(FlaskForm):
    name = StringField(
        "Doctor Name",
        validators=[
            DataRequired(),
            Length(min=2, max=120),
            Regexp(
                r"^[A-Za-z][A-Za-z .'-]*$",
                message="Name must contain letters only (spaces, apostrophes, periods and hyphens are allowed)."
            )
        ]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email(), Length(max=150)]
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, max=128),
            Regexp(
                r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,128}$",
                message="Password must be at least 8 characters and include an uppercase letter, a lowercase letter, and a number."
            )
        ]
    )
    phone = StringField("Phone", validators=[Length(max=20)])
    specialization = StringField("Specialization")
    qualification = StringField("Qualification")
    submit = SubmitField("Create Doctor")


@app.route("/admin/doctors/create", methods=["GET", "POST"])
@role_required("admin")
def create_doctor():
    form = DoctorForm()

    if form.validate_on_submit():
        email = form.email.data.strip().lower()

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return render_template("doctor_form.html", form=form)

        doctor = User(
            name=form.name.data.strip(),
            email=email,
            role="doctor",
            active=True
        )
        doctor.set_password(form.password.data)

        try:
            db.session.add(doctor)
            db.session.flush()

            profile = DoctorProfile(
                user_id=doctor.id,
                phone=form.phone.data.strip(),
                specialization=form.specialization.data.strip(),
                qualification=form.qualification.data.strip()
            )

            db.session.add(profile)
            db.session.commit()

            flash("Doctor account created successfully.", "success")
            return redirect(url_for("admin_dashboard"))

        except Exception:
            db.session.rollback()
            app.logger.exception("Doctor account creation failed")
            flash("Could not create the doctor account.", "error")

    return render_template("doctor_form.html", form=form)


# --------------------------------------------------
# PUBLIC DOCTOR SELF-REGISTRATION
# --------------------------------------------------

@app.route("/doctor/register", methods=["GET", "POST"])
def doctor_registration():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    form = DoctorForm()

    if form.validate_on_submit():
        email = form.email.data.strip().lower()

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists. Please log in.", "error")
            return render_template("doctor_registration.html", form=form)

        doctor = User(
            name=form.name.data.strip(),
            email=email,
            role="doctor",
            active=True
        )
        doctor.set_password(form.password.data)

        try:
            db.session.add(doctor)
            db.session.flush()

            profile = DoctorProfile(
                user_id=doctor.id,
                phone=(form.phone.data or "").strip(),
                specialization=(form.specialization.data or "").strip(),
                qualification=(form.qualification.data or "").strip()
            )

            db.session.add(profile)
            db.session.commit()

            login_user(doctor)
            flash("Doctor account created successfully. Welcome to EyeScan!", "success")
            return redirect(url_for("doctor_dashboard"))

        except Exception:
            db.session.rollback()
            app.logger.exception("Doctor self-registration failed")
            flash("Registration could not be completed. Please try again.", "error")

    return render_template("doctor_registration.html", form=form)


class PatientForm(FlaskForm):
    name = StringField(
        "Patient Name",
        validators=[DataRequired(), Length(min=2, max=120)]
    )
    email = StringField(
        "Patient Email",
        validators=[DataRequired(), Email(), Length(max=150)]
    )
    password = PasswordField(
        "Initial Password",
        validators=[DataRequired(), Length(min=8, max=128)]
    )
    phone = StringField("Phone Number", validators=[Length(max=20)])
    age = StringField("Age", validators=[Length(max=3)])
    gender = SelectField(
        "Gender",
        choices=[("", "Select gender"), ("Male", "Male"), ("Female", "Female"), ("Other", "Other"), ("Prefer not to say", "Prefer not to say")],
        validators=[Optional()]
    )
    submit = SubmitField("Register Patient")


@app.route("/doctor/patients/create", methods=["GET", "POST"])
@role_required("doctor")
def create_patient():
    form = PatientForm()

    if form.validate_on_submit():
        email = form.email.data.strip().lower()

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return render_template("patient_form.html", form=form)

        age_value = (form.age.data or "").strip()
        gender = (form.gender.data or "").strip()
        try:
            age = int(age_value) if age_value else None
            if age is not None and not 1 <= age <= 120:
                raise ValueError
        except ValueError:
            flash("Enter a valid age between 1 and 120.", "error")
            return render_template("patient_form.html", form=form)

        if gender and gender not in {"Male", "Female", "Other", "Prefer not to say"}:
            flash("Select a valid gender option.", "error")
            return render_template("patient_form.html", form=form)

        patient_user = User(
            name=form.name.data.strip(),
            email=email,
            role="patient",
            active=True
        )
        patient_user.set_password(form.password.data)

        try:
            db.session.add(patient_user)
            db.session.flush()

            patient_profile = PatientProfile(
                user_id=patient_user.id,
                patient_code=generate_patient_code(),
                phone=(form.phone.data or "").strip(),
                age=age,
                gender=gender or None
            )

            db.session.add(patient_profile)
            db.session.commit()

            flash(
                f"Patient registered successfully. Patient ID: "
                f"{patient_profile.patient_code}",
                "success"
            )
            return redirect(url_for("doctor_dashboard"))

        except Exception:
            db.session.rollback()
            app.logger.exception("Patient registration failed")
            flash("Could not register the patient.", "error")

    return render_template("patient_form.html", form=form)


# --------------------------------------------------
# ADMIN: ACTIVATE / DEACTIVATE DOCTOR
# --------------------------------------------------

@app.route("/admin/doctors/<int:doctor_id>/status", methods=["POST"])
@role_required("admin")
def toggle_doctor_status(doctor_id):
    doctor = User.query.filter_by(
        id=doctor_id,
        role="doctor"
    ).first_or_404()

    doctor.active = not doctor.active

    db.session.commit()

    status = "activated" if doctor.active else "deactivated"

    flash(
        f"Doctor account {status} successfully.",
        "success"
    )

    return redirect(url_for("admin_dashboard"))

# --------------------------------------------------
# DOCTOR: VIEW / EDIT / DELETE PATIENT ACCOUNTS
# --------------------------------------------------

@app.route("/doctor/patients/<int:patient_id>/view")
@role_required("doctor")
def view_patient(patient_id):
    patient = db.session.get(PatientProfile, patient_id)
    if patient is None:
        return "Patient not found.", 404
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.created_at.desc()).all()
    return render_template("patient_view.html", patient=patient, reports=reports)


@app.route("/doctor/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@role_required("doctor")
def edit_patient(patient_id):
    patient = db.session.get(PatientProfile, patient_id)
    if patient is None:
        return "Patient not found.", 404

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        age_value = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        password = request.form.get("password", "")

        if not name or not email:
            flash("Patient name and email are required.", "error")
            return render_template("patient_edit.html", patient=patient)

        existing_email = User.query.filter(User.email == email, User.id != patient.user_id).first()
        if existing_email:
            flash("That email address is already in use.", "error")
            return render_template("patient_edit.html", patient=patient)

        try:
            age = int(age_value) if age_value else None
            if age is not None and not 1 <= age <= 120:
                raise ValueError
        except ValueError:
            flash("Enter a valid age between 1 and 120.", "error")
            return render_template("patient_edit.html", patient=patient)

        patient.user.name = name
        patient.user.email = email
        patient.phone = phone or None
        patient.age = age
        patient.gender = gender or None
        if password.strip():
            if len(password) < 8:
                flash("Password must contain at least 8 characters.", "error")
                return render_template("patient_edit.html", patient=patient)
            patient.user.set_password(password)

        db.session.commit()
        flash("Patient account updated successfully.", "success")
        return redirect(url_for("doctor_dashboard"))

    return render_template("patient_edit.html", patient=patient)


@app.route("/doctor/patients/<int:patient_id>/delete", methods=["POST"])
@role_required("doctor")
def delete_patient(patient_id):
    patient = db.session.get(PatientProfile, patient_id)
    if patient is None:
        flash("Patient not found.", "error")
        return redirect(url_for("doctor_dashboard"))

    # Remove reports first because they reference the patient profile.
    Report.query.filter_by(patient_id=patient.id).delete(synchronize_session=False)
    user = patient.user
    db.session.delete(patient)
    db.session.flush()
    if user is not None:
        db.session.delete(user)
    db.session.commit()
    flash("Patient account and its reports were deleted.", "success")
    return redirect(url_for("doctor_dashboard"))


# --------------------------------------------------
# REPORT VIEW / EDIT NOTES / DELETE
# --------------------------------------------------

def _authorized_report_or_403(report_id):
    report = db.session.get(Report, report_id)
    if report is None:
        return None, ("Report not found.", 404)
    if current_user.role == "doctor" and report.doctor_id != current_user.id:
        return None, ("Access denied.", 403)
    if current_user.role == "patient":
        patient = PatientProfile.query.filter_by(user_id=current_user.id).first()
        if patient is None or report.patient_id != patient.id:
            return None, ("Access denied.", 403)
    elif current_user.role not in ("doctor", "admin"):
        return None, ("Access denied.", 403)
    return report, None


@app.route("/reports/<int:report_id>")
@login_required
def view_report(report_id):
    report, error = _authorized_report_or_403(report_id)
    if error:
        return error
    original_image_b64 = (
        base64.b64encode(report.original_image).decode("ascii")
        if report.original_image else None
    )
    return render_template(
        "report.html",
        report=report,
        patient_view=(current_user.role == "patient"),
        original_image_b64=original_image_b64,
    )


@app.route("/doctor/reports/<int:report_id>/edit", methods=["GET", "POST"])
@role_required("doctor")
def edit_report_notes(report_id):
    report = db.session.get(Report, report_id)
    if report is None:
        return "Report not found.", 404
    if report.doctor_id != current_user.id:
        return "Access denied.", 403
    if request.method == "POST":
        report.doctor_notes = request.form.get("doctor_notes", "").strip() or None
        db.session.commit()
        flash("Doctor review notes saved. The AI prediction and confidence were not changed.", "success")
        return redirect(url_for("doctor_dashboard"))
    return render_template("doctor_report_edit.html", report=report)


@app.route("/doctor/reports/<int:report_id>/delete", methods=["POST"])
@role_required("doctor")
def delete_doctor_report(report_id):
    report = db.session.get(Report, report_id)
    if report is None:
        flash("Report not found.", "error")
    elif report.doctor_id != current_user.id:
        return "Access denied.", 403
    else:
        db.session.delete(report)
        db.session.commit()
        flash("Report deleted successfully.", "success")
    return redirect(url_for("doctor_dashboard"))


# --------------------------------------------------
# DOCTOR DASHBOARD
# --------------------------------------------------

@app.route("/doctor")
@role_required("doctor")
def doctor_dashboard():
    reports = (
        Report.query
        .filter_by(doctor_id=current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )
    patients = (
        PatientProfile.query
        .join(User, PatientProfile.user_id == User.id)
        .order_by(User.name)
        .all()
    )

    return render_template(
        "doctor.html",
        reports=reports,
        patients=patients,
    )

# --------------------------------------------------
# PATIENT DASHBOARD
# --------------------------------------------------


@app.route("/patient")
@role_required("patient")
def patient_dashboard():
    patient = PatientProfile.query.filter_by(
        user_id=current_user.id
    ).first()

    if patient is None:
        flash("Patient profile not found.", "danger")
        logout_user()
        return redirect(url_for("login"))

    reports = Report.query.filter_by(
        patient_id=patient.id
    ).order_by(Report.created_at.desc()).all()

    return render_template(
        "patient.html",
        patient=patient,
        reports=reports
    )


# --------------------------------------------------
# DOCTOR: UPLOAD RETINAL IMAGE AND SAVE REPORT
# --------------------------------------------------

@app.route("/doctor/upload", methods=["GET", "POST"])
@role_required("doctor")
def doctor_upload():
    if request.method == "GET":
        return render_template("upload.html")

    patient_code = request.form.get("patient_code", "").strip().upper()
    file = request.files.get("image")

    patient = PatientProfile.query.filter_by(
        patient_code=patient_code
    ).first()

    if not patient:
        flash("Patient ID not found. Please check the ID.", "error")
        return redirect(url_for("doctor_upload"))

    if not file:
        flash("Please select a retinal image.", "error")
        return redirect(url_for("doctor_upload"))

    print("DEBUG: File received:", file.filename)
    pil_image, error = validate_upload(file)
    print("DEBUG: Upload validation error:", error)

    if error:
        flash(error, "error")
        return redirect(url_for("doctor_upload"))

    print("DEBUG: Image dimensions:", pil_image.size)

    if not looks_like_retina_image(pil_image):
        print("DEBUG: Basic image-size check failed")
        flash("Please upload a clear retinal fundus image.", "error")
        return redirect(url_for("doctor_upload"))

    print("DEBUG: Basic image-size check passed")
    is_retinal, retinal_probability = validate_retinal_classifier(pil_image)
    print("DEBUG: Retinal probability:", retinal_probability)
    print("DEBUG: Is retinal:", is_retinal)

    if not is_retinal:
        flash(
            "This image is not recognized as a retinal fundus image. "
            "Please upload a valid retinal image.",
            "error"
        )
        return redirect(url_for("doctor_upload"))

    try:
        img_array = preprocess_image(pil_image)

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx] * 100)

     
        info = DISEASE_INFO.get(
            predicted_class,
            {"label": predicted_class}
        )

       
        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Retina Screening Report", ln=True, align="C")

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Patient ID: {patient.patient_code}", ln=True)
        pdf.cell(0, 8, f"Patient Name: {patient.user.name}", ln=True)
        pdf.cell(0, 8, f"Age: {patient.age if patient.age is not None else 'Not provided'}", ln=True)
        pdf.cell(0, 8, f"Gender: {patient.gender or 'Not provided'}", ln=True)
        pdf.cell(0, 8, f"Doctor: {current_user.name}", ln=True)
        pdf.cell(
            0, 8,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ln=True
        )
        pdf.ln(4)
        pdf.cell(0, 8, f"Model result: {info['label']}", ln=True)
        pdf.cell(0, 8, f"Model confidence: {confidence:.1f}%", ln=True)
        pdf.ln(5)

        
        # Save images temporarily so PyFPDF can insert them
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = os.path.join(temp_dir, "original.png")
            

            pil_image.convert("RGB").save(original_path, format="PNG")
           

            # Display images side by side at a smaller size
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(90, 8, "Original Retinal Image", align="C")
           

            image_y = pdf.get_y()

            pdf.image(original_path, x=25, y=image_y, w=75)
            

            # Reserve space below both images
            pdf.set_y(image_y + 115)
            pdf.ln(3)

            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(
                0, 6,
                "This AI-generated result is experimental screening "
                "information, not a medical diagnosis. The model may "
                "be incorrect. Consult a qualified healthcare "
                "professional for assessment."
            )

            pdf_bytes = pdf.output(dest="S").encode("latin-1")



        image_buffer = io.BytesIO()
        pil_image.convert("RGB").save(image_buffer, format="PNG")

        report = Report(
            patient_id=patient.id,
            doctor_id=current_user.id,
            predicted_class=predicted_class,
            confidence=confidence,
            report_pdf=pdf_bytes,
            original_image=image_buffer.getvalue()
        )

        db.session.add(report)
        db.session.commit()

        flash("Report saved to the patient's account.", "success")
        return redirect(url_for("doctor_dashboard"))

    except Exception:
        db.session.rollback()
        app.logger.exception("Doctor upload or report generation failed")
        flash(
            "Processing failed. Check the terminal for the error details.",
            "error"
        )
        return redirect(url_for("doctor_upload"))

    # --------------------------------------------------
# SECURE REPORT DOWNLOAD
# --------------------------------------------------

@app.route("/report/<int:report_id>/download")
@login_required
def download_patient_report(report_id):
    report = db.session.get(Report, report_id)

    if report is None:
        return "Report not found.", 404

    # Patients can download only their own reports.
    if current_user.role == "patient":
        patient = PatientProfile.query.filter_by(
            user_id=current_user.id
        ).first()

        if not patient or report.patient_id != patient.id:
            return "Access denied.", 403

    # Doctors can download only reports they created.
    elif current_user.role == "doctor":
        if report.doctor_id != current_user.id:
            return "Access denied.", 403

    # Only admins, the report's patient, or its doctor can download.
    elif current_user.role != "admin":
        return "Access denied.", 403

    return send_file(
        io.BytesIO(report.report_pdf),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"retina_report_{report.id}.pdf"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
@role_required("doctor")
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": (
                "Please upload a retinal fundus image "
                "in JPG or PNG format."
            )
        }), 400

    file = request.files["image"]

    # Step 1: Validate extension and actual image content.
    pil_image, error = validate_upload(file)

    if error:
        return jsonify({"error": error}), 400

    # Step 2: Basic retina-like screening.
    if not looks_like_retina_image(pil_image):
        return jsonify({
            "error": (
                "This image does not appear to be a retinal "
                "fundus image. Please upload a clear retina "
                "image in JPG or PNG format."
            )
        }), 400
        # Step 3: Validate whether the image is retinal.
    is_retinal, retinal_probability = validate_retinal_classifier(pil_image)
    print("DEBUG - Retinal probability:", retinal_probability)
    print("DEBUG - Is retinal:", is_retinal)

    if not is_retinal:
        return jsonify({
            "error": (
                "This image is not recognized as a retinal "
                "fundus image. Please upload a valid retinal image."
            )
        }), 400

    # Step 4: Predict only after the upload checks pass.
    try:
        img_array = preprocess_image(pil_image)

        predictions = model.predict(
            img_array,
            verbose=0,
        )[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx] * 100)

       
        is_uncertain = confidence < UNCERTAIN_THRESHOLD
        is_normal = (
            not is_uncertain
            and predicted_class == "normal"
        )

        info = DISEASE_INFO.get(
            predicted_class,
            {
                "label": predicted_class,
                "icon": "Warning",
                "color": "#e74c3c",
            },
        )

        result = {
            "is_uncertain": is_uncertain,
            "is_normal": is_normal,
            "predicted_class": predicted_class,
            "label": info["label"],
            "icon": info["icon"],
            "color": info["color"],
            "confidence": round(confidence, 1),
            "original_image": pil_to_base64(
                pil_image.resize((IMG_SIZE, IMG_SIZE))
            ),
        }

        global LAST_RESULT
        LAST_RESULT = result

        return jsonify(result)

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": (
                "Could not process this image. "
                "Please try another clear retinal image."
            )
        }), 500


# --------------------------------------------------
# DOWNLOAD PDF REPORT
# --------------------------------------------------
@app.route("/download_report")
@login_required
@role_required("doctor")
def download_report():

    if not LAST_RESULT:
        return jsonify({
            "error": "Please make a prediction before downloading a report."
        }), 400

    r = LAST_RESULT

    orig_bytes = base64.b64decode(r["original_image"])

    # Use temporary files and remove them even if report generation fails.
    import tempfile

    orig_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as f:
            f.write(orig_bytes)
            orig_path = f.name

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(
            0, 12,
            "EyeScan - Prediction Report",
            ln=True,
            align="C",
        )

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 120, 120)

        pdf.cell(
            0, 8,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ln=True,
            align="C",
        )

        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        y_start = pdf.get_y()

        pdf.image(orig_path, x=55, y=y_start, w=90)

        pdf.set_y(y_start + 95)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, "Original Retinal Image", align="C", ln=True)
        pdf.ln(8)

        pdf.set_font("Helvetica", "B", 15)

        if r["is_uncertain"]:
            pdf.set_text_color(230, 180, 20)
            pdf.cell(
                0, 10,
                "Result: Unable to Determine",
                ln=True,
            )

        elif r["is_normal"]:
            pdf.set_text_color(46, 204, 113)
            pdf.cell(
                0, 10,
                "Result: Looks Normal",
                ln=True,
            )

        else:
            pdf.set_text_color(200, 60, 50)
            pdf.cell(
                0, 10,
                f"Result: {r['label']}",
                ln=True,
            )

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)

        if not r["is_uncertain"]:
            pdf.cell(
                0, 8,
                f"Confidence: {r['confidence']}%",
                ln=True,
            )

        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)

        pdf.multi_cell(
            0,
            6,
            "This AI screening tool is not a medical diagnosis. "
            "Please consult a qualified healthcare provider "
            "for confirmation.",
        )

        pdf_bytes = bytes(pdf.output())

        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)

        filename = (
            "eyescan_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )

    finally:
        for path in (orig_path,):
            if path and os.path.exists(path):
                os.remove(path)


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------


def initialize_database():
    with app.app_context():
        db.create_all()

        # Lightweight SQLite migration for existing portal databases.
        inspector = __import__("sqlalchemy").inspect(db.engine)
        report_columns = {column["name"] for column in inspector.get_columns("report")}
        if "doctor_notes" not in report_columns:
            with db.engine.begin() as connection:
                connection.exec_driver_sql("ALTER TABLE report ADD COLUMN doctor_notes TEXT")
        if "original_image" not in report_columns:
            with db.engine.begin() as connection:
                connection.exec_driver_sql("ALTER TABLE report ADD COLUMN original_image BLOB")

        admin_email = os.environ.get(
            "RETINA_ADMIN_EMAIL",
            "admin@eyescan.local"
        ).strip().lower()

        admin_password = os.environ.get("RETINA_ADMIN_PASSWORD")

        existing_admin = User.query.filter_by(
            email=admin_email
        ).first()

        if existing_admin is None:
            if not admin_password:
                raise RuntimeError(
                    "Set the RETINA_ADMIN_PASSWORD environment variable "
                    "before starting the application for the first time."
                )

            admin = User(
                name="EyeScan Administrator",
                email=admin_email,
                role="admin",
                active=True
            )
            admin.set_password(admin_password)

            db.session.add(admin)
            db.session.commit()

            print("Initial admin account created.")
        else:
            print("Admin account already exists.")


if __name__ == "__main__":
    initialize_database()
    app.run(debug=True)
