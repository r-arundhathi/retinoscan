# EyeScan Retina Portal

## Setup
Keep `app_flask.py` and `templates/` in `C:\retina_project`. Your existing model files should be:
- `models_grouped\best_model_finetuned.keras`
- `outputs_grouped\class_names.json`

In the VS Code terminal, activate your existing virtual environment and run:
```powershell
python -m pip install -r requirements-portal.txt
$env:RETINA_ADMIN_EMAIL="your-admin-email@example.com"
$env:RETINA_ADMIN_PASSWORD="set-a-long-unique-password"
$env:RETINA_SECRET_KEY="set-a-long-random-secret"
python app_flask.py
```
Then open `http://127.0.0.1:5000`.

If you do not set admin variables before the first database initialization, the demo account is `admin@retinaproject.local` / `ChangeMeNow123!`. Change this before handling any real or sensitive data. Environment variables do not reset an existing admin password after the database is created.

## Features
- Admin manages doctor profile details, initial password, and active/inactive status.
- Patients self-register and receive unique Patient IDs.
- Doctors upload JPG/JPEG/PNG retina images, generate predictions, and save them against a Patient ID.
- Patients see only their own reports; doctors see only reports they created.
- Password hashes, Flask sessions, CSRF protection, role checks, server-side file validation.

## Notes
- Reports and images are stored in SQLite as Base64. Protect the database file as sensitive patient data.
- The retina/non-retina screening is a simple heuristic, not a reliable classifier; it may reject valid retina photos or allow other photos. A dedicated retina-vs-non-retina classifier is needed for stronger validation.
- Do not expose Flask's development server to the internet. Production needs HTTPS, a production WSGI server, secure secrets, backups, audit logging, rate limiting, and privacy/legal review.
- Predictions are experimental AI screening outputs, not diagnoses.
