from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv


# Google API imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


load_dotenv()


app = Flask(__name__)
app.secret_key = 'supersecretkey'


# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Allowed file types
ALLOWED_EXTENSIONS = {'pdf', 'txt'}


# Google Drive config
SERVICE_ACCOUNT_FILE = 'credentials.json'  # 🔒 Replace with your service account file path
SCOPES = ['https://www.googleapis.com/auth/drive.file']
TARGET_FOLDER_ID = '1wgQdV5QT64lzhCpWIrcs-qNf2s1DnSPx'  # 📁 Replace with your Google Drive folder ID




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




def upload_to_drive(file_path, file_name):
    """Uploads a file to Google Drive using service account."""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=credentials)


    file_metadata = {
        'name': file_name,
        'parents': [TARGET_FOLDER_ID]
    }


    media = MediaFileUpload(file_path, resumable=True)


    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()


    print(f"📁 Uploaded to Drive with file ID: {uploaded_file.get('id')}")
    return uploaded_file.get('id')




@app.route('/')
def index():
    return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'bookfile' not in request.files:
        flash('⚠️ No file part')
        return redirect(request.url)


    file = request.files['bookfile']


    if file.filename == '':
        flash('⚠️ No selected file')
        return redirect(request.url)


    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)


        # ✅ Upload to Google Drive
        file_id = upload_to_drive(file_path, filename)


        flash(f"✅ File uploaded successfully to Drive! Summary will be generated soon.")
        return redirect(url_for('index'))


    flash('❌ File type not allowed.')
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(debug=True)






