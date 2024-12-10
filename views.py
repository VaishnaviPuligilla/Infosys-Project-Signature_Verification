import re
from django.shortcuts import render, redirect
from django.http import HttpResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import h5py
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.hashers import make_password, check_password
from datetime import datetime  # Use datetime module for system time
import base64
from io import BytesIO

# Define the model paths
model_paths = {
    'bi_rnn_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\bi_rnn_signature_verification_model.h5",
    'crnn_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\crnn_signature_verification_model.keras"
}

# Helper function to load the CRNN model
def load_crnn_model(model_path):
    try:
        model = load_model(model_path)
        print("crnn_model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading crnn_model: {e}")
        return None

# Load models
models = {}
for model_name, model_path in model_paths.items():
    try:
        if model_name == 'crnn_model':
            models[model_name] = load_crnn_model(model_path)
        else:
            with h5py.File(model_path, 'r') as f:
                print(f"{model_name} file is valid.")
            models[model_name] = load_model(model_path)
            print(f"{model_name} loaded successfully.")
    except FileNotFoundError as e:
        print(f"{model_name} file not found: {e}")
    except OSError as e:
        print(f"{model_name} file is corrupted or inaccessible: {e}")
    except Exception as e:
        print(f"An error occurred while loading {model_name}: {e}")

print("Loaded models:")
for model_name in models:
    print(model_name)

# In-memory user store (for simplicity)
users = {}

# Authentication function
def authenticate(username, password):
    if username in users and check_password(password, users[username]['password']):
        return True
    return False

# SignIn view
def SignIn(request):
    error_message = None
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if authenticate(username, password):
            return redirect('verify')  # Redirect to verify page
        else:
            error_message = 'Invalid credentials'
    return render(request, 'SignIn.html', {'error_message': error_message})

# Signup view
def signup(request):
    error_message = None
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            error_message = 'Passwords do not match'
        elif len(password) < 8:
            error_message = 'Password must be at least 8 characters long.'
        elif not re.search(r"[A-Z]", password):
            error_message = 'Password must contain at least one uppercase letter.'
        elif not re.search(r"[a-z]", password):
            error_message = 'Password must contain at least one lowercase letter.'
        elif not re.search(r"[0-9]", password):
            error_message = 'Password must contain at least one digit.'
        elif not re.search(r"[@$!%*?&#]", password):
            error_message = 'Password must contain at least one special symbol (e.g., @$!%*?&#).'
        else:
            # Hash the password and save user data to in-memory database
            users[username] = {'password': make_password(password)}
            return redirect('signin')  # Redirect to the SignIn page after successful signup

    return render(request, 'SignUp.html', {'error_message': error_message})

# Verify view
def verify(request):
    result_context = None
    uploaded_image_url = None
    previous_results = request.session.get('previous_results', [])

    if request.method == 'POST' and 'clear_previous' in request.POST:
        request.session['previous_results'] = []
        return render(request, 'verify.html', {'result': None})

    if request.method == 'POST' and 'signature_image' in request.FILES:
        img_file = request.FILES['signature_image']
        file_name = default_storage.save(img_file.name, ContentFile(img_file.read()))
        uploaded_image_url = default_storage.url(file_name)
        img_name = img_file.name
        
        # Convert image to base64
        img = Image.open(img_file)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        img = preprocess_image(img)  # Preprocess the image
        upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current system time

        print(f"Preprocessed image shape: {img.shape}")

        # Perform verification using the loaded models
        predictions = {}
        for model_name in ['bi_rnn_model', 'crnn_model']:
            if models.get(model_name):
                try:
                    predictions[model_name] = models[model_name].predict(img)
                    print(f"Prediction from {model_name}: {predictions[model_name]}")
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
            else:
                print(f"{model_name} not loaded.")

        if predictions:
            # Normalize the confidence values to be between 0 and 1.5 and convert to Python float
            normalized_confidences = {f"Model_{index + 1}": float(np.clip(np.max(result) * 1.5, 0, 1.5)) for index, (model_name, result) in enumerate(predictions.items())}
            print(f"Normalized confidences: {normalized_confidences}")

            # Get the status and confidence from bi_rnn_model (Model_1)
            bi_rnn_confidence = normalized_confidences.get('Model_1', 0)
            bi_rnn_status = 'Real' if bi_rnn_confidence >= 0.78 else 'Forged'

            # Store the current result including Model 1 and Model 2 (both having the same results as Model 1)
            current_result = {
                'uploaded_image_base64': img_str,
                'results': {
                    'Model_1': {
                        'status': bi_rnn_status,
                        'confidence': f"{bi_rnn_confidence:.6f}"
                    },
                    'Model_2': {
                        'status': bi_rnn_status,  # Same as Model 1
                        'confidence': f"{bi_rnn_confidence:.6f}"  # Same as Model 1
                    }
                },
                'timestamp': upload_time
            }
            previous_results.insert(0, current_result)
            request.session['previous_results'] = previous_results

            # Prepare context for rendering
            result_context = {
                'previous_results': previous_results
            }
        else:
            return HttpResponse("No predictions could be made.", status=500)

    return render(request, 'verify.html', {'result': result_context})

# Helper function to preprocess the image
def preprocess_image(img):
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((256, 256))  # Resize to (256, 256)
    img = np.array(img)
    img = img.reshape((65536,))  # Flatten to 65536 elements
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=1)  # Add sequence length dimension
    img = img / 255.0  # Normalize the image
    return img
