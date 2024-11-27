from django.shortcuts import render, redirect
from django.http import HttpResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import h5py
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.hashers import make_password, check_password

# Define the model paths
model_paths = {
    'bi_rnn_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\bi_rnn_signature_verification_model.h5",
    'crnn_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\crnn_signature_verification_model.keras",
    'rnn_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\rnn_signature_verfication_model.h5",
    'signature_model': r"C:\Users\Vyshujaanu\Desktop\Registration\authentication\signature_model.h5"
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

# SignIn view
def SignIn(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if authenticate(username, password):
            return redirect('verify')  # Redirect to verify page
        else:
            return HttpResponse('Invalid credentials', status=401)
    return render(request, 'SignIn.html')

def authenticate(username, password):
    if username in users and check_password(password, users[username]['password']):
        return True
    return False

# SignUp view
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            # Hash the password and save user data to in-memory database
            users[username] = {'password': make_password(password)}
            return redirect('signin')  # Redirect to SignIn page after successful signup
        else:
            return HttpResponse('Passwords do not match', status=400)

    return render(request, 'SignUp.html')

# Verify view
def verify(request):
    result_context = None
    uploaded_image_url = None
    previous_results = request.session.get('previous_results', [])

    if request.method == 'POST' and 'signature_image' in request.FILES:
        img_file = request.FILES['signature_image']
        file_name = default_storage.save(img_file.name, ContentFile(img_file.read()))
        uploaded_image_url = default_storage.url(file_name)
        img_name = img_file.name
        img = Image.open(img_file)
        img = preprocess_image(img)  # Preprocess the image

        # Perform verification using the loaded models
        predictions = {}
        for model_name in ['bi_rnn_model', 'crnn_model', 'rnn_model', 'signature_model']:
            if models.get(model_name):
                try:
                    predictions[model_name] = models[model_name].predict(img)
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
            else:
                print(f"{model_name} not loaded.")

        if predictions:
            # Normalize the confidence values to be between 0 and 1.5 and format with more than 5 decimal points
            normalized_confidences = {model_name: np.clip(np.max(result) * 1.5, 0, 1.5) for model_name, result in predictions.items()}

            # Determine final status based on confidence threshold (0.80)
            final_status = 'Forged' if any(confidence > 0.80 for confidence in normalized_confidences.values()) else 'Real'

            # Apply final status to all models
            verification_status = {}
            for model_name in normalized_confidences:
                confidence = normalized_confidences[model_name]
                verification_status[model_name] = {'status': final_status, 'confidence': f"{confidence:.6f}"}

            # Store the current result
            current_result = {
                'image_name': img_name,
                'results': {
                    f"Model {idx + 1}": details
                    for idx, (model_name, details) in enumerate(verification_status.items())
                }
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
