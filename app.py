from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

app = Flask(__name__)

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
model.to('cuda').eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    file = request.files['image']
    image = Image.open(file)

    # Transform and predict
    input_images = transform_image(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    # Save result to in-memory file
    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
