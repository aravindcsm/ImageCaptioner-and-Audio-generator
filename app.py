from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from gtts import gTTS
import os


app = Flask(__name__)


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


def text_to_audio(text, audio_path="caption_audio.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)
    return audio_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(url_for('index'))
    
    
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    
    caption = generate_caption(image_path)
    audio_path = text_to_audio(caption, audio_path="static/caption_audio.mp3")

    return render_template('result.html', caption=caption, audio_path=audio_path)


if __name__ == '__main__':
    
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
