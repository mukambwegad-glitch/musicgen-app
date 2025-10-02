from flask import Flask, render_template, request
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch
import scipy.io.wavfile
import uuid
import os

app = Flask(__name__)

# Make sure 'static' folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load model & processor once (at startup)
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        # Generate audio from text prompt
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

        # Save to file in static folder
        sample_rate = model.config.audio_encoder.sampling_rate
        filename = f"static/{uuid.uuid4().hex}.wav"
        scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_values[0, 0].numpy())

        return render_template("index.html", audio_file=filename, prompt=prompt)

    return render_template("index.html", audio_file=None)

if __name__ == "__main__":
    # Use PORT env var from Render (default 10000 if local)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
