from flask import Flask, jsonify, request
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import os
import io
import base64

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2isTYjtuumq6JcPs9USw"
)

HARDCODED_IMAGE_PATH = "IMAGE.jpg"
ANNOTATED_IMAGE_PATH = "result_with_boxes.jpg"

@app.route('/upload_image', methods=['POST'])
def trigger_inference():
    try:
        if not os.path.exists(HARDCODED_IMAGE_PATH):
            return jsonify({'status': 'error', 'message': 'Hardcoded image not found'}), 404

        # Run inference
        result = CLIENT.infer(HARDCODED_IMAGE_PATH, model_id="rice_vision/19")
        predictions = result.get("predictions", [])

        total_grains = len(predictions)
        good_grains = sum(1 for p in predictions if "Good" in p.get("class", ""))
        bad_grains = sum(1 for p in predictions if "Bad" in p.get("class", ""))

        percentage = (good_grains / bad_grains * 100) if bad_grains > 0 else 100.0 if good_grains > 0 else 0.0

        summary = {
            "Total Grains": total_grains,
            "Good Grain": good_grains,
            "Bad Grains": bad_grains,
            "Percentage": round(percentage, 2)
        }

        # Open image and draw boxes
        image = Image.open(HARDCODED_IMAGE_PATH)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        for prediction in predictions:
            x = prediction["x"]
            y = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            class_name = prediction["class"]
            confidence = prediction["confidence"]

            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left, top - 10), f"{class_name} ({confidence:.2f})", fill="red", font=font)

        # Save the annotated image locally
        image.save(ANNOTATED_IMAGE_PATH)

        # Convert annotated image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Add base64 image to summary
        summary['annotated_image'] = img_str

        return jsonify(summary), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
