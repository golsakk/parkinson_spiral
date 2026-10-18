"""
سایت تشخیص پارکینسون از نقاشی مارپیچ با Gradio
"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ============ لود مدل ============
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model/best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============ Grad-CAM ============
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Grad-CAM نصب نیست. فقط پیش‌بینی ساده کار می‌کنه.")

# ============ تابع پیش‌بینی ============
def predict(image):
    if image is None:
        return None, "لطفاً یه عکس آپلود کن.", None

    img = Image.fromarray(image).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    healthy_prob = probs[0][0].item() * 100
    parkinson_prob = probs[0][1].item() * 100

    if predicted.item() == 1:
        result = f"🔴 احتمال پارکینسون: {parkinson_prob:.1f}%"
    else:
        result = f"🟢 سالم: {healthy_prob:.1f}%"

    detail = f"سالم: {healthy_prob:.1f}%  |  پارکینسون: {parkinson_prob:.1f}%"

    # Grad-CAM
    gradcam_img = None
    if GRADCAM_AVAILABLE:
        try:
            target_layers = [model.features[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
            gradcam_img = show_cam_on_image(
                img_array.astype(np.float32),
                grayscale_cam,
                use_rgb=True
            )
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    label_output = {"سالم": healthy_prob / 100, "پارکینسون": parkinson_prob / 100}
    return label_output, detail, gradcam_img


# ============ رابط گرافیکی ============
with gr.Blocks(title="تشخیص پارکینسون") as demo:
    gr.Markdown("""
    # 🧠 تشخیص پارکینسون از نقاشی مارپیچ
    
    یه عکس از نقاشی مارپیچ یا موج آپلود کن. مدل بررسی می‌کنه 
    که خطوط لرزش و بی‌نظمی دارن یا نه.
    
    ⚠️ توجه: این ابزار پژوهشی است، نه تشخیص قطعی پزشکی.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="عکس نقاشی", type="numpy")
            btn = gr.Button("تحلیل کن", variant="primary")

        with gr.Column():
            output_label = gr.Label(label="نتیجه", num_top_classes=2)
            output_detail = gr.Textbox(label="جزئیات")
            output_gradcam = gr.Image(label="نقشه حرارتی (Grad-CAM)")

    btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_detail, output_gradcam]
    )

    gr.Markdown("""
    ---
    دیتاست: Kaggle Parkinson's Drawings  
    مدل: MobileNetV2 + Transfer Learning  
    توضیح‌پذیری: Grad-CAM
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)