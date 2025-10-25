import gradio as gr  # Interface library for building web apps
from fastai.vision.all import *  # FastAI tools for vision tasks
import time  # Used to add delay before clearing output

# Load the pre-trained classification model
learn = load_learner('pneumonia_model.pkl')

# Prediction function – returns a styled message based on the result
def predict(image, lang):
    if image is None:
        if lang == "العربية 🇪🇬":
            return "<div style='color:orange;'>⚠ من فضلك ارفع صورة أولاً!</div>"
        else:
            return "<div style='color:orange;'>⚠ Please upload an image first!</div>"

    pred, _, _ = learn.predict(image)  # Make prediction using the model

    # Return formatted result depending on language and prediction
    if lang == "العربية 🇪🇬":
        if pred == 'PNEUMONIA':
            return "<div style='color:red; font-size:20px;'>🚨 تم الكشف: يوجد التهاب رئوي</div>"
        else:
            return "<div style='color:green; font-size:20px;'>✅ لا يوجد التهاب رئوي</div>"
    else:
        if pred == 'PNEUMONIA':
            return "<div style='color:red; font-size:20px;'>🚨 Detected: Pneumonia</div>"
        else:
            return "<div style='color:green; font-size:20px;'>✅ Clear: No Pneumonia</div>"

# Clear function – resets inputs/outputs with a small delay
def clear():
    time.sleep(0.5)
    return None, ""

# Building the UI using Gradio Blocks layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue")) as demo:

    # Language selector dropdown
    lang_choice = gr.Dropdown(
        choices=["العربية 🇪🇬", "English 🇬🇧"],
        label="🌐 اختر اللغة / Select Language",
        value="العربية 🇪🇬"
    )

    # HTML titles (dynamic based on language)
    title = gr.HTML()
    subtitle = gr.HTML()

    # Default UI text (Arabic)
    title.value, subtitle.value = (
        "<div style='text-align:center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>🔬 كاشف الالتهاب الرئوي من صور الأشعة</div>",
        "<div style='text-align:center; font-size: 20px; font-weight:bold; color: #333; margin-bottom: 30px;'>استخدم النظام لتحليل أشعة الصدر واكتشاف الالتهاب خلال ثوانٍ</div>"
    )

    # Function to update title/subtitle based on selected language
    def update_ui(lang):
        if lang == "العربية 🇪🇬":
            return (
                "<div style='text-align:center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>🔬 كاشف الالتهاب الرئوي من صور الأشعة</div>",
                "<div style='text-align:center; font-size: 20px; font-weight:bold; color: #333; margin-bottom: 30px;'>استخدم النظام لتحليل أشعة الصدر واكتشاف الالتهاب خلال ثوانٍ</div>"
            )
        else:
            return (
                "<div style='text-align:center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>🔬 Pneumonia Detection from X-Ray Images</div>",
                "<div style='text-align:center; font-size: 20px; font-weight:bold; color: #333; margin-bottom: 30px;'>Use this tool to analyze chest X-Rays and detect pneumonia in seconds.</div>"
            )

    # Trigger UI update when language changes
    lang_choice.change(update_ui, inputs=lang_choice, outputs=[title, subtitle])

    # Image upload input inside a horizontal layout row
    with gr.Row():
        image = gr.Image(type="pil", sources=["upload"], label="📤 Upload / تحميل صورة", show_label=True, height=300)

    # Output area for displaying prediction result
    output = gr.HTML()

    # Action buttons
    analyze_btn = gr.Button("🔎 تحليل الصورة", variant="primary", interactive=True)
    reset_btn = gr.Button("🔁 إعادة تعيين", variant="secondary", interactive=True)

    # Update button labels when language changes
    def update_buttons(lang):
        if lang == "العربية 🇪🇬":
            return "🔎 تحليل الصورة", "🔁 إعادة تعيين"
        else:
            return "🔎 Analyze Image", "🔁 Reset"

    lang_choice.change(fn=update_buttons, inputs=lang_choice, outputs=[analyze_btn, reset_btn])

    # Bind actions to buttons
    analyze_btn.click(fn=predict, inputs=[image, lang_choice], outputs=output)
    reset_btn.click(fn=clear, inputs=[], outputs=[image, output])

    # Footer with developer credits and LinkedIn links
    gr.HTML(
        """
        <div style='margin-top: 50px; text-align: center; font-size: 16px; color: gray;'>
            <p><strong>Developed by</strong></p>
            <div style='display: flex; justify-content: center; gap: 100px; flex-wrap: wrap; margin-top: 20px;'>
                <div>
                    <p><strong>Eng. Marwa Waheed</strong></p>
                    <p>
                        <a href='https://www.linkedin.com/in/marwa-waheed-33a249326?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app' 
                           target='_blank' style='color: #0a66c2; text-decoration: none;'>
                           LinkedIn Marwa
                        </a>
                    </p>
                </div>
                <div>
                    <p><strong>Eng. Abdelaziz Diab</strong></p>
                    <p>
                        <a href='https://www.linkedin.com/in/abdelaziz-diab-577828344?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app' 
                           target='_blank' style='color: #0a66c2; text-decoration: none;'>
                           LinkedIn Abdelaziz
                        </a>
                    </p>
                </div>
            </div>
        </div>
        """
    )

# Start the app and generate a public shareable link
demo.launch(share=True)
