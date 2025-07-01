import customtkinter as ctk  # pip install customtkinter
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import tensorflow as tf
import cv2
import os
import threading
from datetime import datetime

# ======================
# 1. Load Models & Data
# ======================

CLASSIFIER_PATH = "eye_disease_classifier.pkl"
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"Could not find {CLASSIFIER_PATH}.")

classifier_data = joblib.load(CLASSIFIER_PATH)
xgb_classifier = classifier_data['model']
label_map = classifier_data['label_map']
feature_extractor_path = classifier_data['tf_model_path']

if not os.path.exists(feature_extractor_path):
    raise FileNotFoundError(f"Could not find {feature_extractor_path}.")
feature_extractor = tf.keras.models.load_model(feature_extractor_path)

# For probability checks (if using an ensemble):
category_estimator = xgb_classifier.estimators_[0]

# ======================
# 2. Image Preprocessing
# ======================

IMAGE_SIZE = (768, 768)

def preprocess_image(image_path):
    """Apply CLAHE, center crop, and resize."""
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        h, w = img.shape[:2]
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        img = img[top:top + crop_size, left:left + crop_size]
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return img
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

# ======================
# 3. Fluorescein Cornea Check
# ======================

def looks_like_fluorescein_corneal(img, fraction_threshold=0.2):
    """
    Check if a significant portion of the image is in the green/blue range
    typical of a fluorescein-stained cornea under cobalt-blue light.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([60, 50, 50], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    fraction = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    return fraction > fraction_threshold

def is_corneal_circle(img):
    """
    Use Hough Circle detection to check for a circular region typical of a cornea.
    Adjust parameters as needed.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=300
        )
        return circles is not None
    except Exception as e:
        print(f"Error in corneal detection: {e}")
        return False

# ======================
# 4. Damage Calculation
# ======================

def calculate_damage(grade_idx, type_idx, category_idx):
    """Custom formula for damage estimate."""
    grade_percent = [0, 25, 50, 75, 30][grade_idx]
    type_percent = [0, 5, 10, 15, 20][type_idx]
    category_percent = [5, 10, 15][category_idx]
    return min(grade_percent + type_percent + category_percent, 100)

# ======================
# 5. Classification Function
# ======================

def classify_slitlamp_image(image_path, threshold=0.7):
    """
    1) Preprocess image.
    2) Check if it looks like a fluorescein cornea (color).
    3) Check if there's a corneal circle (Hough).
    4) (Optional) Confidence threshold from the XGB model.
    5) Classify, compute damage if valid cornea.
    """
    img = preprocess_image(image_path)
    if img is None:
        return None, None, None, None

    if not looks_like_fluorescein_corneal(img, fraction_threshold=0.2):
        return "Not a cornea", None, None, None

    if not is_corneal_circle(img):
        return "Not a cornea", None, None, None

    img_expanded = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img_expanded)
    features = features.reshape(1, -1)
    
    category_probs = category_estimator.predict_proba(features)
    best_cat_idx = np.argmax(category_probs, axis=1)[0]
    best_cat_conf = category_probs[0, best_cat_idx]
    if best_cat_conf < threshold:
        return "Not a cornea", None, None, None

    y_pred = xgb_classifier.predict(features)
    category_idx, type_idx, grade_idx = y_pred[0]
    damage = calculate_damage(grade_idx, type_idx, category_idx)

    category_label = label_map['category'].get(category_idx, f"Unknown {category_idx}")
    type_label = label_map['type'].get(type_idx, f"Unknown {type_idx}")
    grade_label = label_map['grade'].get(grade_idx, f"Unknown {grade_idx}")

    return category_label, type_label, grade_label, damage

# ======================
# 6. GUI Implementation using Grid Layout
# ======================

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Corneal Analysis Suite v2.0")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        self.configure(bg="black")
        self.history = []
        self.processing = False  # flag for progress simulation
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Create a main container frame using grid layout with two columns
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#222222")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1, uniform="col")
        self.main_frame.grid_columnconfigure(1, weight=1, uniform="col")
        self.main_frame.grid_rowconfigure(0, weight=1)

        # -----------------------
        # Left Panel: Image & Controls
        # -----------------------
        self.left_frame = ctk.CTkFrame(self.main_frame, corner_radius=20, fg_color="#333333")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Image preview (centered) – increased display size
        self.img_label = ctk.CTkLabel(self.left_frame, text="Image Preview", anchor="center")
        self.img_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Warning label above the controls if needed
        self.warning_label = ctk.CTkLabel(self.left_frame, text="", text_color="red", font=("Helvetica", 14))
        self.warning_label.grid(row=1, column=0, padx=10, pady=(0,5))
        
        # Upload button
        self.upload_btn = ctk.CTkButton(self.left_frame, text="Upload Slit-Lamp Image", command=self.start_processing, corner_radius=10, width=300)
        self.upload_btn.grid(row=2, column=0, padx=10, pady=(5,5))

        # Progress bar and percentage label underneath the button
        self.progress_bar = ctk.CTkProgressBar(self.left_frame, width=300)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=3, column=0, padx=10, pady=(5,2))
        self.progress_percentage_label = ctk.CTkLabel(self.left_frame, text="0%", font=("Helvetica", 12))
        self.progress_percentage_label.grid(row=4, column=0, padx=10, pady=(2,10))

        # -----------------------
        # Right Panel: Results & History (keeping original size but with rounded textboxes)
        # -----------------------
        self.right_frame = ctk.CTkFrame(self.main_frame, corner_radius=20, fg_color="#333333")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right_frame.grid_rowconfigure(0, weight=1, uniform="row")
        self.right_frame.grid_rowconfigure(1, weight=1, uniform="row")
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Results textbox with original dimensions and a modest corner radius
        self.result_text = ctk.CTkTextbox(self.right_frame, font=("Helvetica", 12),
                                          corner_radius=20, width=500, height=250)
        self.result_text.grid(row=0, column=0, padx=10, pady=10)
        
        # History textbox with original dimensions and a modest corner radius
        self.history_text = ctk.CTkTextbox(self.right_frame, font=("Helvetica", 12),
                                           corner_radius=20, width=500, height=250)
        self.history_text.grid(row=1, column=0, padx=10, pady=10)
        
        # Status bar at bottom of main window
        self.status_var = ctk.StringVar(value="Status: Ready")
        status_bar = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0,20))
        
        # Allow the main window to expand
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def start_processing(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        self.show_image_preview(file_path)
        self.warning_label.configure(text="")  # Clear any previous warnings
        self.update_status("Processing image...")
        self.toggle_ui_state(False)
        self.processing = True
        self.simulate_progress()  # Start the dynamic progress simulation
        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()

    def simulate_progress(self):
        # Start the progress at 0
        self.progress_value = 0
        def update_progress():
            if self.processing:
                # Increase slowly until near 100%
                if self.progress_value < 99:
                    self.progress_value += 1
                    self.progress_bar.set(self.progress_value / 100)
                    self.progress_percentage_label.configure(text=f"{self.progress_value}%")
                self.after(100, update_progress)
            else:
                # Once processing finishes, gradually fill remaining progress over 1 second (10 steps)
                remaining = 100 - self.progress_value
                step = remaining / 10.0
                def finish_progress(count=0):
                    if count < 10:
                        self.progress_value += step
                        self.progress_bar.set(self.progress_value / 100)
                        self.progress_percentage_label.configure(text=f"{int(self.progress_value)}%")
                        self.after(100, finish_progress, count+1)
                    else:
                        self.progress_value = 100
                        self.progress_bar.set(1)
                        self.progress_percentage_label.configure(text="100%")
                finish_progress()
        update_progress()

    def show_image_preview(self, file_path):
        try:
            pil_img = Image.open(file_path)
            # Increase maximum display size (e.g., 600x600)
            pil_img.thumbnail((600, 600))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.img_label.configure(image=tk_img, text="")
            self.img_label.image = tk_img
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")

    def process_image(self, file_path):
        try:
            results = classify_slitlamp_image(file_path)
            self.after(0, self.show_results, results, file_path)
        except Exception as e:
            self.after(0, self.handle_error, e)
        finally:
            self.processing = False  # Stop the progress simulation
            self.after(0, self.toggle_ui_state, True)

    def show_results(self, results, file_path):
        cat_label, type_label, grade_label, damage = results
        timestamp = datetime.now().strftime("%H:%M:%S")
        if cat_label == "Not a cornea":
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", "⚠️ Non-Fluorescein Corneal Image Detected.\n\nPlease upload a proper fluorescein slit-lamp image of the cornea.")
            self.history_text.insert("0.0", f"{timestamp} | Invalid image\n")
            self.warning_label.configure(text="⚠️ Invalid Image Detected!", text_color="red")
            self.update_status("Analysis complete - Non-corneal image")
            return

        result_str = (
            f"=== Analysis Results ===\n\n"
            f"Category: {cat_label}\n"
            f"Type: {type_label}\n"
            f"Grade: {grade_label}\n"
            f"Damage Estimate: {damage}%\n\n"
            f"=== Recommendations ===\n"
        )
        if damage > 75:
            result_str += "Immediate specialist referral required.\nUrgent surgical intervention may be needed."
        elif damage > 50:
            result_str += "Prompt ophthalmologist consultation recommended.\nConsider topical antibiotics."
        elif damage > 25:
            result_str += "Schedule follow-up examination.\nMonitor for progression."
        else:
            result_str += "Routine monitoring recommended.\nPatient education advised."
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", result_str)
        self.history_text.insert("0.0", f"{timestamp} | {cat_label} ({damage}%)\n")
        self.warning_label.configure(text="")
        self.update_status("Analysis complete - Results ready")

    def toggle_ui_state(self, enable):
        state = "normal" if enable else "disabled"
        self.upload_btn.configure(state=state)
        # Optionally hide or show progress bar and percentage label
        if enable:
            self.progress_bar.grid_remove()
            self.progress_percentage_label.grid_remove()
        else:
            self.progress_bar.grid()
            self.progress_percentage_label.grid()

    def update_status(self, message):
        self.status_var.set(f"Status: {message}")
        self.update_idletasks()

    def handle_error(self, error):
        messagebox.showerror("Processing Error", f"An error occurred:\n{str(error)}\n\nPlease try another image.")
        self.update_status("Error occurred - Check console for details")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.destroy()

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()
