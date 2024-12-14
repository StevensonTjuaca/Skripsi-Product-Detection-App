import logging
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

# Redirect sys.stdout and sys.stderr for environments like PyInstaller
if not sys.stdout:
    sys.stdout = open(os.devnull, "w")
if not sys.stderr:
    sys.stderr = open(os.devnull, "w")

# Konfigurasi logging
logging.basicConfig(
    filename="debug.log",  # Log akan disimpan di file debug.log
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Aplikasi dimulai.")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Memuat model yang telah dilatih
model_path = resource_path("model_mobilenet_fixed1.h5")
logging.info(f"Path model TensorFlow: {model_path}")
if not os.path.exists(model_path):
    logging.error(f"Model tidak ditemukan: {model_path}")
    sys.exit("Model tidak ditemukan. Pastikan file model tersedia.")

model = tf.keras.models.load_model(model_path)

# Daftar kelas produk yang sesuai dengan output model
product_labels = ["ButterCookies", "Chitato", "Cocacola", "FrisianFlag", "KokoCrunch", "Milkita", "Neoguri", "Silverqueen", "Togo", "Top"]

# MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5)

# Variabel global untuk kamera dan gambar yang di-capture
cap = None
captured_image = None
uploaded_image = None  # Menyimpan gambar yang di-upload di tampilan File
camera_index = 0  # Default kamera internal

# Fungsi untuk pre-processing gambar
def preprocess_image(image):
    try:
        image = image.resize((224, 224))  # Sesuaikan ukuran dengan model
        image_array = np.array(image) / 255.0  # Normalisasi
        return np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    except Exception as e:
        logging.error(f"Error saat pre-processing gambar: {e}", exc_info=True)
        raise

# Fungsi untuk mendeteksi tangan dan melakukan crop
def detect_and_crop_product(image):
    try:
        h, w, _ = image.shape
        section_width = w // 3  # Lebar tiap bagian (1/3 dari lebar gambar)

        # Memproses setiap bagian gambar (kiri, tengah, kanan)
        for i in range(3):
            # Tentukan batas kiri dan kanan dari setiap bagian
            left = i * section_width
            right = (i + 1) * section_width if i < 2 else w  # Bagian terakhir sampai ujung kanan gambar
            image_section = image[:, left:right]  # Ambil bagian gambar secara horizontal

            # Konversi warna untuk deteksi tangan
            rgb_image = cv2.cvtColor(image_section, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            # Jika tangan terdeteksi dalam bagian ini
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                x_min, y_min = section_width, h
                x_max, y_max = 0, 0

                # Menghitung bounding box untuk area tangan di bagian ini
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * section_width), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Menambahkan margin di sekitar tangan
                margin = 50
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(section_width, x_max + margin)
                y_max = min(h, y_max + margin)

                # Crop area produk dengan memfokuskan lebih pada area yang lebih besar
                product_area = image_section[y_min:y_max, x_min:x_max]
                return product_area  # Mengembalikan crop pada bagian yang terdeteksi

        # Jika tidak ada tangan terdeteksi di semua bagian, kembalikan gambar utuh
        return image
    except Exception as e:
        logging.error(f"Error saat deteksi dan crop produk: {e}", exc_info=True)
        raise

# Memastikan file logo dan background tersedia
logo_path = resource_path("Logo.png")
background_path = resource_path("background.jpg")
logging.info(f"Path logo: {logo_path}")
logging.info(f"Path background: {background_path}")
if not os.path.exists(logo_path):
    logging.error(f"Logo tidak ditemukan: {logo_path}")
    sys.exit("Logo tidak ditemukan. Pastikan file logo tersedia.")
if not os.path.exists(background_path):
    logging.error(f"Background tidak ditemukan: {background_path}")
    sys.exit("Background tidak ditemukan. Pastikan file background tersedia.")


# === Tampilan File ===
def open_file_menu():
    home_frame.pack_forget()
    file_frame.pack(pady=20)

def go_back_to_home():
    global cap, captured_image
    if cap:
        cap.release()
    captured_image = None  # Reset captured image saat kembali ke Home
    file_frame.pack_forget()
    camera_frame.pack_forget()
    home_frame.pack(pady=20)
    img_label.config(image="")
    img_label.image = None
    result_text_file.set("Hasil pengenalan akan muncul di sini.")
    start_button_file.config(state="disabled")

def upload_image():
    global uploaded_image
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            logging.info(f"Gambar berhasil diunggah dari: {file_path}")
            uploaded_image = Image.open(file_path)
            if uploaded_image.mode == 'RGBA':
                uploaded_image = uploaded_image.convert('RGB')
            uploaded_image.thumbnail((250, 250))
            img = ImageTk.PhotoImage(uploaded_image)
            img_label.config(image=img)
            img_label.image = img
            start_button_file.config(state="normal")
            result_text_file.set("Gambar berhasil diunggah. Klik Mulai Pengenalan untuk mendeteksi produk.")
        else:
            logging.warning("Tidak ada file yang dipilih.")
    except Exception as e:
        logging.error(f"Error saat mengunggah gambar: {e}")

# Fungsi tambahan untuk validasi gambar
def validate_image(image):
    try:
        if image is None or image.size == 0:
            logging.error("Gambar tidak valid atau kosong.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error saat validasi gambar: {e}", exc_info=True)
        return False

# Fungsi start_detection_file yang diperbarui
def start_detection_file():
    global uploaded_image
    try:
        if uploaded_image is None:
            logging.error("Gambar tidak ditemukan.")
            result_text_file.set("Gambar belum diunggah.")
            return

        if not validate_image(uploaded_image):
            result_text_file.set("Gambar tidak valid atau kosong.")
            return

        # Preprocess image
        input_image = preprocess_image(uploaded_image)
        logging.info(f"Input image shape: {input_image.shape}")

        # Model prediction
        predictions = model.predict(input_image, verbose=0)[0]
        logging.info(f"Hasil prediksi: {predictions}")

        # Process predictions
        detected_products = []
        confidence_threshold = 0.1  # Threshold keyakinan
        for index, confidence in enumerate(predictions):
            if confidence >= confidence_threshold:
                product_name = product_labels[index]
                detected_products.append((product_name, confidence))

        detected_products.sort(key=lambda x: x[1], reverse=True)

        if detected_products:
            detected_product_text = "\n".join(
                [f"{name} ({conf * 100:.2f}%)" for name, conf in detected_products]
            )
            result_text_file.set(
                f"Produk Terdeteksi:\n{detected_product_text}\nJumlah Produk: {len(detected_products)}"
            )
            logging.info(f"Produk terdeteksi: {detected_products}")
        else:
            result_text_file.set("Tidak ada produk terdeteksi.")
            logging.warning("Tidak ada produk yang terdeteksi.")

    except Exception as e:
        logging.error(f"Error saat deteksi produk: {e}", exc_info=True)
        result_text_file.set("Terjadi kesalahan saat deteksi produk.")


# === Tampilan Camera ===
def open_camera_menu():
    home_frame.pack_forget()
    camera_frame.pack(pady=20)
    restart_camera()

def restart_camera():
    global cap, captured_image, camera_index
    captured_image = None  # Reset captured image setiap kali kamera di-restart
    result_text_camera.set("Hasil pengenalan akan muncul di sini.")
    if cap is not None:
        cap.release()  # Pastikan kamera tidak tetap menyala
    cap = cv2.VideoCapture(camera_index)  # Menggunakan kamera berdasarkan pilihan user
    update_camera_feed()
    # Menampilkan tombol yang tepat saat memasuki tampilan kamera
    capture_button.pack(pady=10)
    detect_button_camera.pack_forget()  # Sembunyikan tombol deteksi
    recapture_button.pack_forget()  # Sembunyikan tombol recapture

def update_camera_feed():
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)
        camera_frame.after(10, update_camera_feed)

def capture_image():
    global captured_image
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Ambil dan crop gambar
            captured_image = detect_and_crop_product(frame)
            display_captured_image()  # Tampilkan hasil crop
            cap.release()  # Tutup kamera setelah mengambil gambar
            capture_button.pack_forget()  # Sembunyikan tombol capture
            detect_button_camera.pack(pady=10)  # Tampilkan tombol untuk memulai pengenalan
        else:
            print("Gagal mengambil gambar dari kamera.")

def display_captured_image():
    if captured_image is not None:
        cropped_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
        img = Image.fromarray(cropped_rgb)  # Buat image dari array
        imgtk = ImageTk.PhotoImage(image=img)  # Konversi ke PhotoImage
        camera_label.config(image=imgtk)  # Tampilkan di label
        camera_label.image = imgtk  # Simpan referensi ke image
    else:
        print("Gambar tidak ditemukan untuk ditampilkan.")

# Fungsi start_detection_on_captured_image yang diperbarui
def start_detection_on_captured_image():
    global captured_image
    try:
        if captured_image is None:
            logging.error("Gambar belum di-capture.")
            result_text_camera.set("Gambar belum di-capture.")
            return

        if not validate_image(captured_image):
            result_text_camera.set("Gambar tidak valid atau kosong.")
            return

        # Convert captured_image to PIL Image for preprocessing
        captured_pil = Image.fromarray(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
        logging.info("Gambar berhasil di-capture untuk deteksi.")

        # Preprocess image
        input_image = preprocess_image(captured_pil)
        logging.info(f"Input image shape: {input_image.shape}")

        # Model prediction
        predictions = model.predict(input_image, verbose=0)[0]
        logging.info(f"Hasil prediksi: {predictions}")

        # Process predictions
        detected_products = []
        confidence_threshold = 0.1  # Threshold keyakinan
        for index, confidence in enumerate(predictions):
            if confidence >= confidence_threshold:
                product_name = product_labels[index]
                detected_products.append((product_name, confidence))

        detected_products.sort(key=lambda x: x[1], reverse=True)

        if detected_products:
            top_product = detected_products[0]
            detected_product_text = f"{top_product[0]} ({top_product[1] * 100:.2f}%)"
            result_text_camera.set(f"Produk Terdeteksi:\n{detected_product_text}")
            logging.info(f"Produk terdeteksi: {detected_products}")
        else:
            result_text_camera.set("Tidak ada produk terdeteksi.")
            logging.warning("Tidak ada produk yang terdeteksi.")

    except Exception as e:
        logging.error(f"Error saat deteksi produk: {e}", exc_info=True)
        result_text_camera.set("Terjadi kesalahan saat deteksi produk.")

        # Sembunyikan tombol "Mulai Pengenalan" dan tampilkan tombol "Camera"
        detect_button_camera.pack_forget()
        recapture_button.pack(pady=10)


# About button callback
def about_app():
    about_message = """
    Aplikasi Deteksi Produk ini dibuat menggunakan Python dengan framework TensorFlow dan MediaPipe untuk deteksi tangan serta MobileNet untuk klasifikasi produk.
    
    **Fitur Utama**:
    - Deteksi produk dengan bantuan kamera dan gambar statis.
    - Pengenalan produk dengan model deep learning yang dilatih menggunakan dataset produk tertentu.

    **Teknologi yang Digunakan**:
    - TensorFlow: Untuk membangun dan melatih model deep learning.
    - MediaPipe: Untuk deteksi tangan berbasis landmark.
    - OpenCV: Untuk manipulasi gambar dan kamera.
    - Tkinter: Untuk antarmuka pengguna berbasis GUI.

    **Spesifikasi Minimum**:
    - RAM: 16 GB
    - Prosesor: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz, 2688 Mhz, 6 Core(s), 12 Logical Processor(s)
    - NVIDIA RTX 3050 TI: NVIDIA dengan CUDA untuk mempercepat inferensi model
    - Resolusi Kamera: Minimum 720p untuk hasil terbaik

    **Pengembang**:
    Stevenson Tjuaca
    Email: tjuacastevenson@gmail.com
    """
    messagebox.showinfo("About", about_message)

# Help button callback
def help_app():
    help_message = """
    **Petunjuk Penggunaan Aplikasi**:
    
    **Mode File**:
    1. Klik tombol "File" di menu utama.
    2. Klik "Upload Gambar" untuk mengunggah file gambar.
    3. Gambar yang berhasil diunggah akan muncul di layar.
    4. Klik "Mulai Pengenalan" untuk mendeteksi produk dalam gambar.
    5. Hasil pengenalan akan ditampilkan di bawah gambar.

    **Mode Kamera**:
    1. Klik tombol "Camera" di menu utama.
    2. Kamera akan menyala dan menampilkan feed langsung.
    3. Klik "Capture" untuk mengambil gambar dari kamera.
    4. Klik "Mulai Pengenalan" untuk mendeteksi produk dalam gambar yang diambil.
    5. Jika ingin mencoba ulang, klik tombol "Camera" untuk restart kamera.

    **Informasi Tambahan**:
    - Pastikan gambar atau kamera memiliki pencahayaan yang baik untuk hasil terbaik.
    - Jika aplikasi tidak mendeteksi produk, periksa apakah gambar memuat produk yang sesuai dengan model.

    Jika Anda mengalami kendala, hubungi tim pengembang melalui email yang tertera di bagian "About".
    """
    messagebox.showinfo("Help", help_message)

# === GUI Setup ===
root = tk.Tk()
root.title("Product Detection App")
root.geometry("800x600")
root.configure(bg="#2b2b2b")  # Background color

bg_image_path = resource_path("background.jpg")
logging.info(f"Path background image: {bg_image_path}")
bg_image = Image.open(resource_path("background.jpg"))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Menambahkan elemen UI lainnya
# Load the logo image
logo_path = resource_path("logo.png")
logging.info(f"Path logo: {logo_path}")
logo_image = Image.open(resource_path("logo.png"))
logo_image = logo_image.resize((80, 80), Image.LANCZOS)  # Resize the logo to fit the UI
logo_photo = ImageTk.PhotoImage(logo_image)

# Display the logo on the Tkinter window
logo_label = tk.Label(root, image=logo_photo, bg=None, borderwidth=0, highlightthickness=0)
logo_label.image = logo_photo  # Keep a reference to the image
logo_label.pack(pady=10) 

# Warna dan font kustom
primary_color = "#00ADB5"  
secondary_color = "SkyBlue"  
font_style = ("Georgia", 12, "bold")

# Frame Home
home_frame = tk.Frame(root, bg="white", highlightthickness=0)
home_frame.pack(pady=20)

file_button = tk.Button(home_frame, text="File", font=font_style, command=open_file_menu, width=15, height=2, bg="#00ADB5", fg="white", activebackground="#FF5722", highlightthickness=0)
file_button.pack(pady=5)

camera_button = tk.Button(home_frame, text="Camera", font=font_style, command=open_camera_menu, width=15, height=2, bg="#00ADB5", fg="white", activebackground="#FF5722", highlightthickness=0)
camera_button.pack(pady=5)

button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=20)

about_button = tk.Button(
    button_frame, 
    text="About", 
    font=font_style, 
    command=about_app,  # Menggunakan fungsi about_app
    width=10, 
    bg="#00ADB5", 
    fg="white", 
    activebackground="#00ADB5", 
    highlightthickness=0
)
about_button.grid(row=0, column=0, padx=10)

help_button = tk.Button(
    button_frame, 
    text="Help", 
    font=font_style, 
    command=help_app,  # Menggunakan fungsi help_app
    width=10, 
    bg="#00ADB5", 
    fg="white", 
    activebackground="#00ADB5", 
    highlightthickness=0
)
help_button.grid(row=0, column=1, padx=10)

camera_options = ["Internal Camera", "External Camera"]
selected_camera = tk.StringVar(value=camera_options[0])

def set_camera_index(value):
    global camera_index
    camera_index = 0 if value == "Internal Camera" else 1

camera_dropdown = tk.OptionMenu(home_frame, selected_camera, *camera_options, command=set_camera_index)
camera_dropdown.config(width=20, font=("Helvetica", 10), bg=primary_color, fg="white", activebackground=secondary_color)
camera_dropdown.pack(pady=10)

# Frame untuk tampilan File
file_frame = tk.Frame(root, bg="white")
back_button_file = tk.Button(file_frame, text="Back to Home", font=font_style, command=go_back_to_home, bg=primary_color, fg="white", activebackground=secondary_color)
back_button_file.pack(pady=5)

upload_button = tk.Button(file_frame, text="Upload Gambar", font=font_style, command=upload_image, bg="#00ADB5", fg="white", activebackground=secondary_color)
upload_button.pack(pady=10)

img_label = tk.Label(file_frame, bg="#222831")
img_label.pack(pady=10)

start_button_file = tk.Button(file_frame, text="Mulai Pengenalan", font=font_style, command=start_detection_file, state="disabled", bg=secondary_color, fg="white", activebackground=primary_color)
start_button_file.pack(pady=10)

result_text_file = tk.StringVar()
result_text_file.set("Hasil pengenalan akan muncul di sini.")
result_label_file = tk.Label(file_frame, textvariable=result_text_file, font=font_style, wraplength=500, justify="left", fg="white", bg="#2b2b2b")
result_label_file.pack(pady=10)

# Frame untuk tampilan Camera
camera_frame = tk.Frame(root, bg="#2b2b2b")
camera_back_button = tk.Button(camera_frame, text="Back to Home", font=font_style, command=go_back_to_home, bg=primary_color, fg="white", activebackground=secondary_color)
camera_back_button.pack(pady=5)

camera_label = tk.Label(camera_frame, bg="#1c1c1c", width=400, height=300)  # Adjusted to a smaller size
camera_label.pack(pady=10)

capture_button = tk.Button(camera_frame, text="Capture", font=font_style, command=capture_image, bg=secondary_color, fg="white", activebackground=primary_color)
capture_button.pack(pady=10)

detect_button_camera = tk.Button(camera_frame, text="Mulai Pengenalan", font=font_style, command=start_detection_on_captured_image, bg="#3c3c3c", fg="white", activebackground=primary_color)
detect_button_camera.pack_forget()

recapture_button = tk.Button(camera_frame, text="Camera", font=font_style, command=restart_camera, bg=primary_color, fg="white", activebackground=secondary_color)
recapture_button.pack_forget()

result_text_camera = tk.StringVar()
result_text_camera.set("Hasil pengenalan akan muncul di sini.")
result_label_camera = tk.Label(camera_frame, textvariable=result_text_camera, font=font_style, wraplength=500, justify="left", fg="white", bg="#2b2b2b")
result_label_camera.pack(pady=10)

root.mainloop()
