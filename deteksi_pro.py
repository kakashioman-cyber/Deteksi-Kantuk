import cv2
import dlib
import csv
import time
from scipy.spatial import distance

# Membuat file CSV dan menulis header-nya
with open('data_kantuk.csv', mode='w', newline='') as file: # Membuka file CSV untuk menulis data
    writer = csv.writer(file) # Membuat objek writer untuk menulis ke file CSV
    writer.writerow(['Timestamp', 'EAR_Value', 'Status']) # Menulis header kolom ke file CSV

# --- PRE-PROCESSING ---
def preprocess_frame(frame):
    # Mengubah ke grayscale & Histogram Equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # konversi ke grayscale
    gray = cv2.equalizeHist(gray) # meningkatkan kontras untuk deteksi wajah yang lebih baik
    return gray

# --- FEATURE EXTRACTION ---
def calculate_ear(eye_points):
    # Rumus EAR untuk ekstraksi fitur
    A = distance.euclidean(eye_points[1], eye_points[5]) # jarak vertikal antara titik 2-6
    B = distance.euclidean(eye_points[2], eye_points[4]) # jarak vertikal antara titik 3-5
    C = distance.euclidean(eye_points[0], eye_points[3]) # jarak horizontal antara titik 1-4
    ear = (A + B) / (2.0 * C) # menghitung EAR
    return ear

# --- CLASSIFICATION ---
def classify_status(ear, threshold=0.25): # threshold EAR untuk menentukan status kantuk
    # Logika klasifikasi
    if ear < threshold: # jika EAR di bawah threshold, dianggap mengantuk
        return "MENGANTUK"
    else:
        return "WASPADA"

# --- MAIN PROGRAM ---
def main():
    # Load dlib (Inisialisasi)
    detector = dlib.get_frontal_face_detector() # detektor wajah frontal dari dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # model prediktor landmark wajah
    
    cap = cv2.VideoCapture(0) # membuka webcam (0 untuk kamera default)
    
    while True:
        ret, frame = cap.read() # membaca frame dari webcam
        if not ret: break
        
        gray = preprocess_frame(frame) # pre-processing frame untuk deteksi wajah yang lebih baik
        faces = detector(gray) # mendeteksi wajah dalam frame yang sudah diproses
        
        # Memanggil fungsi feature_extraction dan classify_status untuk setiap wajah yang terdeteksi
        avg_ear = 0.0
        status = "TIDAK TERDETEKSI"


        if len(faces) > 0: # Cek apakah ada wajah
            for face in faces:
                # Ekstraksi fitur menggunakan dlib untuk mendapatkan landmark wajah
                landmarks = predictor(gray, face)
            
                # Mengambil titik landmark untuk mata kiri dan kanan (indeks standar dlib)
                # Mata kiri: 36-41, Mata kanan: 42-47
                left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
                # Menghitung EAR untuk kedua mata
                ear_left = calculate_ear(left_eye_points)
                ear_right = calculate_ear(right_eye_points)
                avg_ear = (ear_left + ear_right) / 2.0
            
                # Visualisasi titik mata
                for (x, y) in left_eye_points + right_eye_points:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
                # Klasifikasi
                status = classify_status(avg_ear)
            
                # Tampilkan status di layar
                cv2.putText(frame, f"EAR: {avg_ear:.2f} | Status: {status}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
                if status == "MENGANTUK":
                    color = (0, 0, 255)  # Merah (B=0, G=0, R=255)
                else:
                    color = (0, 255, 0)  # Hijau (B=0, G=255, R=0)

                # Menggunakan variabel 'color' tersebut saat menggambar teks
                cv2.putText(frame, f"EAR: {avg_ear:.2f} | Status: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
            # Menyimpan ke CSV setiap frame (atau diberi jeda agar file tidak terlalu besar)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # mendapatkan timestamp saat ini
            with open('data_kantuk.csv', mode='a', newline='') as file: # Membuka file CSV untuk menambahkan data
                writer = csv.writer(file) 
                writer.writerow([timestamp, f"{avg_ear:.4f}", status])
        
        cv2.imshow("Deteksi Kantuk", frame) # menampilkan frame dengan hasil deteksi kantuk
        if cv2.waitKey(1) & 0xFF == ord('q'): break # keluar jika tombol 'q' ditekan
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program ditutup dengan aman. Data tersimpan.")

if __name__ == "__main__":
    main()