import cv2
import numpy as np
import os
import time

def find_haarcascade_file():
    cascade_file = 'haarcascade_frontalface_default.xml'
    opencv_cascades_path = cv2.data.haarcascades
    full_path = os.path.join(opencv_cascades_path, cascade_file)
    if os.path.isfile(full_path):
        return full_path

    cascade_paths = [
        cascade_file,
        '/usr/share/opencv4/haarcascades/' + cascade_file,
        '/usr/local/share/opencv4/haarcascades/' + cascade_file,
        '/usr/share/opencv/haarcascades/' + cascade_file,
        '/usr/local/share/opencv/haarcascades/' + cascade_file,
        '/opt/homebrew/opt/opencv/share/opencv4/haarcascades/' + cascade_file,
    ]

    for path in cascade_paths:
        if os.path.isfile(path):
            return path
    return None

cascade_path = find_haarcascade_file()

if cascade_path is None:
    raise FileNotFoundError("No se pudo encontrar el archivo del clasificador Haar Cascade")
print(f"Clasificador encontrado en: {cascade_path}")
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def save_face(frame, face, counter):
    (x, y, w, h) = face
    face_img = frame[y:y + h, x:x + w]
    if not os.path.exists('faces'):
        os.makedirs('faces')
    filename = f'faces/face_{counter}.jpg'
    cv2.imwrite(filename, face_img)
    print(f"Rostro guardado como {filename}")
    return face_img

def compare_faces(face1, face2):
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > 0.8
def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("No se pudo abrir la cámara")
        return
    cv2.namedWindow('Detección de Caras', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detección de Caras', 640, 480)

    frame_count = 0
    start_time = time.time()
    fps = 0
    face_counter = 0
    saved_faces = []

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error al capturar el frame")
            break

        frame_count += 1

        if frame_count % 3 == 0:
            faces = detect_faces(frame)

            for face in faces:
                (x, y, w, h) = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Cara', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                face_img = frame[y:y + h, x:x + w]
                is_new_face = True
                for saved_face in saved_faces:
                    if compare_faces(face_img, saved_face):
                        is_new_face = False
                        break
                if is_new_face:
                    saved_face = save_face(frame, face, face_counter)
                    saved_faces.append(saved_face)
                    face_counter += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detección de Caras', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ocurrió un error: {e}")