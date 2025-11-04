import os
import cv2
import numpy as np
import face_recognition
import datetime
import time
import pickle

# --- CÁC HẰNG SỐ CẤU HÌNH ---
# Ngưỡng khoảng cách tối đa để coi là KHỚP (giá trị càng nhỏ, càng nghiêm ngặt)
TOLERANCE = 0.55
# Thời gian chờ tối thiểu để quét lại cùng một người (áp dụng cho từng cá nhân)
SCAN_COOLDOWN_SECONDS = 10
# Tên file để lưu trữ các vector mã hóa khuôn mặt
ENCODING_FILE = 'encodings.pkl'

# Biến từ điển lưu thời gian điểm danh thành công cuối cùng CỦA MỖI NGƯỜI trong phiên chạy hiện tại
# Dùng để kiểm soát SCAN_COOLDOWN_SECONDS
last_successful_scan_time_per_person = {}

# --- KHỞI TẠO VÀ TẢI DỮ LIỆU ---

path = 'ImageAttendance'
images = []
initialClassNames = []

# Lọc file ảnh hợp lệ từ thư mục
myList = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Files found in {path}: {myList}")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')

    if curImg is not None:
        images.append(curImg)
        initialClassNames.append(os.path.splitext(cl)[0])
    else:
        print(f"WARNING: Không thể load ảnh {cl}")


# --- HÀM TÍNH TOÁN VÀ TẢI MÃ HÓA ---

def FindEncoding(images, names):
    """Tính toán mã hóa nếu file pickle chưa tồn tại, hoặc tải từ file."""

    if os.path.exists(ENCODING_FILE):
        print("TẢI: Đang tải mã hóa từ file .pkl...")
        try:
            with open(ENCODING_FILE, 'rb') as f:
                data = pickle.load(f)
                return data['encodeListKnow'], data['classNames']
        except Exception as e:
            print(f"LỖI: Không thể tải hoặc đọc file {ENCODING_FILE}. Bắt đầu tính toán lại.")
            os.remove(ENCODING_FILE)

    print("TÍNH TOÁN: Bắt đầu tính toán mã hóa khuôn mặt mới...")
    encodeList = []
    finalClassNames = []

    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.array(img_rgb)

        encodes = face_recognition.face_encodings(img_rgb)

        if encodes:
            encodeList.append(encodes[0])
            finalClassNames.append(names[i])
        else:
            print(f"SKIPPING: Không tìm thấy khuôn mặt trong ảnh của {names[i]}.")

    if encodeList:
        data = {'encodeListKnow': encodeList, 'classNames': finalClassNames}
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"LƯU: Đã lưu mã hóa mới vào {ENCODING_FILE}.")

    return encodeList, finalClassNames


# --- HÀM ĐIỂM DANH (GIỚI HẠN 1 PHÚT) ---

def markAttendance(name):
    """
    Ghi điểm danh vào Attendance.csv.
    Cho phép điểm danh nhiều lần với thời gian tối thiểu giữa các lần điểm danh
    và thời gian tối thiểu hiện tại là >= 1 phút.
    """
    now = datetime.datetime.now()
    # Thời gian tối thiểu: 1 phút
    MIN_INTERVAL = datetime.timedelta(minutes=1)

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        last_attendance_time = {}

        for line in myDataList:
            parts = line.split(',')
            if len(parts) >= 2:
                person_name = parts[0].strip().upper()
                dt_string = parts[1].strip()
                try:
                    last_time = datetime.datetime.strptime(dt_string, "%d/%m/%Y %H:%M:%S")
                    last_attendance_time[person_name] = last_time
                except ValueError:
                    continue

        is_eligible = True

        if name.upper() in last_attendance_time:
            time_since_last = now - last_attendance_time[name.upper()]

            # Nếu khoảng cách thời gian nhỏ hơn 1 phút
            if time_since_last < MIN_INTERVAL:
                is_eligible = False
                print(
                    f"Skip điểm danh cho {name}. Cần {MIN_INTERVAL}m.")

        if is_eligible:
            dtString = now.strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{name},{dtString}\n")
            print(f"ATTENDANCE RECORDED: {name} at {dtString}")
            return True  # Trả về True nếu điểm danh thành công

        return False  # Trả về False nếu trong thời gian tối thiểu


# --- XỬ LÝ KHỞI TẠO CUỐI CÙNG ---

encodeListKnow, classNames = FindEncoding(images, initialClassNames)

if len(encodeListKnow) == 0:
    print("FATAL ERROR: Không tìm thấy vector mã hóa khuôn mặt hợp lệ nào!")
    exit()

print(f'Mã hóa hoàn tất. Số người đã biết: {len(classNames)}')
print(f"Tên đã được mã hóa: {classNames}")

cap = cv2.VideoCapture(0)

# --- VÒNG LẶP CHÍNH ---

while True:
    success, img = cap.read()
    if not success:
        print("Lỗi đọc từ webcam.")
        break

    #lật ngang camera
    img = cv2.flip(img, 1)

    show_success_message = False
    current_time = time.time()

    # --- XỬ LÝ NHẬN DẠNG ---

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodesFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodesFace, tolerance=TOLERANCE)
        faceDis = face_recognition.face_distance(encodeListKnow, encodesFace)

        matchIndex = np.argmin(faceDis)

        name = 'UNKNOW'
        color = (0, 0, 255)  # Đỏ mặc định

        # === KIỂM TRA NGƯỠNG ĐỘ CHÍNH XÁC VÀ COOLDOWN CÁ NHÂN ===
        if matches[matchIndex] and faceDis[matchIndex] < TOLERANCE:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)  # Xanh lá

            # Lấy thời gian scan cuối cùng của người này (mặc định là 0 nếu chưa có)
            last_scan_time = last_successful_scan_time_per_person.get(name, 0)

            # Kiểm tra thời gian chờ CỦA RIÊNG NGƯỜI NÀY
            if (current_time - last_scan_time) >= SCAN_COOLDOWN_SECONDS:

                # Gọi hàm điểm danh (kiểm tra logic 1 phút trong CSV)
                if markAttendance(name):
                    # Cập nhật thời gian cooldown CHỈ CHO NGƯỜI NÀY
                    last_successful_scan_time_per_person[name] = current_time
                    show_success_message = True
            else:
                # Nếu đang trong cooldown, chuyển màu vàng và thông báo đang chờ
                color = (0, 255, 255)  # Vàng

        # --- VẼ KHUNG ---

        # Nhân ngược tọa độ (top, right, bottom, left) lên 4
        top, right, bottom, left = faceLoc
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Vẽ khung chính
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        # Vẽ nền cho tên (phía dưới)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

        # Đặt tên
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('My Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()