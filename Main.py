import os
import cv2
import numpy as np
import face_recognition
import datetime
import time
import pickle
import streamlit as st
import pandas as pd

# --- CÁC HẰNG SỐ CẤU HÌNH ---
# Ngưỡng khoảng cách tối đa để coi là KHỚP (giá trị càng nhỏ, càng nghiêm ngặt)
TOLERANCE = 0.55
# Thời gian chờ tối thiểu để quét lại cùng một người (áp dụng cho từng cá nhân)
SCAN_COOLDOWN_SECONDS = 60
# Tên file để lưu trữ các vector mã hóa khuôn mặt
ENCODING_FILE = 'encodings.pkl'
# Đường dẫn thư mục chứa ảnh khuôn mặt đã biết
IMAGE_PATH = 'ImageAttendance'
# Tên file ghi điểm danh
ATTENDANCE_FILE = 'Attendance.csv'

# Biến từ điển lưu thời gian điểm danh thành công cuối cùng CỦA MỖI NGƯỜI trong phiên chạy hiện tại
# Dùng để kiểm soát SCAN_COOLDOWN_SECONDS
if 'last_successful_scan_time_per_person' not in st.session_state:
    st.session_state.last_successful_scan_time_per_person = {}
# --- KHỞI TẠO STATE MỚI CHO CHỤP ẢNH CAMERA ---
if 'capture_flag' not in st.session_state:
    st.session_state.capture_flag = False
if 'face_register_name' not in st.session_state:
    st.session_state.face_register_name = ""
if 'upload_status' not in st.session_state:
    st.session_state.upload_status = ('', '')  # Khởi tạo trạng thái thông báo

# ĐẶT MẶC ĐỊNH CAMERA LUÔN CHẠY
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = True


# --- Hàm tải dữ liệu lịch sử điểm danh (ĐÃ BỎ CACHE) ---
# Bỏ @st.cache_data để hàm này luôn chạy và tải dữ liệu mới nhất
def load_attendance_data():
    """Tải dữ liệu lịch sử điểm danh từ CSV."""
    if os.path.exists(ATTENDANCE_FILE):
        try:
            # Đảm bảo file có header nếu nó rỗng
            if os.path.getsize(ATTENDANCE_FILE) == 0:
                with open(ATTENDANCE_FILE, 'w') as f:
                    f.write("Name,Time\n")

            df = pd.read_csv(ATTENDANCE_FILE)
            return df
        except pd.errors.EmptyDataError:
            # Xử lý trường hợp file tồn tại nhưng không có dữ liệu (trừ header)
            return pd.DataFrame(columns=['Name', 'Time'])
        except Exception:
            st.error("Lỗi khi đọc file lịch sử điểm danh.")
            return pd.DataFrame(columns=['Name', 'Time'])
    return pd.DataFrame(columns=['Name', 'Time'])  # Trả về DataFrame rỗng nếu file không tồn tại


# --- HÀM HIỂN THỊ LỊCH SỬ ĐIỂM DANH TRONG PLACEHOLDER ---
def display_attendance_history(placeholder):
    """
    Tải dữ liệu mới và cập nhật placeholder hiển thị lịch sử điểm danh.
    """
    df = load_attendance_data()
    # Xóa nội dung cũ trong placeholder và vẽ lại
    with placeholder.container():
        st.subheader("Lịch sử Điểm danh")
        if not df.empty:
            # Chỉ hiển thị 10 dòng cuối, sắp xếp ngược để dòng mới nhất ở trên
            st.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("Lịch sử điểm danh đang trống.")


# --- HÀM TÍNH TOÁN VÀ TẢI MÃ HÓA (Giữ nguyên) ---

@st.cache_resource
def FindEncoding(path):
    """Tính toán mã hóa nếu file pickle chưa tồn tại, hoặc số lượng tên đã biết bị thay đổi, hoặc tải từ file."""
    images = []
    initialClassNames = []

    if not os.path.exists(path):
        st.error(f"Thư mục ảnh '{path}' không tồn tại. Vui lòng tạo thư mục và thêm ảnh.")
        return [], []

    # Lọc file ảnh hợp lệ từ thư mục
    myList = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    st.info(f"Đã tìm thấy {len(myList)} file ảnh trong '{path}'.")

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')

        if curImg is not None:
            images.append(curImg)
            initialClassNames.append(os.path.splitext(cl)[0])
        else:
            print(f"WARNING: Không thể load ảnh {cl}")

    # Tải dữ liệu từ file nếu tồn tại
    if os.path.exists(ENCODING_FILE):
        print("TẢI: Đang kiểm tra mã hóa từ file .pkl...")
        try:
            with open(ENCODING_FILE, 'rb') as f:
                data = pickle.load(f)

                if len(initialClassNames) == len(data['classNames']):
                    print("TẢI: Số lượng khuôn mặt khớp. Tải thành công.")
                    return data['encodeListKnow'], data['classNames']
                else:
                    print(
                        f"THAY ĐỔI: Số lượng khuôn mặt thay đổi. Buộc tính toán lại.")
                    os.remove(ENCODING_FILE)

        except Exception as e:
            print(f"LỖI: Không thể tải hoặc đọc file {ENCODING_FILE}. Bắt đầu tính toán lại.")
            if os.path.exists(ENCODING_FILE):
                os.remove(ENCODING_FILE)

    st.warning("TÍNH TOÁN: Bắt đầu tính toán mã hóa khuôn mặt mới...")
    encodeList = []
    finalClassNames = []

    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Chỉ lấy vị trí khuôn mặt đầu tiên để tính encoding (giảm tải xử lý)
        face_locs = face_recognition.face_locations(img_rgb)

        if face_locs:
            encodes = face_recognition.face_encodings(img_rgb, face_locs)
            if encodes:
                encodeList.append(encodes[0])
                finalClassNames.append(initialClassNames[i])
        else:
            print(f"SKIPPING: Không tìm thấy khuôn mặt trong ảnh của {initialClassNames[i]}.")

    if encodeList:
        data = {'encodeListKnow': encodeList, 'classNames': finalClassNames}
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump(data, f)
        st.success(f"LƯU: Đã lưu {len(finalClassNames)} mã hóa mới vào {ENCODING_FILE}.")
    else:
        st.error("LỖI: Không tìm thấy vector mã hóa khuôn mặt hợp lệ nào!")

    return encodeList, finalClassNames


# --- HÀM GHI ĐIỂM DANH ---

def markAttendance(name):
    """
    Ghi điểm danh vào Attendance.csv, kiểm tra khoảng thời gian tối thiểu 1 phút.
    Trả về True nếu điểm danh thành công, False nếu không đủ điều kiện.
    """
    now = datetime.datetime.now()
    MIN_INTERVAL = datetime.timedelta(minutes=2)

    # Đọc toàn bộ dữ liệu hiện có
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w') as f:
            f.write("Name,Time\n")

    with open(ATTENDANCE_FILE, 'r') as f:
        myDataList = f.readlines()
        last_attendance_time = {}

        for line in myDataList:
            parts = line.split(',')
            if len(parts) >= 2 and parts[0].strip().upper() == name.upper():
                dt_string = parts[1].strip()
                try:
                    # Chú ý: Cần xử lý lỗi ValueError nếu datetime.strptime không khớp định dạng
                    last_time = datetime.datetime.strptime(dt_string.split('\n')[0].strip(), "%d/%m/%Y %H:%M:%S")
                    last_attendance_time[name.upper()] = last_time
                except ValueError:
                    # Bỏ qua dòng bị lỗi định dạng
                    continue

        is_eligible = True

        if name.upper() in last_attendance_time:
            time_since_last = now - last_attendance_time[name.upper()]

            if time_since_last < MIN_INTERVAL:
                is_eligible = False

        if is_eligible:
            dtString = now.strftime("%d/%m/%Y %H:%M:%S")
            # Ghi dòng mới vào cuối file
            with open(ATTENDANCE_FILE, 'a') as f:
                f.write(f"{name},{dtString}\n")

            return True  # Trả về điểm danh thành công

        return False  # Trả về điểm danh không thành công


# --- HÀM LƯU FILE TẢI LÊN ---
def save_uploaded_file(uploaded_file, name):
    """Lưu file được tải lên vào thư mục ImageAttendance với tên là 'name.extension'."""
    if not name or not uploaded_file:
        return False, "Vui lòng nhập Tên và chọn Ảnh."

    # Lấy phần mở rộng gốc của file
    _, file_extension = os.path.splitext(uploaded_file.name)

    # Tạo tên file mới dựa trên tên người dùng (chuẩn hóa tên: thay thế khoảng trắng bằng dấu gạch dưới)
    safe_name = name.strip().replace(' ', '_')
    if not safe_name:
        return False, "Tên không hợp lệ."

    filename = f"{safe_name}{file_extension}"
    filepath = os.path.join(IMAGE_PATH, filename)

    # Kiểm tra nếu file đã tồn tại
    if os.path.exists(filepath):
        return False, f"File ảnh cho '{name}' đã tồn tại. Vui lòng đổi tên hoặc xóa file cũ."

    # Ghi file
    try:
        if not os.path.exists(IMAGE_PATH):
            os.makedirs(IMAGE_PATH)

        # Streamlit uploaded file có phương thức getbuffer()
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, f"Đã lưu ảnh kéo thả cho **{name}** thành công! Vui lòng nhấn **TẢI LẠI MÃ HÓA**."
    except Exception as e:
        return False, f"Lỗi khi lưu file: {e}"


# --- HÀM LƯU TOÀN BỘ KHUNG HÌNH (FULL FRAME) ĐỂ ĐĂNG KÝ MỚI ---
def save_full_frame_for_registration(img_bgr, name):
    """
    Lưu toàn bộ khung hình camera (full frame) dưới dạng JPG để đăng ký khuôn mặt mới.
    Không cắt khuôn mặt.
    """
    if not name:
        return False, "Vui lòng nhập Tên người để lưu ảnh."

    # Chuẩn hóa tên file
    safe_name = name.strip().replace(' ', '_')
    filename = f"{safe_name}.jpg"  # Lưu luôn dưới dạng JPG
    filepath = os.path.join(IMAGE_PATH, filename)

    if os.path.exists(filepath):
        return False, f"Ảnh cho '{name}' đã tồn tại. Vui lòng đổi tên hoặc xóa file cũ."

    # Lưu ảnh (toàn bộ khung hình)
    try:
        if not os.path.exists(IMAGE_PATH):
            os.makedirs(IMAGE_PATH)

        # Lưu trực tiếp ảnh BGR đã lật (img_bgr)
        cv2.imwrite(filepath, img_bgr)

        return True, f"Đã chụp và lưu **TOÀN BỘ KHUNG ẢNH** cho **{name}** thành công! Vui lòng nhấn **TẢI LẠI MÃ HÓA**."
    except Exception as e:
        return False, f"Lỗi khi lưu file: {e}"


# --- CHỨC NĂNG CHÍNH CỦA STREAMLIT ---

def app_main():
    st.set_page_config(page_title="Hệ thống điểm danh khuôn mặt Streamlit", layout="wide")
    st.title("👨‍🏫 Hệ thống Điểm danh Tự động (Streamlit + OpenCV)")

    # Sử dụng st.cache_resource để chỉ chạy hàm này một lần
    encodeListKnow, classNames = FindEncoding(IMAGE_PATH)

    # Cấu hình sidebar
    st.sidebar.header("Cấu hình")
    st.sidebar.markdown(f"**Số người đã biết:** {len(classNames)}")
    st.sidebar.markdown(f"**Ngưỡng chính xác (Tolerance):** {TOLERANCE}")
    st.sidebar.markdown(f"**Cooldown (giây):** {SCAN_COOLDOWN_SECONDS}")

    # --- VÙNG HIỂN THỊ CAMERA VÀ TRẠNG THÁI ---
    col1, col2 = st.columns([2, 1])

    with col2:
        # --- PHẦN QUẢN LÝ ẢNH ---
        st.subheader("Quản lý Ảnh Khuôn mặt Mới")

        # NAME INPUT (Dùng chung cho cả 2 phương pháp)
        st.session_state.face_register_name = st.text_input(
            "1. Nhập Tên người mới (dùng chung cho cả 2 cách):",
            key="master_name_input"
        )

        # ----------------------------------------------------
        st.markdown("##### 2a. Đăng ký bằng File Kéo thả (Lưu ảnh gốc)")
        # Dùng form để xử lý submit rõ ràng và xóa input sau khi submit
        with st.form("upload_form", clear_on_submit=True):
            new_image = st.file_uploader("Chọn Ảnh Khuôn mặt (PNG/JPG):", type=['png', 'jpg', 'jpeg'],
                                         key="upload_file_input")

            submitted = st.form_submit_button("LƯU ẢNH KÉO THẢ")

            if submitted:
                # Dùng tên từ state chung
                name_to_use = st.session_state.face_register_name
                success, msg = save_uploaded_file(new_image, name_to_use)
                st.session_state.upload_status = ('success' if success else 'error', msg)

        # ----------------------------------------------------
        st.markdown("##### 2b. Đăng ký bằng Camera Trực tiếp (Lưu TOÀN BỘ khung hình)")
        # Button for Camera Capture (outside a form for simplicity)
        if st.button("📷 CHỤP ẢNH TỪ CAMERA", use_container_width=True):
            # Kiểm tra xem có đang chạy camera không (mặc dù luôn chạy, nhưng phòng trường hợp lỗi)
            if not st.session_state.run_camera:
                st.session_state.upload_status = ('error',
                                                  "Camera đang tắt. Vui lòng kiểm tra lại thiết bị hoặc khởi động lại ứng dụng.")
            elif not st.session_state.face_register_name.strip():
                st.session_state.upload_status = ('error', "Vui lòng nhập tên người mới ở bước 1.")
            else:
                # Set the flag for the camera loop to capture the next frame
                st.session_state.capture_flag = True
                st.session_state.upload_status = ('info',
                                                  f"Đang chờ khuôn mặt của **{st.session_state.face_register_name}** và chụp...")

        # Hiển thị thông báo sau khi upload/capture
        if 'upload_status' in st.session_state:
            type, msg = st.session_state.upload_status
            if type == 'success':
                st.success(msg)
            elif type == 'error':
                st.error(msg)
            elif type == 'info':
                # Chỉ hiển thị info nếu capture_flag vẫn BẬT
                if st.session_state.capture_flag:
                    st.info(msg)
            # Giữ lại thông báo cho đến khi tương tác mới

        st.markdown("---")
        st.warning("Sau khi thêm/xóa ảnh, bạn **PHẢI** nhấn nút dưới đây để hệ thống nhận diện khuôn mặt mới.")

        # Nút để xóa cache và buộc ứng dụng chạy lại
        if st.button("🔄 TẢI LẠI MÃ HÓA (RE-ENCODE)", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()  # Buộc Streamlit chạy lại từ đầu để tính toán lại encodings

        st.markdown("---")

        st.subheader("Trạng thái Điểm danh")
        # Placeholder cho tin nhắn trạng thái
        status_message = st.empty()

        # Hiển thị danh sách tên đã biết
        st.subheader("Danh sách Khuôn mặt Đã Biết")
        st.markdown(", ".join(classNames) if classNames else "Chưa có khuôn mặt nào được mã hóa.")

        # --- TẠO PLACEHOLDER CHO LỊCH SỬ ĐIỂM DANH ---
        history_placeholder = st.empty()

        # Hiển thị lịch sử lần đầu tiên
        display_attendance_history(history_placeholder)

    with col1:
        st.subheader("Camera Nhận diện")

        # Placeholder cho khung hình camera
        frame_placeholder = st.empty()

    # --- LOGIC VÒNG LẶP CAMERA CHÍNH (LUÔN CHẠY) ---
    if st.session_state.run_camera:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("LỖI: Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập và làm mới trang.")
            st.session_state.run_camera = False  # Đặt về False để không cố gắng chạy lại ngay lập tức
            return

        # Vòng lặp xử lý camera
        while st.session_state.run_camera:

            success, img = cap.read()
            if not success:
                status_message.error("Lỗi đọc từ webcam.")
                time.sleep(1)
                continue

            # Lật ngang camera (img_bgr)
            img = cv2.flip(img, 1)

            current_time_sec = time.time()

            # --- CHUẨN BỊ CHO NHẬN DẠNG (EXPENSIVE: CHẠY TRÊN MỌI KHUNG HÌNH) ---
            # Giảm kích thước ảnh để tăng tốc độ xử lý (luôn chạy trên mọi khung hình)
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # HEAVY COMPUTATION: Tìm vị trí khuôn mặt và mã hóa
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            attendance_status_updated = False
            name_to_register_upper = st.session_state.face_register_name.upper()

            # Vòng lặp này chạy qua tất cả khuôn mặt được tìm thấy trong khung hình
            for encodesFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

                name = 'UNKNOW'
                color = (0, 0, 255)  # Đỏ mặc định (UNKNOW)

                # --- CHẾ ĐỘ ƯU TIÊN: ĐANG CHỤP ẢNH MỚI (Visual Feedback) ---
                if st.session_state.capture_flag:
                    # Nếu cờ đang bật, chúng ta đang chờ khuôn mặt để chụp.
                    name = f"SẴN SÀNG CHỤP: {name_to_register_upper}"
                    color = (0, 255, 255)  # Màu Vàng sáng để báo hiệu sẵn sàng

                    # === XỬ LÝ LƯU TOÀN BỘ KHUNG HÌNH ===
                    # Lưu toàn bộ khung hình BGR đã lật (img)
                    capture_success, capture_msg = save_full_frame_for_registration(img, name_to_register_upper)

                    # Cập nhật thông báo đăng ký
                    if capture_success:
                        st.session_state.upload_status = ('success', capture_msg)
                    else:
                        st.session_state.upload_status = ('error', capture_msg)

                    # Reset cờ
                    st.session_state.capture_flag = False

                    # Buộc Streamlit chạy lại lần nữa để cập nhật thông báo và UI cột 2 ngay lập tức
                    cap.release()
                    st.rerun()


                # --- CHẾ ĐỘ BÌNH THƯỜNG: ĐIỂM DANH ---
                elif encodeListKnow:  # Chỉ chạy điểm danh nếu có encodings đã biết
                    matches = face_recognition.compare_faces(encodeListKnow, encodesFace, tolerance=TOLERANCE)
                    faceDis = face_recognition.face_distance(encodeListKnow, encodesFace)

                    matchIndex = np.argmin(faceDis)

                    # === KIỂM TRA NGƯỠNG ĐỘ CHÍNH XÁC VÀ COOLDOWN CÁ NHÂN ===
                    if matches[matchIndex] and faceDis[matchIndex] < TOLERANCE:
                        name = classNames[matchIndex].upper()
                        color = (0, 255, 0)  # Xanh lá (luôn là xanh lá khi được nhận diện)

                        last_scan_time = st.session_state.last_successful_scan_time_per_person.get(name, 0)

                        # Kiểm tra thời gian chờ CỦA RIÊNG NGƯỜI NÀY
                        if (current_time_sec - last_scan_time) >= SCAN_COOLDOWN_SECONDS:

                            # Gọi hàm điểm danh (kiểm tra logic 1 phút trong CSV)
                            if not attendance_status_updated:  # Chỉ cập nhật 1 lần/frame
                                attendance_success = markAttendance(name)  # Chỉ nhận True/False

                                if attendance_success:
                                    # CẬP NHẬT: Cập nhật lịch sử ngay lập tức bằng placeholder
                                    display_attendance_history(history_placeholder)

                                    # Cập nhật thời gian cooldown CHỈ CHO NGƯỜI NÀY
                                    st.session_state.last_successful_scan_time_per_person[name] = current_time_sec
                                    status_message.success(f"✅ ĐIỂM DANH THÀNH CÔNG: {name}!")
                                    attendance_status_updated = True  # Đã cập nhật trạng thái
                                else:
                                    # Đang trong giới hạn 1 phút của Attendance.csv
                                    status_message.info(f"⏳ {name}: Đã điểm danh gần đây (trong vòng 1 phút).")
                                    # KHÔNG GHI ĐÈ MÀU. Vẫn giữ màu xanh lá (0, 255, 0) đã set ở trên.

                        else:
                            # Đang trong thời gian cooldown SCAN_COOLDOWN_SECONDS
                            remaining_cooldown = int(SCAN_COOLDOWN_SECONDS - (current_time_sec - last_scan_time))
                            status_message.warning(f"🟡 {name}: Vui lòng chờ {remaining_cooldown} giây để quét lại.")
                            # KHÔNG GHI ĐÈ MÀU. Vẫn giữ màu xanh lá (0, 255, 0) đã set ở trên.

                # --- VẼ KHUNG CHUNG ---

                # Nhân ngược tọa độ (top, right, bottom, left) lên 4
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                # Vẽ khung chính
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)

                # Vẽ nền cho tên (phía dưới)
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                # Đặt tên
                cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Hiển thị frame trong Streamlit placeholder
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

            # Tạm dừng ngắn để giảm tải CPU và cho phép Streamlit cập nhật
            time.sleep(0.01)

        # Phần code này sẽ không bao giờ chạy trừ khi có lỗi cap.isOpened()
        cap.release()
        frame_placeholder.empty()
        status_message.empty()
        st.warning("Camera đã bị tắt ngoài ý muốn.")
    elif not encodeListKnow:
        # Trường hợp không thể chạy vì chưa có encodings và lỗi camera
        st.warning("Ứng dụng đã khởi động, nhưng chưa tìm thấy dữ liệu khuôn mặt nào. Vui lòng thêm ảnh mới.")


if __name__ == '__main__':
    app_main()