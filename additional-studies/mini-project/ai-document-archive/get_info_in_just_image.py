import streamlit as st
import numpy as np

import cv2
import piexif

from geopy.geocoders import Nominatim
from datetime import datetime

# 모델 로드
@st.cache_resource
def load_yolo_model():
    # Yolo 모델 구성 파일과 가중치 파일 경로
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    names_path = "coco.names"  # 클래스 이름 파일

    # 모델 로드
    net = cv2.dnn.readNet(weights_path, config_path)

    # 클래스 이름들 로드
    with open(names_path, 'r') as f:
        classes = f.read().strip().split("\n")
    
    return net, classes

net, classes = load_yolo_model()

# EXIF 데이터 읽기
def read_exif_data(image):
    # EXIF 데이터 추출
    exif_info = piexif.load(image.info["exif"])

    return exif_info

# 주요 정보 추출하기
def extract_key_info(exif_info):
    metadata = {}

    # 촬영 일자
    date_time = exif_info.get("0th", {}).get(piexif.ImageIFD.DateTime, None)
    if date_time:
        metadata["date_time"] = date_time.decode("utf-8")

    # 카메라 제조사
    make = exif_info.get("0th", {}).get(piexif.ImageIFD.Make, None)
    if make:
        metadata["make"] = make.decode("utf-8")

    # 카메라 모델
    model = exif_info.get("0th", {}).get(piexif.ImageIFD.Model, None)
    if model:
        metadata["model"] = model.decode("utf-8")

    # GPS 좌표 (위도/경도)
    # 4. GPS 좌표 (위도/경도)
    gps_info = exif_info.get("GPS", {})
    latitude = gps_info.get(piexif.GPSIFD.GPSLatitude, None)
    longitude = gps_info.get(piexif.GPSIFD.GPSLongitude, None)

    if latitude and longitude:
        # 위도/경도 계산
        lat_degree = latitude[0][0] / latitude[0][1]
        lat_minute = latitude[1][0] / latitude[1][1]
        lat_second = latitude[2][0] / latitude[2][1]
        latitude_in_deg = lat_degree + (lat_minute / 60.0) + (lat_second / 3600.0)

        lon_degree = longitude[0][0] / longitude[0][1]
        lon_minute = longitude[1][0] / longitude[1][1]
        lon_second = longitude[2][0] / longitude[2][1]
        longitude_in_deg = lon_degree + (lon_minute / 60.0) + (lon_second / 3600.0)

        metadata["latitude"] = latitude_in_deg
        metadata["longitude"] = longitude_in_deg

    return metadata

# 사진에서 메타데이터 추출
def extract_photo_metadata(image):
    """사진에서 메타데이터 추출"""
    # 1. EXIF 데이터 읽기
    exif_info = read_exif_data(image)

    # 2. 주요 정보 추출 (날짜, 카메라, GPS)
    metadata = extract_key_info(exif_info)

    return metadata

# 객체 탐지 실행 함수
def run_object_detection_model(image):
    # 이미지 준비하기
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # 출력 레이어
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # 객체 탐지 결과
    detected_objects = []
    height, width, channels = image.shape
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:    # 신뢰도가 50%일 때만 탐지하기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 바운딩 박스 좌표
                x = center_x - w // 2
                y = center_y - h // 2

                detected_objects.append({
                    "class": classes[class_id],
                    "confidence": confidence,
                    "box": [x, y, w, h]
                })

    return detected_objects

# 날짜 키워드 추가하기
def create_date_keywords(taken_data):
    # 촬영 일자 파싱
    try:
        date_obj = datetime.strptime(taken_data, "%Y:%m:%d %H:%M:%S")

        # 연도, 월, 일, 시간대 추출
        year = str(date_obj.year)
        month = str(date_obj.month).zfill(2)  # 09처럼 두 자릿수로 표시
        day = str(date_obj.day).zfill(2)
        hour = date_obj.hour

        # 시간대 추출(오전/오후)
        time_of_day = "AM" if hour < 12 else "PM"

        # 키워드 리스트 반환
        date_keywords = [year, month, day, time_of_day]
        return date_keywords
    
    except Exception as e:
        print(f"Error parsing date: {e}")
        return []
    
# 위치 정보 주소 반환
def reverse_geocoding(location):
    """
    location: (latitude, longtitude) 튜플 형태의 좌표
    예: (37.7749, -122.4194)
    """

    geolocator = Nominatim(user_agent="geoapiExercises")

    # Reverse Geocoding (위도, 경도를 주소로 변환)
    location_info = geolocator.reverse(location, language="ko")

    if location_info:
        return location_info.address
    else:
        return None

# 사진 키워드 생성하기
def generate_photo_keywords(metadata, objects):
    """사진 메타데이터와 객체로 키워드 생성"""
    keywords = []

    # 날짜 키워드 추가
    if metadata.get("taken_date"):
        keywords.extend(create_date_keywords(metadata["taken_data"]))

    # 카메라 정보 추가
    if metadata.get("camera_info"):
        keywords.append(metadata["camera_info"])

    # 탐지된 객체 추가
    keywords.extend(objects)

    # 위치 정보를 주소로 변환
    if metadata.get("location"):
        address = reverse_geocoding(metadata["location"])
        if address:
            keywords.append(address)

    return keywords   

# 일반 사진인지 확인
def is_photo(doc_type, content):
    """
    문서 사진과 일반 사진을 구분하는 함수
    image: 이미지 파일
    doc_type: 문서 타입
    content: 문서 내용
    """
    if doc_type == "image" or len(content.strip()) == 0:
        return True
    else:
        return False

# 구조화된 데이터 생성하기
def format_photo_data(metadata, objects):
    """
    구조화된 데이터 생성: 메타데이터와 객체 탐지 결과를 합친다.
    metadata: 사진의 메타데이터(EXIF 데이터 등)
    objects: 객체 탐지 결과 (예: 사람, 자동차 등)
    """
    # 기본 메타 데이터
    photo_data = {
        "date_time": metadata.get("taken_data", ""),
        "make": metadata.get("make", ""),
        "model": metadata.get("model", ""),
        "latitude": metadata.get("latitude", None),
        "longtitude": metadata.get("longitude", None),
        "objects_detected": objects  # 탐지된 객체들
    }

    return photo_data

# 사진 요약 생성
def create_photo_summary(objects, metadata):
    """
    사진 요약 생성: 객체 탐지 결과와 메타데이터를 사용해서 요약을 생성
    objects: 탐지된 객체들 (예: 사람, 자동차 등)
    metadata: 사진의 메타데이터
    """
    # 날짜, 카메라 정보 추출
    date_time = metadata.get("taken_data", "Unknown date")
    camera_make = metadata.get("make", "Unknown camera")
    camera_model = metadata.get("model", "Unknown model")
    latitude = metadata.get("latitude", "Unknown latitude")
    longitude = metadata.get("longitude", "Unknown longitude")

    # 객체 정보 생성
    object_info = []
    for obj in objects:
        object_info.append(f"{obj['class']} (confidence: {obj['confidence']:.2f})")
    
    # 요약 문장 생성
    summary = (
        f"This photo was taken on {date_time} by a {camera_make} {camera_model}. "
        f"The photo includes: {', '.join(object_info)}. "
        f"The photo was taken at latitude {latitude} and longitude {longitude}."
    )
    
    return summary