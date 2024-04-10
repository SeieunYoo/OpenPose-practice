import cv2
import time # time 라이브러리 import

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    time.sleep(1) # 측정하고자 하는 코드 부분
    global points

    print(proto_file,'THis is path')
    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"\n============================== {model_name} Model ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    print(f"{time.time()-start:.4f} sec") # 종료와 함께 수행시간 출력
    cv2.imshow("Output_Keypoints", frame)
    cv2.waitKey(0)
    return frame

BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  }
                  
# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_mpi = "./pose_deploy_linevec.prototxt"
protoFile_mpi_faster = "C:\\Users\\USER\\Downloads\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
# protoFile_coco = "C:\\openpose\\models\\pose\\coco\\pose_deploy_linevec.prototxt"
# protoFile_body_25 = "C:\\Users\\USER\\Downloads\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_mpi = "C:\\Users\\USER\\Downloads\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel"
# weightsFile_coco = "C:\\openpose\\models\\pose\\coco\\pose_iter_440000.caffemodel"
# weightsFile_body_25 = "C:\\Users\\USER\\Downloads\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"

# 이미지 경로
man = "C:\\Users\\USER\\Downloads\\man.jpg"
girl = "C:\\Users\\USER\\Downloads\\girl.jpg"

# 키포인트를 저장할 빈 리스트
points = []

# 이미지 읽어오기
frame_mpii = cv2.imread(girl)
# frame_coco = frame_mpii.copy()
frame_body_25 = frame_mpii.copy()



# MPII Model
start = time.time() # 시작
frame_MPII = output_keypoints(frame=frame_mpii, proto_file=protoFile_mpi_faster, weights_file=weightsFile_mpi,
                             threshold=0.2, model_name="MPII", BODY_PARTS=BODY_PARTS_MPI)


#output_keypoints_with_lines(frame=frame_MPII, POSE_PAIRS=POSE_PAIRS_MPI)

# # COCO Model
# frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
#                              threshold=0.2, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)
# output_keypoints_with_lines(frame=frame_COCO, POSE_PAIRS=POSE_PAIRS_COCO)

# # BODY_25 Model
# frame_BODY_25 = output_keypoints(frame=frame_body_25, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
#                              threshold=0.2, model_name="BODY_25", BODY_PARTS=BODY_PARTS_BODY_25)
# # output_keypoints_with_lines(frame=frame_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)