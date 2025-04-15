import numpy as np
import cv2 as cv

# 캘리브레이션 결과 (예시값, 실제로는 calibration.py에서 출력된 값 사용)
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])

video_file = 'data/chessboard.avi'
board_pattern = (8, 6)
board_cellsize = 0.025
criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

video = cv.VideoCapture(video_file)
assert video.isOpened(), f'Cannot open video {video_file}'

# 피라미드의 3D 좌표 설정
pyramid_base = board_cellsize * np.array([
    [3, 3, 0], [5, 3, 0], [5, 5, 0], [3, 5, 0]
])
pyramid_top = board_cellsize * np.array([[4, 4, -1]])

# 체스보드 3D 좌표
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv.findChessboardCorners(img, board_pattern, criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 피라미드 프로젝션
        proj_base, _ = cv.projectPoints(pyramid_base, rvec, tvec, K, dist_coeff)
        proj_top, _ = cv.projectPoints(pyramid_top, rvec, tvec, K, dist_coeff)

        # 밑면 그리기
        cv.polylines(img, [np.int32(proj_base)], True, (255, 255, 0), 2)

        # 측면 선 그리기
        for pt in proj_base:
            cv.line(img, tuple(np.int32(pt.flatten())), tuple(np.int32(proj_top[0].flatten())), (0, 255, 255), 2)

        # 카메라 위치 정보 표시
        R, _ = cv.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()
        cv.putText(img, f'XYZ: [{cam_pos[0]:.2f} {cam_pos[1]:.2f} {cam_pos[2]:.2f}]',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow('Pose Estimation - Pyramid', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    elif key == 27:
        break

video.release()
cv.destroyAllWindows()
