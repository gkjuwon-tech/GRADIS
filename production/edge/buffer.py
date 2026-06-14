"""
RingBuffer — 우리의 양심.

원본 영상은 여기, RAM 위에서만, 딱 N초 산다. 디스크에 안 적는다.
위험이 감지되면 그 순간 전후를 rescue()로 '구출'해서 경찰한테 보낸다.
아무 일 없으면? 새 프레임이 옛 프레임을 덮어쓴다 = 자연 소멸 = 프라이버시.

이게 GRADIS의 법적 킬포인트다. 변호사들이 이 파일을 사랑한다.
"""
from collections import deque
import threading


class RingBuffer:
    def __init__(self, seconds=30.0, fps=12):
        self.seconds = seconds
        self.maxlen = int(seconds * fps) + 4
        self._buf = deque(maxlen=self.maxlen)   # (ts, frame)  frame = PIL.Image
        self._lock = threading.Lock()

    def push(self, ts, frame):
        with self._lock:
            self._buf.append((ts, frame))

    def rescue(self, now, pre=3.0, post=0.0):
        """[now-pre, now+post] 구간 프레임을 복사해서 돌려준다 (원본 클립)."""
        lo, hi = now - pre, now + post
        with self._lock:
            return [f for (t, f) in self._buf if lo <= t <= hi]

    def wipe(self):
        """전부 소멸. 복구 불가. 드론 종료/임무전환 시 호출."""
        with self._lock:
            self._buf.clear()

    def __len__(self):
        with self._lock:
            return len(self._buf)
