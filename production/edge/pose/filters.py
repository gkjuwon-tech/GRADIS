"""
One-Euro Filter (Casiez et al. 2012) — 키포인트 적응형 평활화.

문제: pose 모델 출력은 매 프레임 미세하게 떨린다(지터). 이 떨림이 속도/각도 계산을
오염시켜 "푸쉬업"을 "낙상"으로, 가만히 있는 사람을 "난동"으로 오판하게 만든다.

순진한 해법(이동평균/저역통과)은 지터는 줄이지만 '지연'을 만든다. 진짜 낙상이
일어났을 때 0.3초 늦게 반응 → 사람이 더 다친다.

One-Euro는 둘 다 잡는다:
  - 천천히 움직일 때: 컷오프 낮춤 → 강하게 평활(지터 제거)
  - 빠르게 움직일 때: 컷오프 높임 → 즉각 반응(지연 없음)
속도에 따라 평활 강도를 적응시키는 게 핵심. 생명구조 신호처리에 정석.

키포인트(17×2)에 대해 벡터화. 신뢰도(conf)는 평활하지 않고 따로 전달.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def _alpha(cutoff, dt):
    tau = 1.0 / (TWO_PI * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroArray:
    """shape (K,2) 신호용 벡터화 One-Euro 필터."""

    def __init__(self, min_cutoff=1.2, beta=0.25, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None      # 평활된 직전 값
        self.dx_prev = None     # 평활된 속도
        self.init_mask = None   # 키포인트별 초기화 여부

    def reset(self):
        self.x_prev = self.dx_prev = self.init_mask = None

    def __call__(self, x, dt, valid):
        """
        x: (K,2) 관측. dt: 초. valid: (K,) bool — 이번에 신뢰 가능한 키포인트만 갱신.
        반환: (K,2) 평활값. 미초기화/무효 키포인트는 관측값 그대로.
        """
        x = np.asarray(x, np.float32)
        K = x.shape[0]
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.init_mask = valid.copy()
            out = x.copy()
            return out

        # 속도 추정 + 평활
        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1 - a_d) * self.dx_prev

        # 속도 크기에 따라 컷오프 적응
        speed = np.linalg.norm(edx, axis=1, keepdims=True)        # (K,1)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)                                    # (K,1)
        x_hat = a * x + (1 - a) * self.x_prev

        # 처음 보이는 키포인트는 평활 없이 채택
        newly = valid & (~self.init_mask)
        x_hat[newly] = x[newly]

        # 이번에 무효인 키포인트는 직전 평활값 유지(폐색 복원이 별도 처리)
        x_hat[~valid] = self.x_prev[~valid]

        self.x_prev = x_hat
        self.dx_prev = edx
        self.init_mask = self.init_mask | valid
        return x_hat
