import numpy as np

def mod2pi(theta):
    return theta - 2.0 * np.pi * np.floor(theta / (2.0 * np.pi))

class DubinsCost:
    def __init__(self, turning_radius=1.0):
        self._R = turning_radius

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value

    def cost(self, a, b):
        """
        計算兩個位態之間的最短 Dubins 路徑長度。
        """
        _, _, _, _, length = self._plan_dubins(a, b)
        return length

    def get_path_points(self, a, b, step=0.5):
        """
        產生真實遵循轉彎半徑的插值座標點。
        """
        t, p, q, mode, _ = self._plan_dubins(a, b)
        if mode is None:
            dist = np.linalg.norm(a[:2] - b[:2])
            num = max(int(dist/step), 2)
            return np.linspace(a[0], b[0], num), np.linspace(a[1], b[1], num)
        
        return self._interpolate(a, t, p, q, mode, step)

    def _plan_dubins(self, a, b):
        p1 = (a[0], a[1], a[2] if len(a) > 2 else 0.0)
        p2 = (b[0], b[1], b[2] if len(b) > 2 else 0.0)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = np.sqrt(dx**2 + dy**2)
        th = np.arctan2(dy, dx)
        alpha, beta = mod2pi(p1[2] - th), mod2pi(p2[2] - th)
        d = dist / self._R
        
        best_len, best_params = float('inf'), (None, None, None, None)
        for mode in ['LSL', 'RSR', 'LSR', 'RSL']:
            res = self._calculate_csc(alpha, beta, d, mode)
            if res:
                t, p, q = res
                if (t + p + q) < best_len:
                    best_len = t + p + q
                    best_params = (t, p, q, mode)
        return best_params[0], best_params[1], best_params[2], best_params[3], best_len * self._R

    def _calculate_csc(self, alpha, beta, d, mode):
        sa, ca, sb, cb, cab = np.sin(alpha), np.cos(alpha), np.sin(beta), np.cos(beta), np.cos(alpha - beta)
        if mode == 'LSL':
            tmp = d + sa - sb
            p_sq = 2 + d*d + 2*(d*(sa - sb) - cab)
            if p_sq < 0: return None
            p = np.sqrt(p_sq)
            t, q = mod2pi(np.arctan2(cb - ca, tmp) - alpha), mod2pi(beta - np.arctan2(cb - ca, tmp))
        elif mode == 'RSR':
            tmp = d - sa + sb
            p_sq = 2 + d*d - 2*(d*(sa - sb) + cab)
            if p_sq < 0: return None
            p = np.sqrt(p_sq)
            t, q = mod2pi(alpha - np.arctan2(ca - cb, tmp)), mod2pi(np.arctan2(ca - cb, tmp) - beta)
        elif mode == 'LSR':
            p_sq = -2 + d*d + 2*(cab + d*(sa + sb))
            if p_sq < 0: return None
            p = np.sqrt(p_sq)
            t = mod2pi(np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p) - alpha)
            q = mod2pi(np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p) - beta)
        elif mode == 'RSL':
            p_sq = -2 + d*d + 2*(cab - d*(sa + sb))
            if p_sq < 0: return None
            p = np.sqrt(p_sq)
            t = mod2pi(alpha - (np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)))
            q = mod2pi(beta - (np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)))
        else: return None
        return t, p, q

    def _interpolate(self, start, t, p, q, mode, step_size):
        x0, y0, th0 = start[0], start[1], start[2] if len(start) > 2 else 0.0
        total_len = (t + p + q) * self._R
        num_pts = max(int(total_len / step_size), 10)
        px, py = [], []
        for i in range(num_pts):
            s = (i / (num_pts - 1)) * (t + p + q)
            if s <= t:
                x, y, _ = self._move(x0, y0, th0, s, mode[0])
            elif s <= t + p:
                x1, y1, th1 = self._move(x0, y0, th0, t, mode[0])
                x, y, _ = self._move(x1, y1, th1, s - t, mode[1])
            else:
                x1, y1, th1 = self._move(x0, y0, th0, t, mode[0])
                x2, y2, th2 = self._move(x1, y1, th1, p, mode[1])
                x, y, _ = self._move(x2, y2, th2, s - t - p, mode[2])
            px.append(x); py.append(y)
        return np.array(px), np.array(py)

    def _move(self, x, y, th, s, m):
        if m == 'L':
            return x + self._R*(np.sin(th+s)-np.sin(th)), y + self._R*(np.cos(th)-np.cos(th+s)), th+s
        elif m == 'R':
            return x + self._R*(np.sin(th)-np.sin(th-s)), y + self._R*(np.cos(th-s)-np.cos(th)), th-s
        else:
            return x + self._R*s*np.cos(th), y + self._R*s*np.sin(th), th