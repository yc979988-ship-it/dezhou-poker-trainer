"""动态对手的隐藏连续参数与可复现会话漂移。

本模块只描述对手的内部行为倾向，不包含会暴露给玩家的固定风格标签。
基础参数由 ``master_seed`` 和 ``opponent_id`` 稳定派生；每个训练场次再在
logit 空间加入小幅扰动。整个过程不读取或修改 Python 的全局随机状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math


Seed = int | str | bytes

# 基础画像按“整体略松的普通朋友局”校准，同时保留明显的个体差异。
# 主要增加主动入池与 limp；PFR 占比、3bet 和激进度不额外上调，避免
# 把“松一点”误调成全员松凶。
_VPIP_RANGE = (0.21, 0.53)
_PFR_SHARE_RANGE = (0.45, 0.88)
_THREE_BET_RANGE = (0.025, 0.14)
_AGGRESSION_FACTOR_RANGE = (0.80, 4.00)
_FOLD_TENDENCY_RANGE = (0.26, 0.70)
_LIMP_TENDENCY_RANGE = (0.05, 0.31)
_MISTAKE_RATE_RANGE = (0.01, 0.10)

# 对有界变量归一化后，在 logit 空间内允许的最大单场移动。
_SESSION_LOGIT_SHIFT = 0.24
_UNIT_DENOMINATOR = float(1 << 64)
_LOGIT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OpponentProfile:
    """一名对手仅供模拟器使用的隐藏参数。

    除 ``aggression_factor`` 外，其余数值字段都是 ``0..1`` 的比例。
    ``pfr`` 必须严格小于 ``vpip``，避免生成“翻前加注比主动入池还多”的
    不可能画像。数据类不可变，防止同一场次中画像被意外改写。
    """

    opponent_id: str
    vpip: float
    pfr: float
    three_bet: float
    aggression_factor: float
    fold_tendency: float
    limp_tendency: float
    mistake_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_id, str) or not self.opponent_id.strip():
            raise ValueError("opponent_id 不能为空")

        probability_fields = (
            "vpip",
            "pfr",
            "three_bet",
            "fold_tendency",
            "limp_tendency",
            "mistake_rate",
        )
        for field_name in probability_fields:
            value = _finite_float(getattr(self, field_name), field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} 必须在 0 到 1 之间")
            object.__setattr__(self, field_name, value)

        aggression_factor = _finite_float(
            self.aggression_factor, "aggression_factor"
        )
        if aggression_factor <= 0.0:
            raise ValueError("aggression_factor 必须大于 0")
        object.__setattr__(self, "aggression_factor", aggression_factor)

        if self.pfr >= self.vpip:
            raise ValueError("pfr 必须严格小于 vpip")


def generate_base_profile(opponent_id: str, master_seed: Seed) -> OpponentProfile:
    """为一名对手生成稳定、无离散标签的基础画像。

    相同的 ``opponent_id`` 与 ``master_seed`` 总是得到逐字段相同的结果；
    不同字段使用独立的 BLAKE2b 域，避免从一个共享随机流取值时产生顺序
    依赖。整数、字符串和字节种子也使用不同的类型前缀。
    """

    _validate_opponent_id(opponent_id)
    _seed_bytes(master_seed)  # 在开始派生前给出清晰的类型错误。

    vpip = _sample_range(
        _VPIP_RANGE,
        _stable_unit("base:vpip", opponent_id, master_seed),
    )
    pfr_share = _sample_range(
        _PFR_SHARE_RANGE,
        _stable_unit("base:pfr_share", opponent_id, master_seed),
    )

    return OpponentProfile(
        opponent_id=opponent_id,
        vpip=vpip,
        pfr=vpip * pfr_share,
        three_bet=_sample_range(
            _THREE_BET_RANGE,
            _stable_unit("base:three_bet", opponent_id, master_seed),
        ),
        aggression_factor=_sample_range(
            _AGGRESSION_FACTOR_RANGE,
            _stable_unit("base:aggression_factor", opponent_id, master_seed),
        ),
        fold_tendency=_sample_range(
            _FOLD_TENDENCY_RANGE,
            _stable_unit("base:fold_tendency", opponent_id, master_seed),
        ),
        limp_tendency=_sample_range(
            _LIMP_TENDENCY_RANGE,
            _stable_unit("base:limp_tendency", opponent_id, master_seed),
        ),
        mistake_rate=_sample_range(
            _MISTAKE_RATE_RANGE,
            _stable_unit("base:mistake_rate", opponent_id, master_seed),
        ),
    )


def drift_for_session(
    base: OpponentProfile,
    session_seed: Seed,
) -> OpponentProfile:
    """返回某训练场次使用的轻微漂移画像。

    扰动发生在每个有界参数归一化后的 logit 空间，因此靠近边界的参数
    不会被简单相加推到合法范围之外。PFR 以 VPIP 中的占比漂移，天然保持
    ``pfr < vpip``。返回新的不可变对象，原始基础画像不会改变。
    """

    if not isinstance(base, OpponentProfile):
        raise TypeError("base 必须是 OpponentProfile")
    _seed_bytes(session_seed)

    def drift_value(
        field_name: str,
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        noise = (
            2.0
            * _stable_unit(
                f"session:{field_name}", base.opponent_id, session_seed
            )
            - 1.0
        )
        return _bounded_logit_jitter(
            value,
            bounds,
            noise * _SESSION_LOGIT_SHIFT,
        )

    vpip = drift_value("vpip", base.vpip, _VPIP_RANGE)
    pfr_share = drift_value(
        "pfr_share",
        base.pfr / base.vpip,
        _PFR_SHARE_RANGE,
    )

    return OpponentProfile(
        opponent_id=base.opponent_id,
        vpip=vpip,
        pfr=vpip * pfr_share,
        three_bet=drift_value(
            "three_bet", base.three_bet, _THREE_BET_RANGE
        ),
        aggression_factor=drift_value(
            "aggression_factor",
            base.aggression_factor,
            _AGGRESSION_FACTOR_RANGE,
        ),
        fold_tendency=drift_value(
            "fold_tendency", base.fold_tendency, _FOLD_TENDENCY_RANGE
        ),
        limp_tendency=drift_value(
            "limp_tendency", base.limp_tendency, _LIMP_TENDENCY_RANGE
        ),
        mistake_rate=drift_value(
            "mistake_rate", base.mistake_rate, _MISTAKE_RATE_RANGE
        ),
    )


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数值")
    return result


def _validate_opponent_id(opponent_id: object) -> None:
    if not isinstance(opponent_id, str) or not opponent_id.strip():
        raise ValueError("opponent_id 不能为空")


def _seed_bytes(seed: Seed) -> bytes:
    """用显式类型前缀编码种子，避免 ``1`` 与 ``"1"`` 发生碰撞。"""

    if isinstance(seed, bytes):
        return b"bytes:" + seed
    if isinstance(seed, str):
        return b"str:" + seed.encode("utf-8")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return b"int:" + str(seed).encode("ascii")
    raise TypeError("seed 必须是 int、str 或 bytes")


def _stable_unit(domain: str, opponent_id: str, seed: Seed) -> float:
    """稳定派生一个 ``[0, 1)`` 浮点数，不依赖 Python ``hash``。"""

    digest = blake2b(digest_size=8, person=b"pkr-opponent-v1")
    for part in (
        domain.encode("utf-8"),
        opponent_id.encode("utf-8"),
        _seed_bytes(seed),
    ):
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return int.from_bytes(digest.digest(), "big") / _UNIT_DENOMINATOR


def _sample_range(bounds: tuple[float, float], unit: float) -> float:
    lower, upper = bounds
    return lower + (upper - lower) * unit


def _bounded_logit_jitter(
    value: float,
    bounds: tuple[float, float],
    logit_shift: float,
) -> float:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError(
            f"基础参数 {value!r} 超出会话漂移范围 [{lower}, {upper}]"
        )

    normalized = (value - lower) / (upper - lower)
    normalized = min(max(normalized, _LOGIT_EPSILON), 1.0 - _LOGIT_EPSILON)
    logit = math.log(normalized / (1.0 - normalized)) + logit_shift
    shifted = 1.0 / (1.0 + math.exp(-logit))
    result = lower + (upper - lower) * shifted
    # 防御浮点舍入；正常计算本身已经严格处于边界内。
    return min(max(result, lower), upper)


__all__ = [
    "OpponentProfile",
    "drift_for_session",
    "generate_base_profile",
]

