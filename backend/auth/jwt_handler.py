import time
from authlib.jose import jwt, JoseError
from backend.config import settings

ALGORITHM = "HS256"


def create_access_token(user_id: int) -> str:
    expire = int(time.time()) + settings.access_token_expire_minutes * 60
    payload = {"sub": str(user_id), "exp": expire}
    key = settings.secret_key.get_secret_value()
    token = jwt.encode({"alg": ALGORITHM}, payload, key)
    # authlib returns bytes; decode to str
    return token.decode() if isinstance(token, bytes) else token


def decode_token(token: str) -> int | None:
    try:
        key = settings.secret_key.get_secret_value()
        claims = jwt.decode(token, key)
        claims.validate()
        return int(claims["sub"])
    except (JoseError, KeyError, ValueError):
        return None
