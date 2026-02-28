import time

import pytest

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    hashed = hash_password("mypassword")
    assert hashed != "mypassword"
    assert verify_password("mypassword", hashed)
    assert not verify_password("wrongpassword", hashed)


def test_different_passwords_different_hashes():
    h1 = hash_password("pass1")
    h2 = hash_password("pass2")
    assert h1 != h2


def test_create_and_decode_access_token():
    token = create_access_token("testuser")
    payload = decode_token(token)
    assert payload["sub"] == "testuser"
    assert payload["type"] == "access"
    assert "exp" in payload


def test_create_and_decode_refresh_token():
    token = create_refresh_token("testuser")
    payload = decode_token(token)
    assert payload["sub"] == "testuser"
    assert payload["type"] == "refresh"


def test_invalid_token_raises():
    with pytest.raises(Exception):
        decode_token("not.a.valid.token")


def test_access_token_has_shorter_expiry_than_refresh():
    access = create_access_token("user")
    refresh = create_refresh_token("user")

    access_exp = decode_token(access)["exp"]
    refresh_exp = decode_token(refresh)["exp"]

    assert refresh_exp > access_exp
