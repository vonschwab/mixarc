"""The artists list must survive every boundary: API body -> request model ->
worker args -> back. A field that is dropped at one hop is the classic
'silently inert feature' bug."""
from __future__ import annotations

from src.playlist.request_models import GeneratePlaylistRequest


def test_artists_round_trips_through_worker_args():
    req = GeneratePlaylistRequest(
        mode="artist", artist="Brian Eno", artists=["Brian Eno", "Harold Budd"],
    )
    args = req.to_worker_args()
    assert args["artists"] == ["Brian Eno", "Harold Budd"]
    back = GeneratePlaylistRequest.from_worker_args(args)
    assert back.artists == ["Brian Eno", "Harold Budd"]


def test_empty_artists_is_omitted_from_worker_args():
    req = GeneratePlaylistRequest(mode="artist", artist="Brian Eno")
    assert "artists" not in req.to_worker_args()


def test_artists_entries_are_cleaned():
    req = GeneratePlaylistRequest.from_worker_args(
        {"mode": "artist", "artists": ["  Brian Eno ", "", None, "Harold Budd"]}
    )
    assert req.artists == ["Brian Eno", "Harold Budd"]


def test_validation_accepts_artists_without_a_scalar_artist():
    req = GeneratePlaylistRequest(mode="artist", artists=["Brian Eno", "Harold Budd"])
    assert req.validation_error() is None


def test_api_body_carries_artists_into_the_request_model():
    from src.playlist_web.schemas import GenerateRequestBody

    body = GenerateRequestBody(
        mode="artist", artist="Brian Eno", artists=["Brian Eno", "Harold Budd"],
    )
    # to_request requires the policy-resolved dial axes (real signature is
    # to_request(self, axes: dict) -> GeneratePlaylistRequest; the brief's
    # zero-arg call is stale against src/playlist_web/app.py:318 and the
    # existing tests/unit/test_web_schemas.py callers).
    req = body.to_request({})
    assert req.artists == ["Brian Eno", "Harold Budd"]
