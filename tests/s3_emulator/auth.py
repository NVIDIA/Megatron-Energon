import hmac
import re
import urllib.parse as _up
from hashlib import sha256
from typing import Dict, Mapping, MutableMapping

__all__ = ["S3Auth", "InvalidSignature"]

_SIGNED_HEADERS_RE = re.compile(r"SignedHeaders=([^,]+)")
_CREDENTIAL_RE = re.compile(r"Credential=([^,]+)")
_SIGNATURE_RE = re.compile(r"Signature=([0-9a-fA-F]+)")


class InvalidSignature(Exception):
    """Raised when the supplied signature does not match."""


class S3Auth:
    """Very small subset implementation of AWS Signature V4 verification.

    Only what is mandatory for the emulator to work for *most* typical SDK
    operations is implemented. Notably, chunked uploads and presigned URLs are
    not supported.
    """

    def __init__(self, credentials: Mapping[str, str], region: str = "us-east-1") -> None:
        """Parameters
        ----------
        credentials:
            Mapping of *access_key* -> *secret_key* accepted by the server.
        region:
            AWS region assumed when verifying the signing key (default
            ``us-east-1``).
        """
        self._creds: Dict[str, str] = dict(credentials)
        self._region = region

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def verify(
        self,
        method: str,
        canonical_uri: str,
        canonical_querystring: str,
        headers: Mapping[str, str] | MutableMapping[str, str],
        payload: bytes,
    ) -> None:
        """Validate the *Authorization* header for the given request.

        Raises
        ------
        InvalidSignature
            If the calculated signature does not match the one supplied by the
            client.
        ValueError
            If mandatory signing headers are missing.
        """
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if auth_header is None:
            raise InvalidSignature("Missing Authorization header")

        signed_headers = _first_group(_SIGNED_HEADERS_RE, auth_header)
        credential_str = _first_group(_CREDENTIAL_RE, auth_header)
        signature = _first_group(_SIGNATURE_RE, auth_header)

        if not (signed_headers and credential_str and signature):
            raise InvalidSignature("Malformed Authorization header")

        access_key, date_str, region, service, terminator = credential_str.split("/")
        if service != "s3" or terminator != "aws4_request":
            raise InvalidSignature("Invalid credential scope")
        if region != self._region:
            print(f"Signature region {region} does not match server region {self._region}")
        secret_key = self._creds.get(access_key)
        if secret_key is None:
            raise InvalidSignature("Unknown access key")

        # ------------------------------------------------------------------
        # Canonical URI & query string (encode & normalise)
        # ------------------------------------------------------------------
        canonical_uri = _canonical_uri(canonical_uri)
        canonical_querystring = _canonical_querystring(canonical_querystring)

        # Construct canonical request ------------------------------------------------
        # 1. Canonical headers
        canonical_headers = ""
        for hdr in signed_headers.split(";"):
            hdr_lower = hdr.lower()
            value = headers.get(hdr) or headers.get(hdr_lower)
            if value is None:
                raise InvalidSignature(f"Signed header '{hdr}' missing from request")
            canonical_headers += f"{hdr_lower}:{_normalize_whitespace(str(value))}\n"
        # 2. Hashed payload
        payload_hash = sha256(payload).hexdigest()
        # 3. Canonical request string
        canonical_request = "\n".join(
            [
                method,
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        hashed_canonical_request = sha256(canonical_request.encode()).hexdigest()

        # ------------------------------------------------------------------
        # String to sign
        amz_date = headers.get("x-amz-date") or headers.get("X-Amz-Date")
        if amz_date is None:
            raise ValueError("Missing x-amz-date header")
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                "/".join([date_str, region, "s3", "aws4_request"]),
                hashed_canonical_request,
            ]
        )

        # ------------------------------------------------------------------
        # Calculate signing key and signature
        date_key = _sign(("AWS4" + secret_key).encode(), date_str)
        region_key = _sign(date_key, region)
        service_key = _sign(region_key, "s3")
        signing_key = _sign(service_key, "aws4_request")
        calc_signature = hmac.new(signing_key, string_to_sign.encode(), sha256).hexdigest()

        # ------------------------------------------------------------------
        if not hmac.compare_digest(calc_signature, signature):
            print(f"Sig mismatch: expected={signature} got={calc_signature}")
            raise InvalidSignature("Signature mismatch")


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _first_group(regex: re.Pattern[str], string: str) -> str | None:
    match = regex.search(string)
    return match.group(1) if match else None


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), sha256).digest()


def _normalize_whitespace(value: str) -> str:
    """Collapse consecutive whitespace (as per AWS SigV4 spec)."""
    return " ".join(value.strip().split())


# ----------------------------------------------------------------------
# Additional canonical helpers
# ----------------------------------------------------------------------


def _percent_encode(value: str) -> str:  # noqa: D401
    return _up.quote(value, safe="-_.~")


def _canonical_uri(uri: str) -> str:  # noqa: D401
    """Return URI‐encoded path as required by SigV4.

    Each segment between ``/`` must be percent-encoded with the AWS safe list
    ``-_.~``. Duplicate slashes are preserved (AWS behaviour).
    """
    if uri == "":
        return "/"
    encoded_parts = [_percent_encode(_up.unquote(part)) for part in uri.split("/")]
    # Leading and trailing slash must be preserved
    prefix = "" if uri.startswith("/") else "/"
    return prefix + "/".join(encoded_parts)


def _canonical_querystring(raw_qs: str) -> str:  # noqa: D401
    if raw_qs == "":
        return ""
    pairs = _up.parse_qsl(raw_qs, keep_blank_values=True)
    encoded_pairs = [(_percent_encode(k), _percent_encode(v)) for k, v in pairs]
    # Sort by key then value to comply with AWS spec
    encoded_pairs.sort()
    return "&".join(f"{k}={v}" for k, v in encoded_pairs)
