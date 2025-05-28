import urllib.parse as _up
from datetime import datetime, timezone
from email.utils import formatdate
from hashlib import md5
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Protocol

from .auth import InvalidSignature, S3Auth
from .state import S3State

__all__ = ["S3RequestHandler"]


class S3RequestHandler(BaseHTTPRequestHandler):
    server: "S3ServerProtocol"  # type: ignore[assignment]

    # Disable default logging to stderr of every single request from BaseHTTPRequestHandler
    def log_message(self, fmt: str, *args):  # noqa: D401, ANN001
        print(f"{self.client_address[0]} - - {fmt % args}")

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------

    def do_PUT(self):  # noqa: N802  (method names determined by BaseHTTPRequestHandler)
        self._handle_write()

    def do_GET(self):  # noqa: N802
        self._handle_read(listing=False)

    def do_HEAD(self):  # noqa: N802
        self._handle_read(listing=False, only_headers=True)

    def do_DELETE(self):  # noqa: N802
        self._handle_delete()

    def do_POST(self):  # noqa: N802
        self._handle_post()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return b""
        data = self.rfile.read(length)
        return data

    def _split_path(self):  # noqa: D401
        """Return ``bucket``, ``key`` (key may be "")."""
        parsed = _up.urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        bucket = parts[0] if parts else ""
        key = "/".join(parts[1:]) if len(parts) > 1 else ""
        return bucket, key, parsed

    def _auth(self, payload: bytes, parsed: _up.ParseResult):
        try:
            self.server.auth.verify(
                method=self.command,
                canonical_uri=parsed.path or "/",
                canonical_querystring=parsed.query,
                headers=self.headers,  # type: ignore[arg-type]
                payload=payload,
            )
        except InvalidSignature as err:
            self._send_error(HTTPStatus.FORBIDDEN, str(err))
            return False
        except ValueError as err:
            self._send_error(HTTPStatus.BAD_REQUEST, str(err))
            return False
        return True

    # ------------------------------------------------------------------

    def _handle_write(self):
        bucket, key, parsed = self._split_path()
        body = self._read_body()
        if not self._auth(body, parsed):
            return

        qs = _up.parse_qs(parsed.query, keep_blank_values=True)

        # ------------------------------------------------------------------
        # Multipart: upload part
        # ------------------------------------------------------------------
        if "uploadId" in qs and "partNumber" in qs:
            upload_id = qs["uploadId"][0]
            try:
                part_no = int(qs["partNumber"][0])
            except ValueError:
                self._send_error(HTTPStatus.BAD_REQUEST, "Invalid partNumber")
                return
            try:
                self.server.state.upload_part(upload_id, part_no, body)
            except KeyError:
                self._send_error(HTTPStatus.NOT_FOUND, "Upload not found")
                return
            self._send_status(HTTPStatus.OK, extra_headers={"ETag": _etag(body)})
            return

        if not bucket:
            self._send_error(HTTPStatus.BAD_REQUEST, "Bucket must be specified")
            return

        if key == "":  # Bucket create
            self.server.state.create_bucket(bucket)
            self._send_status(HTTPStatus.OK)
            return

        # Put object
        self.server.state.put_object(bucket, key, body)
        self._send_status(
            HTTPStatus.OK,
            extra_headers={"ETag": _etag(body)},
        )

    # ------------------------------------------------------------------

    def _handle_read(self, listing: bool, only_headers: bool = False):
        bucket, key, parsed = self._split_path()
        body = b""  # GET/HEAD normally payload considered in signature (hash of empty string)
        if not self._auth(body, parsed):
            return

        if not bucket:
            self._send_error(HTTPStatus.BAD_REQUEST, "Bucket must be specified")
            return

        if key == "":  # List bucket contents
            if not listing:
                # We treat listing with GET only
                try:
                    objects = self.server.state.list_objects(bucket)
                except KeyError:
                    self._send_error(HTTPStatus.NOT_FOUND, "Bucket not found")
                    return
                xml_body = self._render_bucket_list(bucket, objects)
                self._send_bytes(xml_body, content_type="application/xml")
            else:
                self._send_error(HTTPStatus.NOT_IMPLEMENTED, "Listing not implemented")
            return

        try:
            data = self.server.state.get_object(bucket, key)
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            rng = range_header.split("=", 1)[1]
            if "-" not in rng:
                self._send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range")
                return
            start_str, end_str = rng.split("-", 1)
            try:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else len(data) - 1
            except ValueError:
                self._send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range")
                return
            if start > end or start >= len(data):
                self._send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range")
                return
            end = min(end, len(data) - 1)
            slice_data = data[start : end + 1]
            headers = {
                "Content-Range": f"bytes {start}-{end}/{len(data)}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(slice_data)),
                "ETag": _etag(data),
            }
            if only_headers:
                headers.setdefault("Content-Type", "application/octet-stream")
                headers.setdefault("Last-Modified", formatdate(usegmt=True))
                self._send_status(HTTPStatus.PARTIAL_CONTENT, extra_headers=headers)
            else:
                self._send_bytes(
                    slice_data,
                    status=HTTPStatus.PARTIAL_CONTENT,
                    content_type="application/octet-stream",
                    extra_headers=headers,
                )
        else:
            if only_headers:
                self._send_status(
                    HTTPStatus.OK,
                    extra_headers={
                        "Content-Length": str(len(data)),
                        "Accept-Ranges": "bytes",
                        "Content-Type": "application/octet-stream",
                        "Last-Modified": formatdate(usegmt=True),
                        "ETag": _etag(data),
                    },
                )
            else:
                self._send_bytes(
                    data,
                    content_type="application/octet-stream",
                    extra_headers={"Accept-Ranges": "bytes"},
                )

    # ------------------------------------------------------------------

    def _handle_delete(self):
        bucket, key, parsed = self._split_path()
        body = b""  # empty
        if not self._auth(body, parsed):
            return

        if not bucket:
            self._send_error(HTTPStatus.BAD_REQUEST, "Bucket must be specified")
            return

        if key == "":
            try:
                self.server.state.delete_bucket(bucket)
            except (KeyError, RuntimeError) as err:
                self._send_error(HTTPStatus.BAD_REQUEST, str(err))
                return
            self._send_status(HTTPStatus.NO_CONTENT)
            return

        try:
            self.server.state.delete_object(bucket, key)
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_status(HTTPStatus.NO_CONTENT)

    # ------------------------------------------------------------------
    # NEW: POST handler
    # ------------------------------------------------------------------

    def _handle_post(self):
        bucket, key, parsed = self._split_path()
        body = self._read_body()
        if not self._auth(body, parsed):
            return

        qs = _up.parse_qs(parsed.query, keep_blank_values=True)

        # Initiate multipart: POST ?uploads
        if "uploads" in qs or parsed.query == "uploads":
            upload_id = self.server.state.initiate_multipart(bucket, key)
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<InitiateMultipartUploadResult>"
                f"<Bucket>{_escape_xml(bucket)}</Bucket>"
                f"<Key>{_escape_xml(key)}</Key>"
                f"<UploadId>{upload_id}</UploadId>"
                "</InitiateMultipartUploadResult>"
            ).encode()
            self._send_bytes(xml, status=HTTPStatus.OK, content_type="application/xml")
            return

        # Complete multipart: POST ?uploadId=xxxx
        if "uploadId" in qs:
            upload_id = qs["uploadId"][0]
            try:
                self.server.state.complete_multipart(upload_id)
            except KeyError:
                self._send_error(HTTPStatus.NOT_FOUND, "Upload not found")
                return
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<CompleteMultipartUploadResult>"
                f"<Bucket>{_escape_xml(bucket)}</Bucket>"
                f"<Key>{_escape_xml(key)}</Key>"
                f"<UploadId>{upload_id}</UploadId>"
                "</CompleteMultipartUploadResult>"
            ).encode()
            self._send_bytes(xml, status=HTTPStatus.OK, content_type="application/xml")
            return

        self._send_error(HTTPStatus.NOT_IMPLEMENTED, "Unsupported POST request")

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_status(self, status: HTTPStatus, extra_headers: dict[str, str] | None = None):
        self.send_response(status.value)
        headers = {"Server": "s3-emulator"}
        if extra_headers:
            headers.update(extra_headers)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

    def _send_error(self, status: HTTPStatus, message: str):
        print(f"Error {status}: {message}")
        self._send_bytes(message.encode(), status=status, content_type="text/plain")

    def _send_bytes(
        self,
        data: bytes,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "application/octet-stream",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status.value)
        headers = {
            "Server": "s3-emulator",
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        }
        if extra_headers:
            headers.update(extra_headers)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    # ------------------------------------------------------------------

    @staticmethod
    def _render_bucket_list(bucket: str, objects: list[str]) -> bytes:  # noqa: D401
        """Return a minimal XML listing of *objects* in *bucket*."""
        entries = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for key in objects:
            try:
                data = S3RequestHandler.server.state.get_object(bucket, key)  # type: ignore[attr-defined]
                size = len(data)
                etag = _etag(data)
            except Exception:  # noqa: BLE001
                size = 0
                etag = '""'
            entries.append(
                "<Contents>"
                f"<Key>{_escape_xml(key)}</Key>"
                f"<LastModified>{now}</LastModified>"
                f"<ETag>{etag}</ETag>"
                f"<Size>{size}</Size>"
                "</Contents>"
            )
        obj_elems = "".join(entries)
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<ListBucketResult>"
            f"<Name>{_escape_xml(bucket)}</Name>"
            f"{obj_elems}"
            "</ListBucketResult>"
        )
        return xml.encode()


# ----------------------------------------------------------------------
# Protocol bridging: allow type checker to access server attributes
# ----------------------------------------------------------------------


class S3ServerProtocol(Protocol):  # noqa: D101
    state: S3State
    auth: S3Auth


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------


def _escape_xml(text: str) -> str:  # noqa: D401
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _etag(data: bytes) -> str:  # noqa: D401
    return md5(data).hexdigest()
