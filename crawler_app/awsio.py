"""S3 + Lambda access for the Fasthouse SKU checker.

The checker Lambda reads a plain-text SKU list from S3 (one SKU per line)
and is fired by an EventBridge schedule. This module lets the Streamlit
app maintain that list and invoke the function on demand, without touching
the schedule.

Nothing here imports Streamlit, so it can be tested with a fake client.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

BACKUP_FOLDER = "backups"


@dataclass
class AwsConfig:
    """Connection details, read from Streamlit secrets (see
    .streamlit/secrets.toml.example)."""
    region: str = ""
    bucket: str = ""
    sku_key: str = ""                 # e.g. "inputs/fasthouse/SKUS.txt"
    lambda_arn: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    sku_prefix: str = ""              # where to look when picking the list in the UI
    backup_prefix: str = ""           # where previous versions are kept
    results_prefix: str = ""          # where the checker writes its output, optional
    lambda_payload: Dict = field(default_factory=dict)

    @property
    def has_credentials(self) -> bool:
        # Empty keys are valid too: boto3 then falls back to the instance
        # role / environment, which is how this would run inside AWS.
        return bool(self.region and self.bucket)

    @property
    def can_write_skus(self) -> bool:
        return bool(self.has_credentials and self.sku_key)

    @property
    def can_invoke(self) -> bool:
        return bool(self.has_credentials and self.lambda_arn)

    @property
    def function_name(self) -> str:
        """Short name for display; the ARN still works for the API call."""
        return self.lambda_arn.rsplit(":", 1)[-1] if self.lambda_arn else ""


# ---------------------------------------------------------------- SKU list

def parse_skus(text: str) -> List[str]:
    """One SKU per line; blanks, comments and duplicates removed, order kept."""
    seen = set()
    skus = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        sku = raw.strip()
        if not sku or sku.startswith("#"):
            continue
        if sku not in seen:
            seen.add(sku)
            skus.append(sku)
    return skus


def format_skus(skus: Sequence[str]) -> str:
    """Render the list back exactly as the Lambda expects to read it."""
    return "\n".join(skus) + ("\n" if skus else "")


def diff_skus(current: Sequence[str], new: Sequence[str]) -> Dict[str, List[str]]:
    current_set, new_set = set(current), set(new)
    return {
        "added": [s for s in new if s not in current_set],
        "removed": [s for s in current if s not in new_set],
        "unchanged": [s for s in new if s in current_set],
    }


def merge_skus(current: Sequence[str], new: Sequence[str]) -> List[str]:
    """Existing list plus anything new, original order preserved."""
    merged = list(current)
    known = set(current)
    for sku in new:
        if sku not in known:
            known.add(sku)
            merged.append(sku)
    return merged


# ---------------------------------------------------------------- AWS client

@dataclass
class S3Object:
    key: str
    size: int = 0
    last_modified: Optional[str] = None


class SkuCheckerClient:
    """Thin wrapper over the two AWS calls this app makes.

    `s3` and `lambda_` can be injected for tests; otherwise boto3 clients
    are built from the config.
    """

    def __init__(self, config: AwsConfig, s3=None, lambda_=None) -> None:
        self.config = config
        self._s3 = s3
        self._lambda = lambda_

    # -- clients -------------------------------------------------------
    def _client(self, service: str):
        import boto3

        kwargs = {"region_name": self.config.region}
        if self.config.access_key_id and self.config.secret_access_key:
            kwargs["aws_access_key_id"] = self.config.access_key_id
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        return boto3.client(service, **kwargs)

    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = self._client("s3")
        return self._s3

    @property
    def lambda_(self):
        if self._lambda is None:
            self._lambda = self._client("lambda")
        return self._lambda

    # -- S3 ------------------------------------------------------------
    def list_objects(self, prefix: str, suffix: str = "", limit: int = 200) -> List[S3Object]:
        paginator = self.s3.get_paginator("list_objects_v2")
        found: List[S3Object] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix or ""):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/") or (suffix and not key.lower().endswith(suffix.lower())):
                    continue
                modified = item.get("LastModified")
                found.append(S3Object(
                    key=key,
                    size=int(item.get("Size", 0)),
                    last_modified=modified.strftime("%Y-%m-%d %H:%M UTC") if modified else None,
                ))
                if len(found) >= limit:
                    return found
        return found

    def read_text(self, key: str) -> Tuple[str, S3Object]:
        response = self.s3.get_object(Bucket=self.config.bucket, Key=key)
        body = response["Body"].read().decode("utf-8", errors="replace")
        modified = response.get("LastModified")
        meta = S3Object(
            key=key,
            size=int(response.get("ContentLength", len(body))),
            last_modified=modified.strftime("%Y-%m-%d %H:%M UTC") if modified else None,
        )
        return body, meta

    def write_text(self, key: str, text: str, content_type: str = "text/plain") -> None:
        self.s3.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType=content_type,
        )

    def backup_key_for(self, key: str, stamp: Optional[str] = None) -> str:
        """Where the previous version of `key` is copied.

        `backup_prefix` should point OUTSIDE the folder the Lambda reads,
        so old copies can never be mistaken for input. Without it, a
        `backups/` sub-folder beside the file is used.
        """
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        head, _, name = key.rpartition("/")
        base, dot, ext = name.rpartition(".")
        if not dot:
            base, ext = name, "txt"
        backup_name = f"{base}.{stamp}.{ext}"

        prefix = (self.config.backup_prefix or "").strip()
        if prefix:
            return prefix.rstrip("/") + "/" + backup_name
        return f"{head}/{BACKUP_FOLDER}/{backup_name}" if head else f"{BACKUP_FOLDER}/{backup_name}"

    def update_sku_list(self, key: str, skus: Sequence[str], backup: bool = True) -> Dict:
        """Write the list, keeping a timestamped copy of what was there.

        The backup is best-effort: a missing current file (first run) is not
        an error, but a failure to *write* the backup aborts the update so a
        good list is never replaced without a copy.
        """
        result: Dict = {"key": key, "count": len(skus), "backup_key": None}
        if backup:
            try:
                previous, _ = self.read_text(key)
            except Exception as e:  # noqa: BLE001 - no current file is fine
                log.info("No current SKU list to back up at %s (%s)", key, e)
                previous = None
            if previous is not None:
                backup_key = self.backup_key_for(key)
                self.write_text(backup_key, previous)
                result["backup_key"] = backup_key
        self.write_text(key, format_skus(skus))
        return result

    # -- Lambda --------------------------------------------------------
    def invoke_checker(self, payload: Optional[Dict] = None, timeout_note: str = "") -> Dict:
        """Run the checker now. Returns status, any function error and the
        (truncated) response body."""
        started = time.time()
        kwargs = {
            "FunctionName": self.config.lambda_arn,
            "InvocationType": "RequestResponse",
        }
        body = payload if payload is not None else (self.config.lambda_payload or None)
        if body:
            kwargs["Payload"] = json.dumps(body).encode("utf-8")

        response = self.lambda_.invoke(**kwargs)
        raw = response.get("Payload")
        text = ""
        if raw is not None:
            data = raw.read()
            text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)

        return {
            "status_code": int(response.get("StatusCode", 0)),
            "function_error": response.get("FunctionError"),
            "payload": text,
            "elapsed": time.time() - started,
            "note": timeout_note,
        }

    # -- connection test ----------------------------------------------
    def check_access(self) -> List[Tuple[str, bool, str]]:
        """Probe each permission the app needs; returns (label, ok, detail)."""
        checks: List[Tuple[str, bool, str]] = []

        # Listed with the prefix rather than head_bucket: head_bucket sends
        # no s3:prefix, so an IAM policy that scopes ListBucket to a prefix
        # would deny it even though every operation the app needs is allowed.
        prefix = self.config.sku_prefix or (self.config.sku_key.rpartition("/")[0] + "/")
        try:
            self.s3.list_objects_v2(Bucket=self.config.bucket, Prefix=prefix, MaxKeys=1)
            checks.append(("S3 bucket reachable", True, f"{self.config.bucket}/{prefix}"))
        except Exception as e:  # noqa: BLE001
            checks.append(("S3 bucket reachable", False, str(e)))
            return checks

        if self.config.sku_key:
            try:
                _, meta = self.read_text(self.config.sku_key)
                checks.append(("SKU list readable", True, f"{self.config.sku_key} ({meta.last_modified})"))
            except Exception as e:  # noqa: BLE001
                checks.append(("SKU list readable", False, str(e)))

        if self.config.lambda_arn:
            try:
                info = self.lambda_.get_function_configuration(FunctionName=self.config.lambda_arn)
                checks.append((
                    "Lambda visible", True,
                    f"{info.get('FunctionName')} (timeout {info.get('Timeout')}s)",
                ))
            except Exception as e:  # noqa: BLE001
                checks.append(("Lambda visible", False, str(e)))

        return checks
